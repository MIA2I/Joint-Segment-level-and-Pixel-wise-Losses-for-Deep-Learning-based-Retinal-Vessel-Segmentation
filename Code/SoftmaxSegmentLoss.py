import sys
sys.path.append('/home/john/caffe-master/python/')

import caffe
import bwmorph as bw
import numpy as np
from numpy import linalg as LA
 
class SoftmaxSegmentLossLayer(caffe.Layer):
    """
    Compute the Softmax Loss in the same manner but use the skeletal loss as weights
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need 3 inputs to compute distance.")
 
    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # weights matrix
        self.mask = np.zeros_like(bottom[0].data, dtype=np.float32)
        # weights matrix based on segment-level thickness similarity
        self.SLmask = np.ones_like(bottom[1].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
 
    def forward(self, bottom, top):        
        
        # run softmax() layer
        score = np.copy(bottom[0].data)
        
        temp = np.maximum(score[0,0,:,:], score[0,1,:,:])
        score[0,0,:,:] -= temp
        score[0,1,:,:] -= temp
        
        prob = np.exp(score)
        
        temp = prob[0,0,:,:] + prob[0,1,:,:]
        prob[0,0,:,:] /= temp
        prob[0,1,:,:] /= temp
        
	# read the label matrix
        label = np.copy(bottom[1].data[0,0,:,:])
        
        # generate pixel-wise matrix
	# assign wights to different labels to deal with the imbalance problem: 
	#							0.1 for 0 (non-vessel)
	#							0.9 for 1 (vessel)
        label = np.copy(bottom[1].data)
        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = 0.1
        temp[np.where(label==255)] = 0
        self.mask[0,0,:,:] = np.copy(temp)
        temp[...] = 0
        temp[np.where(label>0)] = 0.9
        temp[np.where(label==255)] = 0
        self.mask[0,1,:,:] = np.copy(temp)
        count = np.count_nonzero(self.mask)
        
	# calcl segment-level loss
        IDMask = np.copy(bottom[2].data[0,0,:,:])
        self.SLmask[0,0,:,:] =  2.0 - self.SegmentLoss(prob, label, IDMask)

        #weights: combination of self.mask and self.SLmask
        weights = self.mask * self.SLmask
        
        # calculate loss
        probs = np.copy(prob)
        probs[np.where(probs<1.175494e-38)] = 1.175494e-38
        logprob = -np.log(probs)
        
        data_loss = np.sum(weights*logprob) *1.0 / count
        
        self.diff[...] = np.copy(prob)
        top[0].data[...] = np.copy(data_loss)
 
    def backward(self, top, propagate_down, bottom):
        
        delta = np.copy(self.diff[...])
        
        count = np.count_nonzero(self.mask)
        
        delta[np.where(self.mask>0)] -= 1
        
        # generate pixel-wise matrix
	# Re-assign wights to different labels: 
	#				0.2 for 0 (non-vessel out of searching range)
	#				0.35 for 0 (non-vessel in searching range)				
	#				0.8 for 1 (vessel)
        label = np.copy(bottom[1].data)
        Range = np.copy(np.absolute(bottom[2].data))
        Range[np.where(label==1)] = 0
        mask = np.copy(bottom[0].data)
        temp = np.copy(label)
        temp[...] = 0
        temp[np.where(label==0)] = 0.2
        temp[np.where(np.absolute(Range)>0)] = 0.35
        temp[np.where(label>0)] = 0.8
        temp[np.where(label==255)] = 0
        mask[0,0,:,:] = np.copy(temp)
        mask[0,1,:,:] = np.copy(temp)
        
        #weights: combination of self.mask and self.SLmask
        weights = mask * self.SLmask
        
        delta *= weights
        bottom[0].diff[...] = delta * 1.0 / count
        
    def SegmentLoss(self, prob, label, IDMask):
        
	# convert input probability map into binary image (hard segmentation)
        img = np.copy(prob[0,1,:,:])
        img[np.where(img<0.5)] = 0
        img[np.where(img>0)] = 1
        
	# calculate the skeletons of the binary image
        imgtemp = np.copy(img)
        skel = bw.bwmorph_thin(imgtemp)
        skel = skel.astype(float)
        skel *= IDMask
        
	# denote vessel pixels in the segmentation map located outside the searching range as outliers
        outlier = np.copy(img)
        outlier[np.where(np.absolute(IDMask)>0)] = 0
        outlier[np.where(label==1)] = 0
        outlier[np.where(label==255)] = 0
        
	# segment-level thickness similarity matrix
        Similarity = np.ones_like(IDMask, dtype=np.float)
        Similarity[np.where(outlier>0)] = 0
        
        image = np.copy(prob[0,1,:,:])
	# For the i-th segment in IDMask (segment index starts from 1):
	#			-i: the pixel values of the skeleton segment
	#			i: the searching range of the i-th segment
        LocationsSrc = {index: np.where(np.absolute(skel)==index+1) for index in range(np.amax(IDMask))}
        LocationsRef = {index: np.where(IDMask==-(index+1)) for index in range(np.amax(IDMask))}
	
	# iteratively compare each pair of vessel segments
        for index in range(np.amax(IDMask)):
            
            SrcX = LocationsSrc[index][0]
            SrcY = LocationsSrc[index][1]
            RefX = LocationsRef[index][0]
            RefY = LocationsRef[index][1]
            
            AvgRefThickness = 0
            
            if np.size(RefX) > 0:
                AvgRefThickness = np.sum(label[np.where(np.absolute(IDMask)==index+1)]) * 1.0 / np.size(RefX)
            else:
                continue
                
            AvgSrcThickness = 0
            
            if np.size(SrcX) > 0.8 * np.size(RefX):
                AvgSrcThickness = np.sum(img[np.where(np.absolute(IDMask)==index+1)]) * 1.0 / np.size(SrcX) 
                
            if AvgRefThickness > 0:
                Similarity[np.where(np.absolute(IDMask)==index+1)] = 1.0 - np.absolute(AvgSrcThickness-AvgRefThickness)  / AvgRefThickness
            else:
                Similarity[np.where(np.absolute(IDMask)==index+1)] = 0
            
        return Similarity

