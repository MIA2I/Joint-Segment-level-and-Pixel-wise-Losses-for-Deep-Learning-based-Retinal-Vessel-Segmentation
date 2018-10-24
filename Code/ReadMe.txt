The foler contains the following files:

1. data_layers.py: The python layer to load data (image, label, IDMask). The IDMask matrix contains the index and the searching
                   range of skeleton segments (index starts from 1). For the i-th segment, the skeleton pixels are with value i,
                   while the pixels of the searching range are with the value i. The sub-folder GenerateIDMask contains the 
                   matlab source code to generate IDMask matrix from each manual annotation.
2. SoftmaxSegmentLoss.py: The defined segment-level loss is implemented by using the segment-level thickness similarity as the 
                   weights of the cross entropy loss.
3. bwmorph.py: The python function to calculate the skeletons from the manually annotated vessels.
4. deploy.prototxt, train_val.prototxt and solver.prototxt: The definitions of the model and the training prarameters.
