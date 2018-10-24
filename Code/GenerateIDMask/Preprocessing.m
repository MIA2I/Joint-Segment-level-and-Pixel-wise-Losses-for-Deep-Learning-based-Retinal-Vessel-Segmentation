clc
close all;
clear all;

num = {'21';'22';'23';'24';'25';'26';'27';'28';'29';'30';'31';'32';'33';'34';'35';'36';'37';'38';'39';'40'};

image_index = 0;
for index = 1:20
    
    image = zeros(640,640,'uint8');
    label = zeros(640,640,'uint8');
    mask = zeros(640,640,'uint8');
    
    img = imread(strcat('image/', num{index}, '_training.tif'));
    msk = imread(strcat('mask/', num{index}, '_training_mask.gif'));
    lab = imread(strcat('1st_manual/', num{index}, '_manual1.gif'));
    
    image(28:611,35:599,:) = img(:,:,:);
    label(28:611,35:599) = lab;
    mask(28:611,35:599) = msk;
    
    for x = 0:12
        for y = 0:12
            
            image_crop = image(1+48*x:64+48*x,1+48*y:64+48*y,:);
            label_crop = label(1+48*x:64+48*x,1+48*y:64+48*y);
            mask_crop = mask(1+48*x:64+48*x,1+48*y:64+48*y);
            
            if (nnz(mask_crop)/64/64 < 0.5)
                continue;
            end
            
            image_num = int2str(image_index);
            
            imwrite(image_crop, strcat('Image/', image_num, '.png'));
            imwrite(label_crop, strcat('Label/', image_num, '.png'));
            imwrite(mask_crop, strcat('Mask/', image_num, '.png'));
            
            image_index = image_index + 1;
            
            image_num = int2str(image_index);
            
            image_lr = fliplr(image_crop);
            label_lr = fliplr(label_crop);
            mask_lr = fliplr(mask_crop);
            
            imwrite(image_lr, strcat('Image/', image_num, '.png'));
            imwrite(label_lr, strcat('Label/', image_num, '.png'));
            imwrite(mask_lr, strcat('Mask/', image_num, '.png'));
            
            image_index = image_index + 1;
            
            image_num = int2str(image_index);
            
            image_ud = flipud(image_crop);
            label_ud = flipud(label_crop);
            mask_ud = flipud(mask_crop);
            
            imwrite(image_ud, strcat('Image/', image_num, '.png'));
            imwrite(label_ud, strcat('Label/', image_num, '.png'));
            imwrite(mask_ud, strcat('Mask/', image_num, '.png'));
            
            image_index = image_index + 1;
            
            image_num = int2str(image_index);
            
            image_noise = imnoise(image_crop,'gaussian',0,0.001);
            
            imwrite(image_noise, strcat('Image/', image_num, '.png'));
            imwrite(label_crop, strcat('Label/', image_num, '.png'));
            imwrite(mask_crop, strcat('Mask/', image_num, '.png'));
            
            image_index = image_index + 1;
            
            image_num = int2str(image_index);
            
            image_clahe = adapthisteq(image_crop);
            
            imwrite(image_clahe, strcat('Image/', image_num, '.png'));
            imwrite(label_crop, strcat('Label/', image_num, '.png'));
            imwrite(mask_crop, strcat('Mask/', image_num, '.png'));
            
            image_index = image_index + 1;
            
            image_num = int2str(image_index);
            
            image_brightness = imadjust(image_crop);
            
            imwrite(image_brightness, strcat('Image/', image_num, '.png'));
            imwrite(label_crop, strcat('Label/', image_num, '.png'));
            imwrite(mask_crop, strcat('Mask/', image_num, '.png'));
            
            image_index
            
        end
    end
    
end

