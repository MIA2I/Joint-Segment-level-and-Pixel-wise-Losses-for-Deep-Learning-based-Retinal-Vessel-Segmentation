function [ IDMask, RefThickness ] = GenerateIDMask( RefVessels, Mask )

%Hyperparameters
Levels = 2;
minLength = 4; % the predefined minimum length of the skeleton segment
maxLength = 8; % the predefined maximum length of the skeleton segment

%Initialization
Mask(Mask>0) = 1;
[height, width] = size(RefVessels);
RefVessels = uint8(RefVessels);
RefVessels(RefVessels>0) = 1;
RefSkeleton = bwmorph(RefVessels,'thin',inf);

% Quantitize the thickness of each pixel
[ RefThickness, RefminRadius, RefmaxRadius ] = CalcThickness( RefSkeleton, RefVessels);

% Generate the searching radius of each pixel
SearchingRadius = RefThickness + 2;
SearchingRadius(RefSkeleton==0) = 0;

% Gnerate the searching range of each skeleton pixel
SearchingMask = GenerateRange(SearchingRadius, Mask);

% Segment the target skeleton map
[ SegmentID ] = SegmentSkeleton( RefSkeleton, minLength, maxLength );
SegmentID(Mask==0) = 0;

% Calculate the skeletal similarity for each segment
IDMask = zeros(size(SearchingMask),'double');
for Index = 1:max(max(SegmentID))
    
    SegmentRadius = SearchingRadius;
    SegmentRadius(SegmentID~=Index) = 0;
    SegmentMask = GenerateRange(SegmentRadius, Mask);

    IDMask(SegmentMask>0) = Index;
    IDMask(SegmentID==Index) = -Index;
end
