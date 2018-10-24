function [ Thickness, minRadius, maxRadius ] = CalcThickness( Skeleton, Guidance)
% Calculating the vessel thickness of each pixel in the skeleton map
% Input:  Skeleton: the extracted skeleton map
%         Guidance: the original manual annotation
% Output: Thickness: an array, which instores the vessel thicness of each
%         pixel in the skeleton map.
[height, width] = size(Skeleton);
Thickness = zeros(size(Skeleton), 'double');
minRadius = 100;
maxRadius = 0;
for x = 1:height
    for y = 1:width
        if(Skeleton(x, y) > 0)
            top = max(x - 10, 1);
            bottom = min(x + 10, height);
            left = max(y - 10, 1);
            right = min(y + 10, width);
            r = FindNearest(Guidance(top:bottom, left:right), x - top + 1, y - left + 1 );
            Thickness(x,y) = r;
            if (r < minRadius)
                minRadius = r;
            end
            if (r > maxRadius)
                maxRadius = r;
            end
        end
    end
end


function [ Radius ] = FindNearest( SearchingWindow, x, y )
% Finding the minimum radius of the pixel (x,y) in the searching window.
% Input:  SearchingWindow, the searching range
%         (x,y) the coordinates of the centeral pixel
% Output: the minimum distance of the target pixel in the searching window,
%         which can be regarded as the radius of the maximum inscrobed
%         circle centered at the pixel (x,y).
[height, width] = size(SearchingWindow);
Radius = 100;
for h = 1:height
    for w = 1:width
        if(SearchingWindow(h, w) > 0)
            continue;
        else
            r = sqrt(double((h - x)^2 + (w - y)^2));
            if (r < Radius)
                Radius = r;
            end
        end
    end
end