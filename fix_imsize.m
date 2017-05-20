function [ fixed_img ] = fix_imsize( image, width, height )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    
    newsize = [width height];
    oldsize = size(image);
    delta = newsize - oldsize(1:2);
    delta_2 = floor(double(delta)/2.0);
    offset = delta - 2*delta_2;
    if (delta(1) < 0)
        %crop y
        image = image(max(-delta_2(1)-offset(1), 1):oldsize(1)+delta_2(1)-1, :, :);
    end
    if (delta(2) < 0)
        %crop x
        image = image(:, max(-delta_2(2)-offset(2), 1):oldsize(2)+delta_2(2)-1, :);
    end
    fixed_img = image;
    if (delta(1) > 0 && delta(2) > 0)
        fixed_img = padarray(image, delta_2);
    end
    fixed_img = imresize(fixed_img, newsize); %security
end

