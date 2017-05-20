function [ rescaled_img ] = fov_rescale( q_img, q_fov, pano_width, out_size )
%fov_rescale
%   This function rescales the input q_img according to given field-of-view
%   q_fov (in degrees) and theoretical size of the full 360deg panorama 
%   pano_width (in px).
%   The image is rescaled according to the FOV so that the image is sized
%   relatively with respect to the panorama width.
%   Furthermore, the image is cropped or padded to maintain the defined
%   out_size.
%   In case that the resulting image is bigger than out_size, the image is 
%   cropped. In case that the resulting image is smaller than the out_size,
%   the image is padded on both sides with black.

    assert(q_fov < pi, 'Field-of-view cannot be larger than PI.');
    
    q_img_size = size(q_img);
    nw = (q_fov / (2*pi)) * pano_width;
    scale = nw / q_img_size(2);
    image = imresize(q_img, scale);

    newsize =  out_size;
    oldsize = size(image);
    delta = newsize - oldsize(1:2);
    delta_2 = floor(double(delta)/2.0);
    offset = delta - 2*delta_2;
    if (delta(1) < 0)
        %crop y
        image = image(max(-delta_2(1)-offset(1), 1):oldsize(1)+delta_2(1)-1, :, :);
        delta(1) = 0;
    end
    if (delta(2) < 0)
        %crop x
        image = image(:, max(-delta_2(2)-offset(2), 1):oldsize(2)+delta_2(2)-1, :);
        delta(2) = 0;
    end
    delta_2 = floor(double(delta)/2.0);
    fixed_img = image;
    if (delta(1) >= 0 && delta(2) >= 0)
        fixed_img = padarray(image, delta_2);
    end
    rescaled_img = imresize(fixed_img, newsize); %security
end

