original_image = imread('Original_Image/baby_HR.png');

if size(original_image, 3) == 3
    original_image = rgb2gray(original_image);
end

noisy_image = imnoise(original_image, 'gaussian', 0, 0.00);

net = denoisingNetwork('DnCNN');
denoised_image = denoiseImage(noisy_image, net);


montage({original_image, noisy_image, denoised_image})
title('First is the original image, second is the noised image, third is the denoised image')