I = imread('Original_Image/KuroidoGankax8_LRBI.png');; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Crop = I(42.5:62.5,5:25,:); %first number is heightLast number is width
figure;
subplot(211);imshow(I);
axis off
subplot(212);imshow(Crop)
axis off
imwrite(Crop,'Patches_Created/KuroidoGankax8_LRBI.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CALCULATE THE PSNR AND SSIM
GT = imread('Patches_Created/KuroidoGankax8_LRBI.png'); 
ModelImg = imread('Patches_Created/KuroidoGankax8_LRBI.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psnr = compute_psnr(GT,ModelImg);
fprintf('The PSNR value is %0.2f.\n', psnr);
[ssimval, ssimmap] = ssim(GT,ModelImg);  
fprintf('The SSIM value is %0.3f.\n',ssimval);