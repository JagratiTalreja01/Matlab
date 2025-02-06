I = imread('Original_Image/baby_Generated_By_Your_Method.png');; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Crop = I(150:250,150:250,:); %first number is heightLast number is width
figure;
subplot(211);imshow(I);
axis on
subplot(212);imshow(Crop)
axis on
imwrite(Crop,'Patches_Created/baby_Generated_By_Your_Method.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATE THE PSNR AND SSIM
GT = imread('Patches_Created/baby_GT.png'); 
ModelImg = imread('Patches_Created/baby_Generated_By_Your_Method.png');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psnr = compute_psnr(GT,ModelImg);
fprintf('The PSNR value is %0.2f.\n', psnr);
[ssimval, ssimmap] = ssim(GT,ModelImg);  
fprintf('The SSIM value is %0.3f.\n',ssimval);