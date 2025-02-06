I = imread('Original_Image/baby_GT.png');
Crop = I(150:250,150:250,:); %first number is heightLast number is width
figure;
subplot(211);imshow(I);
axis on
subplot(212);imshow(Crop);
axis on 
imwrite(Crop,'Patches_Created/baby_GT.png');%SAVE THE MODEL PATCH IMAGE%%%%%%%%%
