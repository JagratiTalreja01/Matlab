imageRGB = imread("Original_Image/monarch_x4_DBPN.png");
imageRGB = im2double(imageRGB);
imshow(imageRGB)
title("Monarch Image")

noisyRGB = imnoise(imageRGB,"gaussian",0,0.01);
%imshow(noisyRGB)
title("Noisy Image")

[noisyR,noisyG,noisyB] = imsplit(noisyRGB);

net = denoisingNetwork("dncnn");

denoisedR = denoiseImage(noisyR,net);
denoisedG = denoiseImage(noisyG,net);
denoisedB = denoiseImage(noisyB,net);

denoisedRGB = cat(3,denoisedR,denoisedG,denoisedB);
%imshow(denoisedRGB)
montage({imageRGB, noisyRGB, denoisedRGB})
title("Denoised Image")

