clear; close all; clc;

[file,path] = uigetfile({'*.png'},'Select a file',[pwd,'\HR\']);


if file==0
    fprintf('No file\n')
else
scale_all=extractBetween(path,length(path)-1,length(path)-1);
dataset = extractBetween(path,'HR\','\x');    
    
% position=[
% 3 1
% 3 2
% ];
% scale_all=3;
image_name=file;
% dataset = {'Set5'};
% dataset = {'Set5','Set14','B100','Urban100','Manga109'};
I=imread(fullfile(path,image_name));



while(1)
figure(1);
[J,rect] = imcrop(I);

if rect==0
    fprintf('No Slect File')
    break;
end    
patch_save='Result\';
%
%methods = {'VDSR','IMDN','LapSRN','AWSRN','CARN','rc20r2rg1nup','MSRN','RCAN'};
methods = {'AWSRN'};


ext = {'*.jpg', '*.png', '*.bmp'};
num_method = length(methods);
num_set = length(dataset);

%
scale=str2num(scale_all{1});
idx_set=1;
idn=1;
% for i=1:1:length(position)
% 
%     p=position(i,:)
    ud=rect(2);
    lr=rect(1);
%     %postion patch
%     h=(1+hb*(ud-1)):(hb+hb*(ud-1));       
%     w=(1+wb*(lr-1)):(wb+wb*(lr-1));
%     h=(1+hb*ud):(hb+hb*ud);       
%     w=(1+wb*lr):(wb+wb*lr);
    name_HR=image_name;%\HR\B100\x4\img093_HR_x4.png
cnt_c='A'    ;
id_plot=3;
    for idx_method = 1:num_method
        idm=1;
                    filepaths = [];
                    for idx_ext = 1:length(ext)
                        filepaths = cat(1, filepaths, dir(fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)], ext{idx_ext})));
                    end
%                     image_name=image_name_all{idn};
%                     name_HR=image_name_all{idn};%\HR\B100\x4\img093_HR_x4.png

%                   idx_im= strfind(filepaths.name,image_name);
                    im_HR = imread(fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)], name_HR));
                    
                    if strcmp(methods{idx_method},'VDSR')
                        %\VDSR\VDSR_Set14_x4\img013.png
                        if strcmp(dataset{idx_set},'Manga109') || scale==8
                           continue;
                        end 
                        name_SR = strrep(name_HR, ['_HR_x', num2str(scale)],'');
                        if strcmp(dataset{idx_set},'Set5')
                            name_SR = strrep(name_HR, ['_HR_x', num2str(scale)],'_GT');
                        end
                        im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},'_',dataset{idx_set},'_x',num2str(scale)], name_SR));
                    elseif strcmp(methods{idx_method},'IMDN')
                        %\IMDN\IMDN_x4\Urban100\x4
                        if scale==8
                           continue;
                        end 
                        name_SR = strrep(name_HR, 'img', 'img_');
                        name_SR = strrep(name_SR, ['_HR_x', num2str(scale)], ['x',num2str(scale)]);
                        im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},['_x', num2str(scale)],'/',dataset{idx_set},'/',['x', num2str(scale)]], name_SR));
                    elseif strcmp(methods{idx_method},'LapSRN')
                        %\LapSRN\Urban100\x4\img_001_LapSRN.png
                        name_SR = strrep(name_HR, 'img', 'img_');
                        name_SR = strrep(name_SR, ['HR_x', num2str(scale)], [methods{idx_method}]);
                        im_SR = imread(fullfile([methods{idx_method},'/',dataset{idx_set},'/',['x', num2str(scale)]], name_SR));
                    elseif strcmp(methods{idx_method},'CARN')
                        %CARN\CARN\Urban100\x4\SR\img_001_SRF_4_SR.png
                        if strcmp(dataset{idx_set},'Manga109')|| scale==8
                           continue;
                        end
                        for idx_im = 1:length(filepaths)
                            if strfind(filepaths(idx_im).name,image_name)
                                idx_im=idx_im;
                                break;
                            end
                        end
                        if strcmp(dataset{idx_set},'Urban100')
                            name_SR = strrep(name_HR, 'mg', 'mg_');
                            name_SR = strrep(name_SR, ['HR_x',num2str(scale)], ['SRF_',num2str(scale),'_SR']);
                        else
                            name_SR = ['img_',sprintf('%03d',idx_im),'_SRF_',num2str(scale),'_SR.png'];
                        end
                        im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},'/',dataset{idx_set},['/x', num2str(scale)],'/SR'], name_SR));
                    elseif strcmp(methods{idx_method},'rc20r2rg1nup')
                        %\rc20r2rg1nup\results-B100\3096_x2_SR.png
                        name_SR = strrep(name_HR, ['HR_x', num2str(scale)], ['x',num2str(scale),'_SR']);
                        im_SR = imread(fullfile([methods{idx_method},'/results-',dataset{idx_set}], name_SR));
                    elseif strcmp(methods{idx_method},'AWSRN')
                        %\AWSRN\AWSRN\x4\Urban100\img001_x4_SR.png
                        if strcmp(dataset{idx_set},'Manga109')
                           continue;
                        end
                        if scale==8
                            name_SR = strrep(name_HR, ['HR_x', num2str(scale)], ['x',num2str(scale),'_SR']);
                            im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},'/',['x', num2str(scale)],'/',dataset{idx_set}], name_SR));
                        else
                            name_SR = strrep(name_HR, ['HR_x', num2str(scale)], ['x',num2str(scale),'_SR']);
                            im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},'/',['x', num2str(scale)],'/',dataset{idx_set}], name_SR));
                        end
                    elseif strcmp(methods{idx_method},'MSRN')
                        %MSRN\MSRN\B100\x4\img001_MSRN_x4.png
                        name_SR = strrep(name_HR, ['HR_x', num2str(scale)], [methods{idx_method},'_x',num2str(scale)]);
                        im_SR = imread(fullfile([methods{idx_method},'/',methods{idx_method},'/',dataset{idx_set}],'/',['x', num2str(scale)], name_SR));
                    elseif strcmp(methods{idx_method},'RCAN')
                        %MSRN\MSRN\B100\x4\img001_MSRN_x4.png
                        
                        if strcmp(dataset{idx_set},'Urban100')
                            name_HR1 = strrep(name_HR, 'img', 'img_');
                            name_SR = strrep(name_HR1,['HR_x', num2str(scale)], [methods{idx_method},'_x',num2str(scale)]);
                            im_SR = imread(fullfile([methods{idx_method},'/',dataset{idx_set}],'/',['x', num2str(scale)], name_SR));
                        else
                            name_SR = strrep(name_HR, ['HR_x', num2str(scale)], [methods{idx_method},'_x',num2str(scale)]);
                            im_SR = imread(fullfile([methods{idx_method},'/',dataset{idx_set}],'/',['x', num2str(scale)], name_SR));
                        end
                        
                    end

        

        if 3 == size(im_HR, 3)
            im_HR_YCbCr = single(rgb2ycbcr(im2double(im_HR)));
            im_HR_Y = im_HR_YCbCr(:,:,1);
            im_SR_YCbCr = single(rgb2ycbcr(im2double(im_SR)));
            im_SR_Y = im_SR_YCbCr(:,:,1);
        else
            im_HR_Y = single(im2double(im_HR));
            im_SR_Y = single(im2double(im_SR));
        end
        i=0;
        w=round(rect(1):rect(1)+rect(3));
        h=round(rect(2):rect(2)+rect(4));
        p_HR=im_HR_Y(h,w,:);         
        p_SR=im_SR_Y(h,w,:); 
        [PSNR_im, SSIM_im] = Cal_Y_PSNRSSIM(p_HR*255, p_SR*255, scale, scale);
        fprintf('x%d %s %s: PSNR= %f SSIM= %f\n', scale,methods{idx_method}, name_SR, PSNR_im, SSIM_im);

        % image image_name='img016_HR_x4.png';
        name_save0 = strrep(name_HR, '_HR.png','');
        name_hr = [patch_save,'x',num2str(scale),'_',dataset{idx_set},'_',...
            name_save0, sprintf('-P%d_HR_-lr%d_ud%d-HR.bmp',i,lr,ud)];
        name_sr = [patch_save,'x',num2str(scale),'_',dataset{idx_set},'_',...
            name_save0, sprintf('-P%d_M%d_%s-lr%d_ud%d.bmp',i,idx_method,methods{idx_method},lr,ud)];

        imwrite(uint8(im_HR(h,w,:)),name_hr,'BMP');
        imwrite(uint8(im_SR(h,w,:)),name_sr,'BMP');

%         if strcmp(dataset{idx_set},'Urban100')
%             name_LR = strrep(name_HR, ['_HR'_],'');
%         else
        name_LR = strrep(name_HR, '_HR_','');
%         end
        im_lr = imread(fullfile('./LR/LRBI/', dataset{idx_set}, ['x', num2str(scale)], name_LR));
        im_lr_up = imresize(im_lr,scale,"bicubic");
        name_lr_full = [patch_save,'x',num2str(scale),'_',dataset{idx_set},'_',...
                name_save0, sprintf('-P%d_LR_-lr%d_ud%d-lr.bmp',i,lr,ud)];
        % image
        imwrite(im_lr_up(h,w,:),name_lr_full,'BMP');
    
    
        bboxes=[w(1) h(1)  rect(3) rect(4)];
        im_HR_d=im_HR;
        im_HR   = insertShape(im_HR,'Rectangle',bboxes,'LineWidth',5);
        name_hr_full = [patch_save,'x',num2str(scale),'_',dataset{idx_set},'_',...
                name_save0, sprintf('-P%d_HR_-lr%d_ud%d-HRFull.bmp',i,lr,ud)];
        % image
        imwrite(im_HR,name_hr_full,'BMP');
        figure(2),
        subplot(2,6,1);imshow(im_HR_d(h,w,:));title('HR')
        subplot(2,6,2);imshow(im_lr_up(h,w,:));title('LR')
        title_a=sprintf('x%d %s %.3f %.4f', scale,methods{idx_method}, PSNR_im, SSIM_im);
        subplot(2,6,id_plot);imshow(im_SR(h,w,:));title(title_a)
        subplot(2,6,id_plot+1);imshow(im_HR);title(sprintf('x%d %s', scale,name_save0))
        id_plot=id_plot+1;
        
        filename = string(append(name_hr_full,'.xlsx'));
        writematrix(['psnr_',methods{idx_method}],  filename,'Sheet','data','Range',[cnt_c, '1']);
        writematrix(['ssim_',methods{idx_method}],  filename,'Sheet','data','Range',[char(cnt_c+1), '1']);
        idm=idm+1;
        writematrix(PSNR_im,filename,'Sheet','data','Range',[char(cnt_c),'2:',char(cnt_c),num2str(idm)]);
        writematrix(SSIM_im,filename,'Sheet','data','Range',[char(cnt_c+1),'2:',char(cnt_c+1),num2str(idm)]);
        cnt_c=char(cnt_c+2);
    end
%         figure(1),imshow(im_HR);title(sprintf('x%d %s', scale,name_save0));
end
% end
end
function [psnr_cur, ssim_cur] = Cal_Y_PSNRSSIM(A,B,row,col)
% shave border if needed
if nargin > 2
    [n,m,~]=size(A);
    A = A(row+1:n-row,col+1:m-col,:);
    B = B(row+1:n-row,col+1:m-col,:);
end
% RGB --> YCbCr
if 3 == size(A, 3)
    A = rgb2ycbcr(A);
    A = A(:,:,1);
end
if 3 == size(B, 3)
    B = rgb2ycbcr(B);
    B = B(:,:,1);
end
% calculate PSNR
A=double(A); % Ground-truth
B=double(B); %

e=A(:)-B(:);
mse=mean(e.^2);
psnr_cur=10*log10(255^2/mse);

% calculate SSIM
[ssim_cur, ~] = ssim_index(A, B);
end


function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 || nargin > 5)
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

[M N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);	%
    K(1) = 0.01;								      % default settings
    K(2) = 0.03;								      %
    L = 255;                                  %
end

if (nargin == 3)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 4)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 5)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end
