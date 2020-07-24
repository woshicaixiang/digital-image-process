clear
clc

bayer=[0 32 8 40 2 34 10 42;
       48 16 56 24 50 18 58 26;
       12 44 4 36 14 46 6 38;
       60 28 52 20 62 30 54 22;
       3 35 11 43 1 33 9 41;
       51 19 59 27 49 17 57 25;
       15 47 7 39 13 45 5 37;
       63 31 55 26 61 29 53 21];
 bayer=bayer';%bayer抖动矩阵
 
  image=imread('D:\project1_digital_image_process\cai_3.jpg');%读入图像
  image_gray=double(rgb2gray(image));%转化为灰度图像
  [r,c]=size(image_gray);
  image_bayer=zeros(r,c);%建立抖动后矩阵
  figure(1)
  imshow(rgb2gray(image));%显示原图
  
  for i=1:r
      for j=1:c
      im=mod(i,8)+1;%寻找在模版中的行位置
      jm=mod(j,8)+1;%寻找在模版中的列位置
      
      nowpoint=image_gray(i,j)/256*64;%将灰度转化到[0,64]
      
      if(nowpoint<=bayer(im,jm))%如果转化后灰度级小于等于模版对应位置数值
          image_bayer(i,j)=0;%该像素设为黑
      else
          image_bayer(i,j)=255;%否则该像素设为白
      end
      
      end
  end
  image_bayer=uint8(image_bayer);%转化为uint8类型
  figure(2)
  imshow(image_bayer);%显示bayer抖动后图像
  %image_gray=uint8(image_gray);
  
  image_gray=im2double(rgb2gray(image));%[0,1]范围内的灰度double型图像矩阵
  image_bayer=im2double(uint8(image_bayer));%[0,1]范围内的灰度double型图像矩阵
  
  peaksnr=psnr(image_bayer,image_gray)%计算psnr峰值信噪比
  ssimval=ssim(image_bayer,image_gray)%计算ssim结构相似性
  