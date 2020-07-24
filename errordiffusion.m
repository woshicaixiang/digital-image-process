clear
clc

image=imread('D:\project1_digital_image_process\cai_3.jpg');%读入图像
image_gray=im2double(rgb2gray(image));%转化为[0,1]范围内double型灰度图像
[row,col]=size(image_gray);
image_floyd=zeros(row,col);%建立处理后矩阵
figure(1)
imshow(rgb2gray(image));%显示原图

%误差扩散系数
a=7/16;
b=3/16;
c=5/16;
d=1/16;

%以从左到右、从上到下顺序处理每一个像素
for i=1:row-1
    for j=2:col-1
        now=image_gray(i,j);%读取灰度值
        if now<1/2  %如果小于阈值1/2
            image_floyd(i,j)=0; %该像素设为黑色
        else
            image_floyd(i,j) = 1; %否则该像素设为白色
        end
        err=(now-image_floyd(i,j)); %计算原灰度值与阈值的误差
        
        %将误差扩散给周边像素，改变像素灰度值
        image_gray(i,j+1)=image_gray(i,j+1)+err*a; 
        image_gray(i+1,j-1)=image_gray(i+1,j-1)+err*b;
        image_gray(i+1,j)=image_gray(i+1,j)+err*c;
        image_gray(i+1,j+1)=image_gray(i+1,j+1)+err*d;
    end
end

figure(2)
imshow(image_floyd,[0,1]);%显示处理后图像

image_gray0=im2double(rgb2gray(image));%[0,1]范围内的灰度double型图像矩阵

peaksnr=psnr(image_floyd,image_gray0)%计算psnr峰值信噪比
ssimval=ssim(image_floyd,image_gray0)%计算ssim结构相似性
