% 本代码对于200*200像素的图像，运行6min左右，500*500的图像，运行40min左右
clc;clear;
tic;
%% 读取连续调图像
img=imread('D:\project1_digital_image_process\cai_3.jpg');
A=rgb2gray(img);
A=double(A)/255;
[m,n]=size(A);
figure(1);
imshow(A,[]);%title('连续调图像');
filter = fspecial('gaussian',[3,3],0.5);    %3*3高斯滤波器 
A = imfilter(A, filter, 'replicate');   

B=rand(m,n);    %随机初始半色调图像
for i=1:m %二值化
    for j=1:n
        if B(i,j)>0.5
            B(i,j)=1;
        else
            B(i,j)=0;
        end
    end
end

%% DBS算法
step=5; %迭代次数
error=zeros(1,step+1);  %记录误差和
error(1)=Sum_error(A,B);
for k=1:step
    for i=2:m-1
        for j=2:n-1
            Window=A(i-1:i+1,j-1:j+1);
            C=B(i-1:i+1,j-1:j+1);
            C=swap_not( Window, C, filter); %反向与邻域八像素比较
            B(i-1:i+1,j-1:j+1)=C;           %更新半色调图像
        end
    end
    error(k+1)=Sum_error(A,B);
end
toc;
figure(2);
plot(1:step+1,error);title('误差总和随迭代次数的变化')
figure(3);
imshow(B,[0,1]);%title('DBS算法半色调化图像')
