clear
clc

image=imread('D:\project1_digital_image_process\cai_3.jpg');%����ͼ��
image_gray=im2double(rgb2gray(image));%ת��Ϊ[0,1]��Χ��double�ͻҶ�ͼ��
[row,col]=size(image_gray);
image_floyd=zeros(row,col);%������������
figure(1)
imshow(rgb2gray(image));%��ʾԭͼ

%�����ɢϵ��
a=7/16;
b=3/16;
c=5/16;
d=1/16;

%�Դ����ҡ����ϵ���˳����ÿһ������
for i=1:row-1
    for j=2:col-1
        now=image_gray(i,j);%��ȡ�Ҷ�ֵ
        if now<1/2  %���С����ֵ1/2
            image_floyd(i,j)=0; %��������Ϊ��ɫ
        else
            image_floyd(i,j) = 1; %�����������Ϊ��ɫ
        end
        err=(now-image_floyd(i,j)); %����ԭ�Ҷ�ֵ����ֵ�����
        
        %�������ɢ���ܱ����أ��ı����ػҶ�ֵ
        image_gray(i,j+1)=image_gray(i,j+1)+err*a; 
        image_gray(i+1,j-1)=image_gray(i+1,j-1)+err*b;
        image_gray(i+1,j)=image_gray(i+1,j)+err*c;
        image_gray(i+1,j+1)=image_gray(i+1,j+1)+err*d;
    end
end

figure(2)
imshow(image_floyd,[0,1]);%��ʾ�����ͼ��

image_gray0=im2double(rgb2gray(image));%[0,1]��Χ�ڵĻҶ�double��ͼ�����

peaksnr=psnr(image_floyd,image_gray0)%����psnr��ֵ�����
ssimval=ssim(image_floyd,image_gray0)%����ssim�ṹ������
