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
 bayer=bayer';%bayer��������
 
  image=imread('D:\project1_digital_image_process\cai_3.jpg');%����ͼ��
  image_gray=double(rgb2gray(image));%ת��Ϊ�Ҷ�ͼ��
  [r,c]=size(image_gray);
  image_bayer=zeros(r,c);%�������������
  figure(1)
  imshow(rgb2gray(image));%��ʾԭͼ
  
  for i=1:r
      for j=1:c
      im=mod(i,8)+1;%Ѱ����ģ���е���λ��
      jm=mod(j,8)+1;%Ѱ����ģ���е���λ��
      
      nowpoint=image_gray(i,j)/256*64;%���Ҷ�ת����[0,64]
      
      if(nowpoint<=bayer(im,jm))%���ת����Ҷȼ�С�ڵ���ģ���Ӧλ����ֵ
          image_bayer(i,j)=0;%��������Ϊ��
      else
          image_bayer(i,j)=255;%�����������Ϊ��
      end
      
      end
  end
  image_bayer=uint8(image_bayer);%ת��Ϊuint8����
  figure(2)
  imshow(image_bayer);%��ʾbayer������ͼ��
  %image_gray=uint8(image_gray);
  
  image_gray=im2double(rgb2gray(image));%[0,1]��Χ�ڵĻҶ�double��ͼ�����
  image_bayer=im2double(uint8(image_bayer));%[0,1]��Χ�ڵĻҶ�double��ͼ�����
  
  peaksnr=psnr(image_bayer,image_gray)%����psnr��ֵ�����
  ssimval=ssim(image_bayer,image_gray)%����ssim�ṹ������
  