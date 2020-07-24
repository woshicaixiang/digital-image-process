% ���������200*200���ص�ͼ������6min���ң�500*500��ͼ������40min����
clc;clear;
tic;
%% ��ȡ������ͼ��
img=imread('D:\project1_digital_image_process\cai_3.jpg');
A=rgb2gray(img);
A=double(A)/255;
[m,n]=size(A);
figure(1);
imshow(A,[]);%title('������ͼ��');
filter = fspecial('gaussian',[3,3],0.5);    %3*3��˹�˲��� 
A = imfilter(A, filter, 'replicate');   

B=rand(m,n);    %�����ʼ��ɫ��ͼ��
for i=1:m %��ֵ��
    for j=1:n
        if B(i,j)>0.5
            B(i,j)=1;
        else
            B(i,j)=0;
        end
    end
end

%% DBS�㷨
step=5; %��������
error=zeros(1,step+1);  %��¼����
error(1)=Sum_error(A,B);
for k=1:step
    for i=2:m-1
        for j=2:n-1
            Window=A(i-1:i+1,j-1:j+1);
            C=B(i-1:i+1,j-1:j+1);
            C=swap_not( Window, C, filter); %��������������رȽ�
            B(i-1:i+1,j-1:j+1)=C;           %���°�ɫ��ͼ��
        end
    end
    error(k+1)=Sum_error(A,B);
end
toc;
figure(2);
plot(1:step+1,error);title('����ܺ�����������ı仯')
figure(3);
imshow(B,[0,1]);%title('DBS�㷨��ɫ����ͼ��')
