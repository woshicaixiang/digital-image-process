function [ newC ] = swap_not( Window, C, filter)
%  将模板内中心点与邻域内其他八个像素点，以及自身反相共九种情况，选取使误差总和最小的情况
%% 交换swap，记录每次交换后的误差总和
D1=ones(1,9);
for i=1:9
    temp=C(5);
    C(5)=C(i);
    C(i)=temp;
    C_filter=imfilter(C, filter, 'replicate');
    D1(i)=Sum_error(Window,C_filter);
    C(i)=C(5);
    C(5)=temp;
end
[min_pos,k]=min(D1);    %选取误差总和中最小的进行交换
temp=C(5);
C(5)=C(k);
C(k)=temp;
%% 反相not
C(5)=double(~C(5));
C_filter=imfilter(C, filter, 'replicate');
min_nag=Sum_error(Window,C_filter);
C(5)=double(~C(5));
% for i=1:9
%     temp=C(5);
%     C(5)=C(i);
%     C(i)=temp;
%     %C_filter=imfilter(C, filter, 'replicate');
%     D2(i)=Sum_error(Window,C);
%     C(i)=C(5);
%     C(5)=temp;
% end
% 比较
if min_pos>min_nag      %如果反相后误差变小，则将中心值反相，否则维持原状
    C(5)=double(~C(5));
end
% if min_pos<min_nag
%         temp=C(5);
%         C(5)=C(k);
%         C(k)=temp;
% else
%         C(5)=double(~C(5));
%         temp=C(5);
%         C(5)=C(q);
%         C(q)=temp;
% end
newC=C;
end

