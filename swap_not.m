function [ newC ] = swap_not( Window, C, filter)
%  ��ģ�������ĵ��������������˸����ص㣬�Լ������๲���������ѡȡʹ����ܺ���С�����
%% ����swap����¼ÿ�ν����������ܺ�
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
[min_pos,k]=min(D1);    %ѡȡ����ܺ�����С�Ľ��н���
temp=C(5);
C(5)=C(k);
C(k)=temp;
%% ����not
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
% �Ƚ�
if min_pos>min_nag      %������������С��������ֵ���࣬����ά��ԭ״
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

