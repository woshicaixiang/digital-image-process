function [ error ] = Sum_error( A, B )
%   ��������ͼ��֮��ĸ����ص������ܺ�
E=A-B;
E=E.^2;    %ŷʽ����
error = sqrt(sum(sum(E)));
%E=abs(E);  %����ֵ����
%error = sum(sum(E));
end

