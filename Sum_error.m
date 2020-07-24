function [ error ] = Sum_error( A, B )
%   计算两个图像之间的各像素点的误差总和
E=A-B;
E=E.^2;    %欧式距离
error = sqrt(sum(sum(E)));
%E=abs(E);  %绝对值距离
%error = sum(sum(E));
end

