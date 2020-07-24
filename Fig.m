clear;
clc;
x = 0:0.001:1;
figure(1);
for beta=2:1:10
    y = (log(x*(beta-1)+1))./(log(beta));
    plot(x,y);
    

    hold on
end
 title('图像亮度增强曲线')  
 xlabel('输入')
 ylabel('输出')
 l = {'beta=2','beta=3','beta=4','beta=5','beta=6','beta=7','beta=8','beta=9','beta=10'};
% legend(l(1),l(2),l(3),l(4),l(5),l(6),l(7),l(8),l(9));
legend(l)