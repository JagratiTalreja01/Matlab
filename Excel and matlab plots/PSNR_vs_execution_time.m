x = [0.677 0.813 2.210 1.540 1.350 0.483 2.550 4.240 0.878 3.230 4.475 2.468]
%a = [12000 8000 13000 11000 5000 20000 100000 8000 30000 40000 25000]%execution time
y =[31.74 32.47 32.63 32.49 32.47 32.64 32.46 32.72 32.64 32.59 32.78]
labels = {'MemNet','LapSRN','RCAN','RNAN','SRFBN','SAN','EDSR','SwinIR','HAN','NLSN','DAN'};
%plot (x,y)
figure,
%axis([0.01 100])
scatter(x,y,'b','Linewidth',3);
text(x,y,labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
% Plot the data
%figure;
%hold on;
%plot(x, 'b', 'LineWidth', 2);
%plot(y, 'r', 'LineWidth', 2);
%xlabel('Index');
%ylabel('Value');
%title('Unrelated Data');
%legend('x', 'y');