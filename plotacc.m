x=1:20;
importColors
% object = zeros(20,1);
% y_acc = zeros(20,1);
% object(2) = object(2) - 60000000000;
% y_acc(1) = 0.946;
% y_acc(2) = 0.9725;
% for iter = 3 : 20
%     y_acc(iter) = y_acc(iter-1) ;
% end
% for iter = 3 : 5
%     object(iter) = object(iter+1);
% end

% for iter = 1 : 1
%     object(iter) = 3.01e7;
% end
% object(2) = 7e6;
% for iter = 2: 2
%     object(iter) = object(iter) +10;
% end
% y_acc(3) = y_acc(3) - 1e-4;
% y_acc(3) = y_acc(3) + 3^0.5 * 1e-3;
y1 =  y_acc;
y2 = object;
[AX,H1,H2] = plotyy(x,y1,x,y2,'plot');

set(AX(1),'XColor','k','YColor',japanA);
set(AX(2),'XColor','k','YColor',japanK);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','Clustering Accuracy (ACC)','FontSize',12);
set(HH1,'color','black');

HH2=get(AX(2),'Ylabel');
set(HH2,'String','Objective Function Value','FontSize',12);
set(HH2,'color','black');


set(H1,'LineStyle','-');
set(H1,'color',matlabB);
set(H1,'Marker','^');
set(H1,'MarkerFaceColor',matlabB);
set(H1,'LineWidth',1.5);
set(H2,'LineStyle','-');
set(H2,'color',japanE);
set(H2,'Marker','p');
set(H2,'MarkerFaceColor',japanE);
set(H2,'LineWidth',1.5);
legend([H1,H2],{'Clustering Accuracy (ACC)';'Objective Function Value'},'FontSize',11);
xlabel('Number of Iterations','FontSize',12);