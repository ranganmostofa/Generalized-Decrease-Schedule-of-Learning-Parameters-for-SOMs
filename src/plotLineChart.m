function plotLineChart(lattice,histoData)
figure(20);
m = size(lattice,1); n = size(lattice,2);
p = 1;
histoData = histoData(:,:,[1 3 2]);

axes('position',[0 0 1 1]); 
imagesc(histoData/5);
alpha (1); axis off; 

% ymaxx = max(max(max(histoData))) + 1;
for j = 1:n % x axis varying
    for i = 1:m % y axis varying
        ax = axes('position',[(j-1)/m (n-i)/n 1/m 1/n]); hold on; % i = y axis (row) ; j = x axis (column)
        hiss = reshape(lattice(i,j,:),1,size(lattice,3));
        ymaxx = max(max(max(lattice))) + 1;
%         colors = {'r', 'b', 'g', 'y'};
        % Plots different colour points for each data type
        ylim([0 ymaxx]); xlim([0 5]);
        plot(ax,1:size(lattice,3),hiss,'w.-','LineWidth',1.5)
%         axis off;
        set(gca,'Color','None')
        
%         for k = 1:numel(hiss)
%             randPos = rand(2,hiss(k))* .8 + .1;
%             plot(ax,randPos(1,:), randPos(2,:),'ko','MarkerFaceColor',colors{k});
% %             bar(ax,k, hiss(k),colors{k});
%         end
        
        set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]);
        set(ax,'Box','on')
%         set(gca,'Visible','off');
%         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
        p = p + 1;

    end
end

% drawnow
end
