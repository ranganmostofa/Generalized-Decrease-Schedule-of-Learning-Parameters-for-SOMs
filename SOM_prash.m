% hw6 Problem 2
function [finalLattice, histoData, densityLattice,dataInput,dataClasses] = SOM4
% Prashant Kalvapalle 
% Comp 504 HW6 - Base code for all problems
 
% NOTE : Initial (and final) lattice is a cell representation, In the function it is
% used as a multi-dimensional matrix

latticeSize = [10 10]; 
initRadius = max(latticeSize); % Initial radius of influence

numIters = 20000; % number of learning steps
alphaI = .8; % learning rate

% Input data entry
trainDataFilename = 'iris-train copy.txt';
testDataFilename = 'iris-test copy.txt';

[trainInput,trainOutput] = loadData(trainDataFilename);
[testInput,testOutput] = loadData(testDataFilename);
dataInput = [trainInput ; testInput]';
dataClasses = [trainOutput ; testOutput];

dimDataInput = size(dataInput,1); % gives the dimensionality of data space
latticeCell = createInitLattice(dimDataInput,latticeSize); % weights initialization

% Perform self organization
[finalLattice, stepsToConv] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI,dataClasses);

% % giving the final weights of the lattice in Cell form
% finalLatticeCell = mat2cell(finalLattice,ones(1,latticeSize(1)),ones(1,latticeSize(2)),2); finalLatticeCell = cellfun(@(x)reshape(x,2,1),finalLatticeCell,'un',0);

[densityLattice, ~, histoData] = calcDensityLattice(finalLattice,dataInput,dataClasses,size(latticeCell));
densityLattice = mat2gray(densityLattice);
figure; imagesc(densityLattice); colormap(flipud(gray)); colorbar; title('Density of Inputs mapped to each Prototype')

if stepsToConv < numIters
    disp(['SOM Converged in ',num2str(stepsToConv),' steps'])
else 
    disp(['Maximum iterations exhausted = ',num2str(stepsToConv),' steps'])
end

% figure;
% dI = reshape(dataInput',[],4,3); hold on;
% plot3(dI(:,1,1),dI(:,1,2),dI(:,1,3),'r.');  plot3(dI(:,2,1),dI(:,2,2),dI(:,2,3),'m.');  plot3(dI(:,3,1),dI(:,3,2),dI(:,3,3),'g.');  plot3(dI(:,4,1),dI(:,4,2),dI(:,4,3),'y.');  
% plot3(finalLattice(:,:,1),finalLattice(:,:,2),finalLattice(:,:,3),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot3(finalLattice(:,:,1),finalLattice(:,:,2),finalLattice(:,:,3),'b-'); plot3(finalLattice(:,:,1)',finalLattice(:,:,2)',finalLattice(:,:,3)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space: Final after ',num2str(stepsToConv),' Learning Steps'])
% legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')

% plotHistoChart(histoData);
plotDotChart(histoData,densityLattice);

end


function [finalLattice, stepsToConv] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI,dataClasses)
% the self organizing map steps here

% convert the input lattice cell into a multi-dimensional Matrix 
Z = cellfun(@(x)reshape(x,1,1,[]),latticeCell,'un',0);
lattice = cell2mat(Z); % this is a multi-dimensional Matrix, with third dimension holding different input dimensions

r = (1:size(lattice,1))';c = 1:size(lattice,2); 
latticeIndices(:,:,1) = r(:,ones(1,size(lattice,2))); latticeIndices(:,:,2) = c(ones(1,size(lattice,1)),:);  % latticeIndices : holds the i,j indices of the 2d lattice space

% figure;
% % subplot(2,2,1);
% dI = reshape(dataInput',[],4,3); 
% plot3(dI(:,1,1),dI(:,1,2),dI(:,1,3),'r.');hold on;  plot3(dI(:,2,1),dI(:,2,2),dI(:,2,3),'m.');  plot3(dI(:,3,1),dI(:,3,2),dI(:,3,3),'g.');  plot3(dI(:,4,1),dI(:,4,2),dI(:,4,3),'y.');  
% plot3(lattice(:,:,1),lattice(:,:,2),lattice(:,:,3),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot3(lattice(:,:,1),lattice(:,:,2),lattice(:,:,3),'b-'); plot3(lattice(:,:,1)',lattice(:,:,2)',lattice(:,:,3)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); zlabel('Third data dimension'); title('Plot of prototypes in input space : Initial');
% legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
% grid on;
% zlim([-1 1]);
% dum = 2;
% [~, oldMapData, ~] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
stepsToConv = numIters;

for i = 1:numIters
%     radius = initRadius; % can do decay here
    decayIters = 10000;
    radius = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
    alpha = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
      
    % pick an x (data point) randomly
    x = dataInput(:,randi(size(dataInput,2)));
   
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = reshape(x,1,1,[]) - lattice; % a 3D matrix of difference between every weight and x
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % finding norm or eucledian distance
  
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner); 
    c = [win_row win_col];
    
    % make a neighbourhood function in a matrix
    neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius);
    
    % update the weights - Learning rule
    lattice = lattice + alpha * neighbourhoodFn .* differenceMatrix;
    
    % Checking for convergence every 1000 steps
%     if mod(i,1000) == 0
%         [~, ~, histoData] = calcDensityLattice(lattice,dataInput,dataClasses,size(latticeCell));
%         plotLineChart(lattice,histoData);
%         title(['Prototypes mapped in lattice space: at ',num2str(i),' iterations'])
% % %         mapData = calcDataMapping(lattice,dataInput); 
%         [~, mapData, histoData] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
%         match = (mapData(1,:) == oldMapData(1,:) & mapData(2,:) == oldMapData(2,:));
% %         figure(3); hold on; plot(i,sum(match),'k.');
%         if sum(match)/size(dataInput,2) >= (1 - 1e-3) % < .1 percentage change in prototype assignment
%             stepsToConv = i;
%         else
%             oldMapData = mapData;
%         end
%     end
    % making plots at particular learning steps as defined in the vector
%     plotIters = decayIters; % decayIters
%     if sum(i == [plotIters/4 plotIters/2 plotIters numIters])
%         % Plot the mapping and input data
%         figure%(1); subplot(2,2,dum);
%         dI = reshape(dataInput',[],4,3); 
%         plot3(dI(:,1,1),dI(:,1,2),dI(:,1,3),'r.'); hold on; plot3(dI(:,2,1),dI(:,2,2),dI(:,2,3),'m.');  plot3(dI(:,3,1),dI(:,3,2),dI(:,3,3),'g.');  plot3(dI(:,4,1),dI(:,4,2),dI(:,4,3),'y.');
%         plot3(lattice(:,:,1),lattice(:,:,2),lattice(:,:,3),'ko','MarkerFaceColor','k','MarkerSize',4);
%         plot3(lattice(:,:,1),lattice(:,:,2),lattice(:,:,3),'b-'); plot3(lattice(:,:,1)',lattice(:,:,2)',lattice(:,:,3)','b-');
%         xlabel('First data dimension'); ylabel('Second data dimension'); zlabel('Third data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
%         legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
%         grid on;
%         zlim([-1 1]);
%         dum = dum + 1;
%         
%     end
%     % making plots every 1000 learning steps to visually approximate
%     % learning steps to convergence
%         if ~mod(i,1000)
% %         Plot the mapping and input data
%         figure(2)
% %         subplot(2,2,dum);
%         dI = reshape(dataInput',[],4,2); hold on;
%         plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
%         plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
%         plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
%         xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
%         legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
%         hold off; 
%         drawnow; 
% %         dum = dum + 1;
%         end
    
    if stepsToConv < numIters
        break
    end
    
end
finalLattice = lattice;

end


function [densityLattice,mapData,histoData]  = calcDensityLattice(lattice,dataInput,dataClasses,sizeOflatticeCell)
% calculates the prototype each data point is mapped to = recall step
densityLattice = zeros(sizeOflatticeCell);
mapData = zeros([2 size(dataInput,2)]);
histoData = zeros([sizeOflatticeCell,3]);
% seq = [ones(1,1000) 2*ones(1,1000) 3*ones(1,1000) 4*ones(1,1000)]; 
for i = 1:size(dataInput,2)
    x = dataInput(:,i);
    
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = reshape(x,1,1,[]) - lattice; % a 3D matrix
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % a 2D matrix for euclidian distances to x
    
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner); 
    c = [win_row win_col];
    % update the density lattice
    densityLattice(c(1),c(2)) = densityLattice(c(1),c(2)) + 1;
    mapData(:,i) = [win_row win_col];
%     histoWrite = ([1 0 0 0] * (i <= 1000) + [0 1 0 0] * (i > 1000 & i <= 2000) + [0 0 1 0] * (i > 2000 & i <= 3000) + [0 0 0 1] * (i > 3000 & i <= 4000));
    histoData(c(1),c(2),:) = histoData(c(1),c(2),:) + reshape(dataClasses(i,:),1,1,[]);
end
    
end


function neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius)

distNeighbour = sum(abs(latticeIndices - reshape(c,1,1,[])),3); % Manhattan distance metric for the neighbourhood function
% EqDistNeighbour = sqrt(sum((latticeIndices - reshape(c,1,1,[])).^2,3)); % eucleidian distance metric for the neighbourhood function
neighbourhoodFn = exp(-((distNeighbour)./(radius)).^2);

end


function latticeCell = createInitLattice(dimDataInput,latticeSize)
% creates random weight vectors in a cell 
latticeCell = cell(latticeSize);
latticeCell = arrayfun(@(x) rand(dimDataInput,1),latticeCell, 'uni',0);
end


function x1 = createGaussians(dim,var,mean)
% creates a normal random vector with dim dimension. mean of first 2
% dimensions set by [mean1 mean2]
x1 = sqrt(var)*randn(dim(1),dim(2));
% x1=detrend(x1);
x1(1,:) = x1(1,:) + mean(1);
x1(2,:) = x1(2,:) + mean(2);
end


function plotHistoChart(histoData)
figure;
m = size(histoData,1); n = size(histoData,2);
p = 1;
ymaxx = max(max(max(histoData))) + 1;
for j = 1:n
    for i = 1:m
        ax = axes('position',[(i-1)/m (n-j)/n 1/m 1/n]); hold on;
        hiss = reshape(histoData(i,j,:),1,size(histoData,3));
        colors = {'r', 'm', 'g', 'y'};
        % Plots different bars for each data type
        for k = 1:numel(hiss)
            bar(ax,k, hiss(k),colors{k});
        end
        ylim([0 ymaxx]);
        set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]);
        set(ax,'Box','on')
%         set(gca,'Visible','off');
%         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
        p = p + 1;

    end
end
end



function plotDotChart(histoData,densityLattice)
figure;
m = size(histoData,1); n = size(histoData,2);
p = 1;
% ymaxx = max(max(max(histoData))) + 1;
for j = 1:n % x axis varying
    for i = 1:m % y axis varying
        ax = axes('position',[(j-1)/m (n-i)/n 1/m 1/n]); hold on;
        hiss = reshape(histoData(i,j,:),1,size(histoData,3));
        colors = {'r', 'b', 'g', 'y'};
        % Plots different colour points for each data type
        ylim([0 1]); xlim([0 1]);
        for k = 1:numel(hiss)
            randPos = rand(2,hiss(k))* .8 + .1;
            plot(ax,randPos(1,:), randPos(2,:),'ko','MarkerFaceColor',colors{k});
%             bar(ax,k, hiss(k),colors{k});
        end
        
        set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]);
        set(ax,'Box','on')
%         set(gca,'Visible','off');
%         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
        p = p + 1;

    end
end

axes('position',[0 0 1 1]); 
imagesc(densityLattice); colormap(flipud(gray)); 
alpha (.5); axis off; 
end


function [input,output] = loadData(filename)

fileID = fopen(filename,'r');

formatSpec = '%f %f %f %f';

sizeA = [4, Inf];

A = fscanf(fileID,formatSpec,sizeA)';

fclose(fileID);

input = zeros(size(A,1)/2,size(A,2));
output = zeros(size(A,1)/2,size(A,2)-1);

for i = 1:size(A,1)
    for j = 1:size(A,2)
        if mod(i,2) == 0
            if j ~= 4
                output(i/2,j) = A(i,j);
            end
        else
            input((i+1)/2,j) = A(i,j);
        end
    end
end

end

