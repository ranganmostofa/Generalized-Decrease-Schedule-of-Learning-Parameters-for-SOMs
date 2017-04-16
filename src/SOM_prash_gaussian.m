% SOM for 2d gaussians
function SOM2
% Prashant Kalvapalle
% Comp 504 HW6 - Base code for all problems
% close 1;
% NOTE : Initial (and final) lattice is a cell representation, In the function it is
% used as a multi-dimensional matrix

latticeSize = [8 8];
initRadius = max(latticeSize)/2; % Initial radius of influence

numIters = 24000; % number of learning steps
alphaI = .4; % learning rate

nEmbedEval = 50;
tolerance = .1;

% Input data entry
dataInput = [createGaussians([2 1000],.1,[7 7]), createGaussians([2 1000],.1,[0 7]), createGaussians([2 1000],.1,[7 0]), createGaussians([2 1000],.1,[0 0]),]; % each COLUMN is a data vector
dI = reshape(dataInput',[],4,2); % resheped to give data classes in columns, and data co-ordinates in 3rd dimension

dimDataInput = size(dataInput,1); % gives the dimensionality of data space
lattice = createInitLattice(dimDataInput,latticeSize, mean(dataInput,2), std(dataInput,0,2)); % weights initialization

% Perform self organization
[finalLattice, stepsToConv, embedding, topology, avgEmbedding, avgTopology, totalError, avTotalError, radius, alpha] = selfOrganize(lattice,dataInput,dI,numIters,initRadius,alphaI,nEmbedEval,tolerance);

% % giving the final weights of the lattice in Cell form
% finalLatticeCell = mat2cell(finalLattice,ones(1,latticeSize(1)),ones(1,latticeSize(2)),2); finalLatticeCell = cellfun(@(x)reshape(x,2,1),finalLatticeCell,'un',0);

%% plotting density plot
% [densityLattice, ~, histoData] = calcDensityLattice(finalLattice,dataInput,size(latticeCell));
% densityLattice = mat2gray(densityLattice);
% figure; imagesc(densityLattice); colormap(flipud(gray)); colorbar; title('Density of Inputs mapped to each Prototype')

%% printing convergence steps
if stepsToConv < numIters
    disp(['SOM Converged in ',num2str(stepsToConv),' steps'])
else
    disp(['Maximum iterations exhausted = ',num2str(stepsToConv),' steps'])
end

%% plotting final prototype configuration w coloured known data classes
% plotFinalMap(dI,finalLattice,stepsToConv)

%% plotting histogram
% plotHistoChart(histoData);

%% plot other things related to embedding
plotErrorStuff(embedding,topology, avgEmbedding, avgTopology, totalError, avTotalError, radius,alpha,stepsToConv)

end


function [finalLattice, stepsToConv, embedding, topology, avgEmbedding, avgTopology, totalError, avTotalError, radiusVec, alphaVec] = selfOrganize(lattice,dataInput,dI,numIters,initRadius,alphaI,nEmbedEval,tolerance)
% the self organizing map steps here
checkLength = 5;

r = (1:size(lattice,1))';c = 1:size(lattice,2);
latticeIndices(:,:,1) = r(:,ones(1,size(lattice,2))); latticeIndices(:,:,2) = c(ones(1,size(lattice,1)),:);  % latticeIndices : holds the i,j indices of the 2d lattice space

% % Initial parameters
% radius = initRadius;
% alpha = alphaI;

% plotting initial weights
plotMappings(dI,lattice,1,1)
dum = 2;

radiusSteps = initRadius:-(initRadius-1)/4:1;
alphaSteps = alphaI:-(alphaI - .01)/4 :.01;

% [~, oldMapData, ~] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
stepsToConv = numIters;
embedding = ones(6,numIters/nEmbedEval); topology = ones(1,numIters/nEmbedEval); totalError = topology;
avgEmbedding = ones(2,numIters/(nEmbedEval * 10)); avgTopology =  ones(1,numIters/(nEmbedEval * 10)); avTotalError = avgTopology; indexEmbedAvg = 1;
radiusVec = zeros(1,numIters); alphaVec = radiusVec;
progress = 1;

for i = 1:numIters
    %     radius = initRadius; % can do decay here
    %     embedding(:,i) = calcEmbed(dataInput, lattice); embedding(6,i) = i;
%     decayIters = 10000;

    %% dynamic scheduling of radius and learning rate
    if ~mod(i/(nEmbedEval * 10),checkLength)     
        if abs(1-totalError(1,indexEmbedAvg)/mean(totalError(1,indexEmbedAvg-checkLength + 1:indexEmbedAvg))) <= 0.1
            progress = min(progress + 1,5);
        end        
    end
        radiusVec(i) = radiusSteps(progress); radius = radiusVec(i);
        alphaVec(i) = alphaSteps(progress); alpha = alphaVec(i);
%     radius = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
%     alpha = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
%     
    %% pick an x (data point) randomly
    x = dataInput(:,randi(size(dataInput,2)));
    
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = repmat(reshape(x,1,1,[]),[size(lattice,1),size(lattice,2),1]) - lattice; % a 3D matrix of difference between every weight and x
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % finding norm or eucledian distance
    
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner);
    c = [win_row win_col];
    
    % make a neighbourhood function in a matrix
    neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius);
    
    % update the weights - Learning rule
    lattice = lattice + alpha * repmat(neighbourhoodFn,[1,1,size(differenceMatrix,3)]) .* differenceMatrix;
    
    % Calculate embedding every nEmbedEval (~100) steps
    if mod(i,nEmbedEval) == 0
        indexEmbed = i/nEmbedEval;
        embedding(1:5,indexEmbed) = calcEmbed(dataInput, lattice); embedding(6,indexEmbed) = i;
        topology(1,indexEmbed)  = calcTopology(lattice,dataInput); 
        totalError(1,indexEmbed) = (embedding(1,indexEmbed) + topology(1,indexEmbed))/2;
    end
    
    % Averaging embedding every 10 data points (~1000 steps)
    if mod(i,nEmbedEval * 10) == 0
        indexEmbedAvg = i/(nEmbedEval * 10) + 1;
        avgEmbedding(1,indexEmbedAvg) = mean(embedding(1,indexEmbed - 9:indexEmbed)); avgEmbedding(2,indexEmbedAvg) = i;
        avgTopology(1,indexEmbedAvg)  = mean(topology(1,indexEmbed - 9:indexEmbed));  
        avTotalError(1,indexEmbedAvg) = (avgEmbedding(1,indexEmbedAvg) + avgTopology(1,indexEmbedAvg))/2; 
    end
    
    % making plots at particular learning steps as defined in the vector
%     if sum(i == [decayIters/10 decayIters/2 decayIters])
     if ~mod(i,1000)   
        % Plot the mapping and input data
        plotMappings(dI,lattice,i,dum)
        dum = dum + 1;
    end
    
    if avTotalError(1,indexEmbedAvg) < tolerance        
        embedding = embedding(:,totalError < 1); topology = topology(:,totalError < 1); totalError = totalError(totalError < 1);
        avgEmbedding = avgEmbedding(:,avTotalError < 1); avgTopology = avgTopology(:,avTotalError < 1); avTotalError = avTotalError(avTotalError < 1);
        radiusVec = radiusVec(radiusVec > 0); alphaVec = alphaVec(alphaVec > 0);
        stepsToConv = i;
        break
    end
end
finalLattice = lattice;
avgEmbedding(1,1) = embedding(1,1); avgEmbedding(2,1) = embedding(6,1);
avgTopology(1,1)  = topology(1,1);
avTotalError(1,1) = totalError(1,1);
end


function topologyMetric  = calcTopology(lattice,dataInput)
% calculates the neighbourhood of top two winners for some randomly sampled data points
% densityLattice = zeros(sizeOflatticeCell);
% histoData = zeros([sizeOflatticeCell,4]);
% seq = [ones(1,1000) 2*ones(1,1000) 3*ones(1,1000) 4*ones(1,1000)];
nPointsEval = .1 * size(dataInput,2); % what % of points to select to look at 

topologyMetric = 0; % mapData = zeros([2 size(dataInput,2)]);
randomizedDataInput = dataInput(:,randperm(size(dataInput,2)));

for i = 1:nPointsEval
    x = randomizedDataInput(:,i);
    
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = repmat(reshape(x,1,1,[]),[size(lattice,1),size(lattice,2),1]) - lattice; % a 3D matrix
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % a 2D matrix for euclidian distances to x
    
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner);
    c = [win_row win_col];
    
    % finding second winner
    distToXMatrix(c(1),c(2)) = inf;
    [~, winner2] = min(distToXMatrix(:)); [win2_row, win2_col] = ind2sub(size(distToXMatrix), winner2);
    c2 = [win2_row win2_col];
    topologyMetric = topologyMetric + (norm(abs(c - c2)) > sqrt(2));
    
%     % update the density lattice
%     densityLattice(c(1),c(2)) = densityLattice(c(1),c(2)) + 1;
%     mapData(:,i) = [win_row win_col];
%     histoWrite = ([1 0 0 0] * (i <= 1000) + [0 1 0 0] * (i > 1000 & i <= 2000) + [0 0 1 0] * (i > 2000 & i <= 3000) + [0 0 0 1] * (i > 3000 & i <= 4000));
%     histoData(c(1),c(2),:) = histoData(c(1),c(2),:) + (reshape(histoWrite,1,1,[]));
end
topologyMetric = topologyMetric/nPointsEval;
end


function neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius)

distNeighbour = sum(abs(latticeIndices - repmat(reshape(c,1,1,[]),[size(latticeIndices,1),size(latticeIndices,2),1])),3); % Manhattan distance metric for the neighbourhood function
% EqDistNeighbour = sqrt(sum((latticeIndices - reshape(c,1,1,[])).^2,3)); % eucleidian distance metric for the neighbourhood function
neighbourhoodFn = exp(-((distNeighbour)./(radius)).^2);

end


function lattice = createInitLattice(dimDataInput,latticeSize, dataMean, dataSD)
% creates random weight vectors in a 3d matrix

meanMatrix = repmat(reshape(dataMean,1,1,[]),latticeSize);
sdMatrix = repmat(reshape(dataSD/10,1,1,[]),latticeSize);

lattice = rand([latticeSize dimDataInput]) .* sdMatrix + meanMatrix;
end


function x1 = createGaussians(dim,var,mean)
% creates a normal random vector with dim dimension. mean of first 2
% dimensions set by [mean1 mean2]
x1 = sqrt(var)*randn(dim(1),dim(2));
% x1=detrend(x1);
x1(1,:) = x1(1,:) + repmat(mean(1),[1,size(x1,2)]);
x1(2,:) = x1(2,:) + repmat(mean(2),[1,size(x1,2)]);
end


function embedding = calcEmbed(dataInput, lattice)
linearLattice = reshape(lattice,size(dataInput,1),[]);
mData = mean(dataInput,2);
vData = var(dataInput,0,2);
mProt = mean(linearLattice,2);
vProt = var(linearLattice,0,2);
embedding = [mData vData mProt vProt]'; embedding = mean(embedding,2);
% embedding = [(1 - mean(mProt./mData) + 1 - mean(vProt./vData))/2; embedding];
embedding = [ 1 - mean(vProt./vData); embedding];

end


function plotHistoChart(histoData)
figure;
m = size(histoData,1); n = size(histoData,2);
p = 1;
for j = 1:n
    for i = 1:m
        ax = axes('position',[(i-1)/m (n-j)/n 1/m 1/n]); hold on;
        hiss = reshape(histoData(i,j,:),1,4);
        colors = {'r', 'm', 'g', 'y'};
        % Plots different bars for each data type
        for k = 1:numel(hiss)
            bar(ax,k, hiss(k),colors{k});
        end
        ylim([0 200]);
        set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]);
        set(ax,'Box','on')
        %         set(gca,'Visible','off');
        %         set(gca,'position',[i/m (n-j)/n 1/m 1/n])
        p = p + 1;
        
    end
end
end

% function plotMappings(dI,lattice,i,dum)
% % Plot the mapping and input data in one subplot
% figure(1); subplot(2,2,dum);
% hold on;
% plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
% plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); title(['',num2str(i),' Learning Steps'])
% % legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
% 
% end

function plotMappings(dI,lattice,iter,dum)
% Plot the mapping and input data in one subplot
figure(1); 
r = 5; c = 5; 
i = ceil(dum/c); j = mod(dum - 1,c); % i = y axis (row) ; j = x axis (column)
% subplot(2,2,dum);
ax = axes('position',[(j)/c (r-i)/r 1/r 1/c]);
hold on;
plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
legend(num2str(iter))
% xlabel('First data dimension'); ylabel('Second data dimension'); title(['',num2str(iter),' Learning Steps'])
% legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
set(ax,'YTickLabel',[]);set(ax,'XTickLabel',[]); set(ax,'Box','on')
end


function plotFinalMap(dI,finalLattice,stepsToConv)
% plotting final prototype configuration w coloured known data classes
figure;
hold on;
plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
plot(finalLattice(:,:,1),finalLattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
plot(finalLattice(:,:,1),finalLattice(:,:,2),'b-'); plot(finalLattice(:,:,1)',finalLattice(:,:,2)','b-');
xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space: Final after ',num2str(stepsToConv),' Learning Steps'])
legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
end

function plotErrorStuff(embedding,topology,avgEmbedding,avgTopology, totalError, avTotalError,radius,alpha,stepsToConv)

%% plotting Mean and variance changes along with decrease schedules
figure(3);
subplot(2,2,1); plot(embedding(6,:), embedding([3,5],:)); xlabel('Learning steps'); ylabel('Embedding metric'); title('Plot of Variance embedding'); legend('VarianceData','VariancePrototype')
subplot(2,2,2); plot(embedding(6,:), embedding([2,4],:)); xlabel('Learning steps'); ylabel('Embedding metric'); title('Plot of Mean embedding'); legend('meanData','meanPrototype')
decayIters = 10000;
% radius = zeros(1,numIters); alpha = radius;
% for i = 1:numIters
%     radius(i) = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
%     alpha(i) = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
% end
subplot(2,2,3); plot(1:stepsToConv, radius); xlabel('Learning steps'); ylabel('Radius'); title('Plot of radius decrease schedule');
subplot(2,2,4); plot(1:stepsToConv, alpha); xlabel('Learning steps'); ylabel('alpha'); title('Plot of alpha decrease schedule');

%% plotting embedding and topology history
% avEmbed = movmean(embedding(1,:),10);
% avTopo = movmean(topology(1,:),10); 

figure;
subplot(3,1,1);
plot(embedding(6,:), embedding(1,:)); xlabel('Learning steps'); ylabel('Embedding error metric'); title('Plot of Embedding History')
hold on; plot(avgEmbedding(2,:), avgEmbedding(1,:),'r');
subplot(3,1,2);
plot(embedding(6,:), topology(1,:)); xlabel('Learning steps'); ylabel('Topology error metric'); title('Plot of Topology History')
hold on; plot(avgEmbedding(2,:), avgTopology,'r');
subplot(3,1,3);
plot(1:stepsToConv, radius); xlabel('Learning steps'); ylabel('Radius'); title('Plot of radius and alpha decrease schedule');
hold on; plot(1:stepsToConv, 10 * alpha); legend('Neighbourhood Radius','Learning rate x 10');

%% plotting sum of embedding and topology

figure;
subplot(2,1,1);
plot(embedding(6,:), totalError); xlabel('Learning steps'); ylabel('Total error metric'); title('Plot of Total Error History')
hold on; plot(avgEmbedding(2,:), avTotalError,'r');
subplot(2,1,2);
plot(1:stepsToConv, radius); xlabel('Learning steps'); ylabel('Radius'); title('Plot of radius and alpha decrease schedule');
hold on; plot(1:stepsToConv, 10 * alpha); legend('Neighbourhood Radius','Learning rate x 10');

end
