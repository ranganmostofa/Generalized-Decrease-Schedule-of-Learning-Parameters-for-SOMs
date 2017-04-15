% SOM for 2d gaussians
function SOM2
% Prashant Kalvapalle
% Comp 504 HW6 - Base code for all problems
% close 1;
% NOTE : Initial (and final) lattice is a cell representation, In the function it is
% used as a multi-dimensional matrix

latticeSize = [8 8];
initRadius = max(latticeSize); % Initial radius of influence

numIters = 20000; % number of learning steps
alphaI = .8; % learning rate

nEmbedEval = 100;

% Input data entry
dataInput = [createGaussians([2 1000],.1,[7 7]), createGaussians([2 1000],.1,[0 7]), createGaussians([2 1000],.1,[7 0]), createGaussians([2 1000],.1,[0 0]),]; % each COLUMN is a data point

dimDataInput = size(dataInput,1); % gives the dimensionality of data space
latticeCell = createInitLattice(dimDataInput,latticeSize); % weights initialization

% Perform self organization
[finalLattice, stepsToConv, embedding,embeddingHamel] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI,nEmbedEval);

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
% figure;
% dI = reshape(dataInput',[],4,2); hold on;
% plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
% plot(finalLattice(:,:,1),finalLattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
% plot(finalLattice(:,:,1),finalLattice(:,:,2),'b-'); plot(finalLattice(:,:,1)',finalLattice(:,:,2)','b-');
% xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space: Final after ',num2str(stepsToConv),' Learning Steps'])
% legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')

%% plotting histogram
% plotHistoChart(histoData);

%% plotting embedding history
figure;
plot(embedding(6,:), embedding(1,:)); xlabel('Learning steps'); ylabel('Embedding metric'); title(['Plot of Embedding history'])

plot(embeddingHamel); xlabel('Learning steps'); ylabel('Hamel Embedding metric'); title(['HAMEL: Plot of Embedding history'])

%% plotting Mean and variance changes along with decrease schedules
figure(3);
subplot(2,2,1); plot(embedding(6,:), embedding([3,5],:)); xlabel('Learning steps'); ylabel('Embedding metric'); title('Plot of Variance embedding'); legend('VarianceData','VariancePrototype')
subplot(2,2,2); plot(embedding(6,:), embedding([2,4],:)); xlabel('Learning steps'); ylabel('Embedding metric'); title('Plot of Mean embedding'); legend('meanData','meanPrototype')
decayIters = 10000; radius = zeros(1,numIters); alpha = radius;
for i = 1:numIters
    radius(i) = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
    alpha(i) = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
end
subplot(2,2,3); plot(1:numIters, radius); xlabel('Learning steps'); ylabel('Radius'); title('Plot of radius decrease schedule');
subplot(2,2,4); plot(1:numIters, alpha); xlabel('Learning steps'); ylabel('alpha'); title('Plot of alpha decrease schedule');
end


function [finalLattice, stepsToConv, embedding,embeddingHamel] = selfOrganize(latticeCell,dataInput,numIters,initRadius,alphaI,nEmbedEval)
% the self organizing map steps here

% convert the input lattice cell into a multi-dimensional Matrix
Z = cellfun(@(x)reshape(x,1,1,[]),latticeCell,'un',0);
lattice = cell2mat(Z); % this is a multi-dimensional Matrix, with third dimension holding different input dimensions

r = (1:size(lattice,1))';c = 1:size(lattice,2);
latticeIndices(:,:,1) = r(:,ones(1,size(lattice,2))); latticeIndices(:,:,2) = c(ones(1,size(lattice,1)),:);  % latticeIndices : holds the i,j indices of the 2d lattice space

figure(1);
subplot(2,2,1);
dI = reshape(dataInput',[],4,2); hold on;
plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
xlabel('First data dimension'); ylabel('Second data dimension'); title('Plot of prototypes in input space : Initial');
% legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')

dum = 2;
% [~, oldMapData, ~] = calcDensityLattice(lattice,dataInput,size(latticeCell)); % table of the prototype where each data point maps
stepsToConv = numIters;
embedding = ones(6,numIters/nEmbedEval);
embeddingHamel = ones(1,numIters);


for i = 1:numIters
    %     radius = initRadius; % can do decay here
%     embedding(:,i) = calcEmbed(dataInput, lattice); embedding(6,i) = i;
    embeddingHamel(i) = calcEmbedHamel(dataInput,lattice);
    
    decayIters = 10000;
    radius = initRadius * ((i <= decayIters/5) + .8 * (i > decayIters/5 & i <= decayIters/2) + .5 * (i > decayIters/2 & i <= decayIters*.8)+ .2 * (i > decayIters*.8));
    alpha = alphaI * ((i <= decayIters/10) + .5 * (i > decayIters/10 & i <= decayIters/2.5) + .125 * (i > decayIters/2.5 & i <= decayIters*.8)+ .025 * (i > decayIters*.8));
    
    % pick an x (data point) randomly
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
    
    % Calculate embedding every 100 steps
    if mod(i,nEmbedEval) == 0
        indexEmbed = i/nEmbedEval;
        embedding([1:5],indexEmbed) = calcEmbed(dataInput, lattice); embedding(6,indexEmbed) = i;
    end
    
    % making plots at particular learning steps as defined in the vector
    if sum(i == [decayIters/10 decayIters/2 decayIters])
        % Plot the mapping and input data
        figure(1); subplot(2,2,dum);
        dI = reshape(dataInput',[],4,2); hold on;
        plot(dI(:,1,1),dI(:,1,2),'r.');  plot(dI(:,2,1),dI(:,2,2),'m.');  plot(dI(:,3,1),dI(:,3,2),'g.');  plot(dI(:,4,1),dI(:,4,2),'y.');
        plot(lattice(:,:,1),lattice(:,:,2),'ko','MarkerFaceColor','k','MarkerSize',4);
        plot(lattice(:,:,1),lattice(:,:,2),'b-'); plot(lattice(:,:,1)',lattice(:,:,2)','b-');
        xlabel('First data dimension'); ylabel('Second data dimension'); title(['Plot of prototypes in input space at ',num2str(i),' Learning Steps'])
        %         legend('Input data1','Input data2','Input data3','Input data4','Prototype vectors')
        dum = dum + 1;
    end
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


function [densityLattice,mapData,histoData]  = calcDensityLattice(lattice,dataInput,sizeOflatticeCell)
% calculates the prototype each data point is mapped to
densityLattice = zeros(sizeOflatticeCell);
mapData = zeros([2 size(dataInput,2)]);
histoData = zeros([sizeOflatticeCell,4]);
% seq = [ones(1,1000) 2*ones(1,1000) 3*ones(1,1000) 4*ones(1,1000)];
for i = 1:size(dataInput,2)
    x = dataInput(:,i);
    
    % find euclidian distances and difference between chosen x and all W's
    differenceMatrix = repmat(reshape(x,1,1,[]),[size(lattice,1),size(lattice,2),1]) - lattice; % a 3D matrix
    distToXMatrix = sqrt(sum((differenceMatrix).^2,3)); % a 2D matrix for euclidian distances to x
    
    % find the winner = c = [win_row win_col]
    [~, winner] = min(distToXMatrix(:)); [win_row, win_col] = ind2sub(size(distToXMatrix), winner);
    c = [win_row win_col];
    % update the density lattice
    densityLattice(c(1),c(2)) = densityLattice(c(1),c(2)) + 1;
    mapData(:,i) = [win_row win_col];
    histoWrite = ([1 0 0 0] * (i <= 1000) + [0 1 0 0] * (i > 1000 & i <= 2000) + [0 0 1 0] * (i > 2000 & i <= 3000) + [0 0 0 1] * (i > 3000 & i <= 4000));
    histoData(c(1),c(2),:) = histoData(c(1),c(2),:) + (reshape(histoWrite,1,1,[]));
end

end


function neighbourhoodFn = makeNeighbourhoodFn(latticeIndices,c,radius)

distNeighbour = sum(abs(latticeIndices - repmat(reshape(c,1,1,[]),[size(latticeIndices,1),size(latticeIndices,2),1])),3); % Manhattan distance metric for the neighbourhood function
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
x1(1,:) = x1(1,:) + repmat(mean(1),[1,size(x1,2)]);
x1(2,:) = x1(2,:) + repmat(mean(2),[1,size(x1,2)]);
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

function embedding = calcEmbed(dataInput, lattice)
linearLattice = reshape(lattice,size(dataInput,1),[]);
mData = mean(dataInput,2);
vData = var(dataInput,0,2);
mProt = mean(linearLattice,2);
vProt = var(linearLattice,0,2);
embedding = [mData vData mProt vProt]'; embedding = mean(embedding,2);
embedding = [(1 - mean(mProt./mData) + 1 - mean(vProt./vData)); embedding];
end


function embedding = calcEmbedHamel(dataInput, lattice)
linearLattice = reshape(lattice,size(dataInput,1),[]);
mData = mean(dataInput,2);
vData = var(dataInput,0,2);
mProt = mean(linearLattice,2);
vProt = var(linearLattice,0,2);

nProt = size(linearLattice,2);
nData = size(dataInput,2);

% Hamel formulas for 95% accuracy
fy = finv(0.95/2,nData-1,nProt-1);
vleftLimit = (vData./vProt)/fy;
vrightLimit = (vData./vProt)*fy;
vEmbed = vleftLimit<=1 & 1<=vrightLimit;

mleftLimit = (mData-mProt) - 1.96*sqrt(vData/nData + vProt/nProt);
mrightLimit = (mData-mProt) + 1.96*sqrt(vData/nData + vProt/nProt);
mEmbed = mleftLimit<=1 & 1<=mrightLimit;

fSignificance = vData / sum(vData);

embedding = mean(fSignificance.*(vEmbed .* mEmbed));
end
