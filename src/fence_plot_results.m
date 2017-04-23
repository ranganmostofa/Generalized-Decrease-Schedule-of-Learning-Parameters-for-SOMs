function fence_plot_results

%%  Load data
    Wdata = load('../results/prototypes_data.mat');
    
    W_diamonds = Wdata.prototype_Diamonds;
    W_iris     = Wdata.prototype_Iris;
    W_differentVariance = Wdata.prototype_differentVariance;
    W_fourGaussians  = Wdata.prototype_fourGaussians;
    W_overlapping    = Wdata.prototype_overlapping;
    W_singleGaussian = Wdata.prototype_singleGaussian;
    W_subset  = Wdata.prototype_subset;
    W_wingNut = Wdata.prototype_wingNut;
    
    iris_data = load('../dataset/iris_data.mat');
    X_iris = [iris_data.trainInput,iris_data.testInput];
    [~,D_iris] = max([iris_data.trainOutput,iris_data.testOutput],[],1);
    
    diamonds_data = load('../dataset/two_diamonds_data.mat');    
    wingNut_data = load('../dataset/wingnut_data.mat');
    
%%  inner function for overlaying mU and prototypes W vectors on same plot.
    function fig = plot_mU_W_label(W,tstr)
    % nested function. wrapper for plotting functions.
        [lat_wid,lat_len,~] = size(W);
        fig = figure();
        whitebg([0 0 0]);
        axis([1 lat_len+1 1 lat_wid+1]);
        fig = plot_mU(fig,1-colormap(gray),W,2);
        fig = decorate_weight_vector(fig,W,[1 1 1],1);
        fig = decorate_class_label(fig,W,X,D,label_text,label_color,6);
        title(tstr);
    end

    function fig = plot_mU_W(W,tstr)
    % nested function. wrapper for plotting functions.
        [lat_wid,lat_len,~] = size(W);
        fig = figure();
        whitebg([0 0 0]);
        axis([1 lat_len+1 1 lat_wid+1]);
        fig = plot_mU(fig,1-colormap(gray),W,2);
        fig = decorate_weight_vector(fig,W,[1 1 1],1);
        title(tstr);
    end

    %% Plot mU fences and weight vectors for all data    
    unique_class = 3;
    label_color = mat2cell(hsv(unique_class),ones(1,unique_class),3);
    label_text  = {'SET','VER','VIR'};
    X=X_iris; D=D_iris;
    plot_mU_W_label(W_iris,['Iris dataset: fence plot and '...
        'weight vectors']);
    
    unique_class = 2;
    label_color = mat2cell(hsv(unique_class),ones(1,unique_class),3);
    label_text  = {'Class\_1','Class\_2'};
    X=diamonds_data.X; D=diamonds_data.Y;
    plot_mU_W_label(W_diamonds,['Diamonds dataset: fence plot and '...
        'weight vectors']);
    
    X=wingNut_data.X; D=wingNut_data.Y;
    plot_mU_W_label(W_wingNut,['wingNut dataset: fence plot and '...
        'weight vectors']);
    
    plot_mU_W(W_differentVariance,['Different Variance dataset: '...
        'fence plot and weight vectors']);       
    plot_mU_W(W_fourGaussians,['Four Gaussians dataset: fence plot and '...
        'weight vectors']);
    plot_mU_W(W_overlapping,['Overlapping dataset: fence plot and '...
        'weight vectors']);
    plot_mU_W(W_singleGaussian,['Single Gaussians dataset: fence plot and '...
        'weight vectors']);    
    plot_mU_W(W_subset,['Subset dataset: fence plot and '...
        'weight vectors']);
    
    display('Finished plotting fence plots');
end


%% mU matrix
function fig = plot_mU(fig,fence_color,W,boundary_width)
%% Header
% Plot mU matrix (fences). Only works for rectangular/square lattices.
% Input
%   fig : scalar : plotted in the figure.
%   fence_color : mx3 matrix : color map for the fence. The distances
%                 between prototypes are linearly mapped to this colormap.
%   W : matrix : prototype matrix of dimension
%                lattice_width-by-lattice_length-by-prototype_dimension.
% Return :
%   fig : scalar : figure window number.
%%
    [lat_wid,lat_len,dim] = size(W);
    Wpad = nan*ones(lat_wid+2,lat_len+2,dim);  % 1 layer padding.
    Wpad(2:lat_wid+1,2:lat_len+1,:) = W;
    hdiff = Wpad(1:lat_wid+1,2:lat_len+1,:) - ...
        Wpad(2:lat_wid+2,2:lat_len+1,:);
    vdiff = Wpad(2:lat_wid+1,1:lat_len+1,:) - ...
        Wpad(2:lat_wid+1,2:lat_len+2,:); 
    
    % find distance. hdiff : horiontal neighbors, 
    hdiff = sqrt(sum(hdiff.^2,3)./sum(Wpad(1:lat_wid+1,2:lat_len+1,:).^2,3));
    
    % vdiff for vertical.
    vdiff = sqrt(sum(vdiff.^2,3)./sum(Wpad(2:lat_wid+1,1:lat_len+1,:).^2,3));
    [xg,yg] = meshgrid(1:lat_wid+1,1:lat_len+1);
    hdiff(isnan(hdiff)) = 0;
    vdiff(isnan(vdiff)) = 0;   
    scale = max([ max(abs(hdiff(:))), max(abs(vdiff(:))) ]);
    
    hdiff = 1-hdiff/scale;    % on gray scale, assume better matches 
    vdiff = 1-vdiff/scale;    % move to white.
    
    figure(fig);
    colormap(fence_color);
    
    %plot horizontal fences
    mesh(xg,yg,0*xg,hdiff,'EdgeColor','flat','LineWidth',boundary_width,...
        'FaceAlpha',0,'MeshStyle','row','Marker','none');
    hold on;
    grid off;
    
    %plot vertical fences
    mesh(xg,yg,0*xg,vdiff,'EdgeColor','flat','LineWidth',boundary_width,...
        'FaceAlpha',0,'MeshStyle','column','Marker','none');
    
    view([0 90]);      
end

%% plot weight vectors on lattice
function fig = decorate_weight_vector(fig,W,line_color,line_width)
%% Header
% Decorate plot with weight vectors.  Plots the weight vectors in unit
% squares whose corner coordinates are given by lattice 'latf' (see below).
% INPUT:
%   fig : scalar : plotted in the figure.
%   W : matrix : prototype matrix of dimension
%                lattice_width-by-lattice_length-by-prototype_dimension.
%   line_color : 3-element vector : weights plotted with this color.
% RETURN :
%   fig : scalar : figure window number.
%%
    [lat_wid,lat_len,dim] = size(W);
    M = lat_wid*lat_len;
    xval = 0.1:(0.8)/(dim+1):0.9;
    xval = xval(2:end-1);
    v = zeros(M,2);
    figure(fig);
    hold on;
    grid off;
    
    for i=1:lat_len
        for j=1:lat_wid
            y = (0.3)* (W(j,i,:)/max(abs(W(j,i,:)))) + 0.5; %y is between 0.2 to 0.8
            y = reshape(y,1,dim);
            plot(xval+i, y+j,...
                'Color',line_color,'LineWidth',line_width);
        end
    end
end


%% plot class label text
function fig = decorate_class_label(fig,W,X,D,label_text_array,...
    label_color_array,label_fontsize)
%% Header
% Plot text class labels.
% Input
%   fig : scalar: plots in this figure.
%   W : matrix : prototype matrix of dimension
%                lattice_width-by-lattice_length-by-prototype_dimension.
%   X    : matrix: Input data matrix (columns are data samples)
%   D    : vector: class labels (positive integers).
%   label_text_array : cell array : cell array having text labels for each
%                              class.
%   label_color_array : cell array : cell array having rgb triples for each
%                              class.
%   label_fontsize : scalar : font size for label text
% Return :
%   fig : scalar : figure window number.
%%
    [lat_wid,lat_len,dim] = size(W);
    uniqD = unique(D);
    bin = zeros(lat_wid,lat_len,length(uniqD));    
    nX    = size(X,2);
    get_class_index = @(p) (find(uniqD==D(p)));
    m_store = zeros(1,nX,2);
    c_store = zeros(1,nX);
    
    for p = 1:nX        
        Q = (W - repmat(reshape(X(:,p),[1 1 dim]),lat_wid,lat_len)).^2; 
        diff = sum(Q,3);
        [~, m_index] = min(diff(:));
        [m_x,m_y] = ind2sub(size(diff),m_index);
        m_store(1,p,:) = [m_x,m_y];
        c_store(p) = get_class_index(p);
        bin(m_x,m_y,c_store(p)) = bin(m_x,m_y,c_store(p)) + 1;
    end
    
    [density,majority_class] = max(bin,[],3);
    density(density~=0) = 1;    % create mask for non-null lattice cells.
    majority_class  = majority_class .* density;
    
    v = [0.2 0.2];
 
    fig=figure(fig);
    hold on;
    grid off;
    for i = 1:lat_len
        for j = 1:lat_wid
            ver =[v(1)+i,v(2)+j];
            class_index = majority_class(j,i);
            if(class_index ~= 0)
                text(ver(:,1), ver(:,2),label_text_array{uniqD(class_index)}, ...
                    'Color',label_color_array{uniqD(class_index)},...
                    'FontSize',label_fontsize);
            end
        end
    end
end