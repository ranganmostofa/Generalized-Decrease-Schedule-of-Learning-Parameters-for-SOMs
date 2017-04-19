%% Two diamonds dataset
%Alfred Ultsch, Clustering with SOM: U*C, 
%in Proc. Workshop on Self Organizing Feature Maps ,pp 31-37 Paris 2005.
function generate_two_diamonds_data
%% HEADER
%Generate two diamond dataset
%INPUT
%   no input (however, the data files must be present in directory
%   structure of this project. That is in, ../dataset/*).
%RETURN
%  none : No return value. As a side effect, this function saves smartphone
%         human activity dataset to ../dataset/two_diamond_data.mat file.
%%
%   
    f_X = fopen('../dataset/FCPS/01FCPSdata/TwoDiamonds.lrn','rt');
    f_Y = fopen('../dataset/FCPS/01FCPSdata/TwoDiamonds.cls','rt');
    
    X = textscan(f_X,'%f%f%f\n','HeaderLines',4);
    Y = textscan(f_Y,'%f%d\n','HeaderLines',1);
    X = [X{1} X{2} X{3}];
    Y = [Y{1} Y{2}];
    
    [~,Ix] = sort(X(:,1));
    X = X(Ix,:);
    
    [~,Iy] = sort(Y(:,1));
    Y = Y(Iy,:);
    
    X = X(:,2:end)';
    Y = Y(:,2:end)';

    %save the data to in matlab format, so that loading is easy.
    save('../dataset/two_diamonds_data','X','Y'); 
end