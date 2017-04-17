%% Data generator smartphone dataset
function generate_hapt_data
%% HEADER
%Generate smartphone human activity dataset
%INPUT
%   no input (however, the data text files must be present in directory
%   structure of this project. That is in, ../dataset/*).
%RETURN
%  none : No return value. As a side effect, this function saves smartphone
%         human activity dataset to ../dataset/hapt_data.mat file.
%%
%   
    Xtrain = dlmread('../dataset/hapt_data_set/Train/X_train.txt')';
    Ytrain = dlmread('../dataset/hapt_data_set/Train/Y_train.txt')';
    SIDtrain=dlmread('../dataset/hapt_data_set/Train/subject_id_train.txt')';

    Xtest = dlmread('../dataset/hapt_data_set/Test/X_test.txt')';
    Ytest = dlmread('../dataset/hapt_data_set/Test/Y_test.txt')';
    SIDtest=dlmread('../dataset/hapt_data_set/Test/subject_id_test.txt')';


    f_featureDescription = fopen('../dataset/hapt_data_set/features.txt','rt');
    f_activityLabel = fopen('../dataset/hapt_data_set/activity_labels.txt','rt');
    featureDescription_data = textscan(f_featureDescription,'%s\n',...
    'HeaderLines',0);
    activityLabel_data = textscan(f_activityLabel,'%d%s\n',...
      'HeaderLines',0);
    featureDescription = featureDescription_data{1};
    activityLabel = activityLabel_data{2};

    %save the data to in matlab format, so that loading is easy.
    save('../dataset/hapt_data','Xtrain','Ytrain','SIDtrain','Xtest','Ytest',...
      'SIDtest','featureDescription','activityLabel'); 
end