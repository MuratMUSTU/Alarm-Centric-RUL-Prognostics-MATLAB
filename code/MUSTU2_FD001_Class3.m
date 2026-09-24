%% SECOND PIPELINE (MODEL OPTIMIZATION and SELECTION)
clc;
clear;
load TrainData1sfnr

%For Ablation Study of no preprocessing (using raw data)
%load TrainData1
%TrainData1sfnr=TrainData1(:,[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]);
%ValidationData1sfnr=ValidationData1(:,[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]);
%TestData1sfnr=TestData1(:,[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]);

% Step1-Extract variables
predictorNames = TrainData1sfnr.Properties.VariableNames;
predictorNames = setdiff(predictorNames, {'Engine_ID', 'Time','TTF','RUL'});

X = TrainData1sfnr(:, predictorNames);
Y = TrainData1sfnr.RUL;

classOrder = ["alarm","warning","normal"];
Y = categorical(lower(string(Y)), classOrder);

groups = TrainData1sfnr.Engine_ID; % Create group vector
TTF = TrainData1sfnr.TTF;

% Step2-Create GROUP-AWARE CV
uniqueEngines = unique(groups);
% Create grouped CV partition, so same group (engine) stays in same fold
%Firstly, Create Outer CV with engine level split
rng(42,"twister");
cvGroup = cvpartition(numel(uniqueEngines), 'KFold', 5); 
% Then manually map folds:
cvIndices = zeros(size(groups));
for i = 1:cvGroup.NumTestSets
    testEngines = uniqueEngines(test(cvGroup, i));
    % Mapping for outer CV
    cvIndices(ismember(groups, testEngines)) = i;
end
cv_outer = cvpartition(cvIndices, 'KFold', 5);

%Secondly, Create inner CV with different engine level split
rng(100,"twister");
cv_inner = cvpartition(numel(uniqueEngines), 'KFold', 5);

cvInnerIndices = zeros(size(groups));
for i = 1:cv_inner.NumTestSets
    testEngines = uniqueEngines(test(cv_inner, i));
    %inner mapping
    cvInnerIndices(ismember(groups, testEngines)) = i;
end
%Final inner CV for group-aware
cv_inner = cvpartition(cvInnerIndices, 'KFold', 5);

% Step3-Define custom F1 evaluation function
% Already defined as alarmF1Loss.m 

%% STEP 4A - Train Ensemble Bag (Global Bayesian optimization + Group-Aware CV)

% Define related COST matrix (the best one)
costMatrix = [0 8 9;
              9  0  1;
              10  1  0];

% Global Bayesian optimization for AlarmF1Loss
%Bag
optimVars = [
    optimizableVariable('NumLearningCycles',[30 150],'Type','integer')
    optimizableVariable('MaxNumSplits',[400 16000],'Type','integer')
    optimizableVariable('MinLeafSize',[1 2],'Type','integer')
];
  
rng(42)

ObjFcn = @(params) ensembleObjectiveFcn1(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 50, ...
    'IsObjectiveDeterministic', true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel', false);


function objective = ensembleObjectiveFcn1(params, X, Y, costMatrix, cv_inner)

Ypred = Y;
Ypred(:) = missing;

    for i = 1:cv_inner.NumTestSets

        % fold-specific deterministic seed
        rng(1000 + i,"twister");

        trainIdx = training(cv_inner, i);
        testIdx  = test(cv_inner, i);
        
        t = templateTree( ...
    'MaxNumSplits', params.MaxNumSplits, ...
    'MinLeafSize', params.MinLeafSize);

        model = fitcensemble( ...
            X(trainIdx,:), Y(trainIdx), ...
            'Method','Bag', ...
            'Learners', t, ...
            'NumLearningCycles', params.NumLearningCycles, ...
            'Cost', costMatrix ...
        );
        
        Ypred(testIdx) = predict(model, X(testIdx,:));
    end


    % Minimize F1
    objective = alarmF1Loss(Y, Ypred);

end
 
%% STEP 5A - Group-Aware CV Evaluation

%Take best hyperparameters
bestParams_ens = resultsBO.XAtMinObjective;
%bestParams_ens.NumLearningCycles=106;
%bestParams_ens.MaxNumSplits=15740;
%bestParams_ens.MinLeafSize=1;

% Apply manual cross-validation loop
% Initialize correctly
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
    rng(2000 + i,"twister");
    
    trainIdx = training(cv_outer, i);
    testIdx  = test(cv_outer, i);
    
     t = templateTree( ...
    'MaxNumSplits', bestParams_ens.MaxNumSplits, ...
    'MinLeafSize', bestParams_ens.MinLeafSize);

    model_fold = fitcensemble( ...
        X(trainIdx,:), Y(trainIdx), ...
        'Method','Bag', ...
        'Learners', t, ...
        'NumLearningCycles', bestParams_ens.NumLearningCycles, ...
        'Cost', costMatrix ...
    );
     
    Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end
confMat = confusionmat(Y, Ypred)
% Step 6A —Compute custom AlarmF1 loss
loss_ens = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (Ens): ', num2str(loss_ens)]);
% Compare models manually based on F1, NOT MATLAB’s error.
% LOWEST alarmF1Loss means highest F1

%Step7A-Train FINAL model (FULL DATA)

t = templateTree( ...
    'MaxNumSplits', bestParams_ens.MaxNumSplits, ...
    'MinLeafSize', bestParams_ens.MinLeafSize);
rng(999,"twister");
finalModel_ensb = fitcensemble( ...
    X, Y, ...
    'Method','Bag', ...
    'Learners', t, ...
    'NumLearningCycles', bestParams_ens.NumLearningCycles, ...
    'Cost', costMatrix ...
);

% This is our final deployed model
disp(finalModel_ensb.ClassNames)
%save finalModel_ensb
finalModel=finalModel_ensb;
%load finalModel_ensb

% After this step for Ensemble Bagged, PASS TO STEP 8A, 
% DO NOT PASS to following STEP 4B

%% STEP 4B-Train Ensemble Boosted (Global Bayesian optimization + Group-Aware CV)

% Define related COST matrix (the best one)
costMatrix = [0 9 10;
              8  0  1;
              9  1  0];

% Global Bayesian optimization for AlarmF1Loss
%AdaBoost
optimVars = [
    optimizableVariable('NumLearningCycles',[50 300],'Type','integer')
    optimizableVariable('LearnRate',[0.01 0.3],'Transform','log')
    optimizableVariable('MaxNumSplits',[2 50],'Type','integer')
    optimizableVariable('MinLeafSize',[1 20],'Type','integer')
];

rng(42)

ObjFcn = @(params) ensembleObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 50, ...
    'IsObjectiveDeterministic', true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel', false);


function objective = ensembleObjectiveFcn(params, X, Y, costMatrix, cv_inner)

Ypred = Y;
Ypred(:) = missing;

    for i = 1:cv_inner.NumTestSets

        % fold-specific deterministic seed
        rng(1000 + i,"twister");

        trainIdx = training(cv_inner, i);
        testIdx  = test(cv_inner, i);
        
        t = templateTree( ...
    'MaxNumSplits', params.MaxNumSplits, ...
    'MinLeafSize', params.MinLeafSize);

        model = fitcensemble( ...
            X(trainIdx,:), Y(trainIdx), ...
            'Method','AdaBoostM2', ...
            'Learners', t, ...
            'NumLearningCycles', params.NumLearningCycles, ...
            'LearnRate', params.LearnRate, ...
            'Cost', costMatrix ...
        );
        
        Ypred(testIdx) = predict(model, X(testIdx,:));
    end
 % 

    % Minimize F1
    objective = alarmF1Loss(Y, Ypred);

end
 
%% STEP 5B - Group-Aware CV Evaluation

%Take best hyperparameters
bestParams_ens = resultsBO.XAtMinObjective;
%bestParams_ens.NumLearningCycles=296;
%bestParams_ens.LearnRate=0.28917;
%bestParams_ens.MaxNumSplits=46;
%bestParams_ens.MinLeafSize=5;
% Apply manual cross-validation loop
% Initialize correctly
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
    rng(2000 + i,"twister");
    
    trainIdx = training(cv_outer, i);
    testIdx  = test(cv_outer, i);
    
    t = templateTree( ...
    'MaxNumSplits', bestParams_ens.MaxNumSplits, ...
    'MinLeafSize', bestParams_ens.MinLeafSize);

    model_fold = fitcensemble( ...
        X(trainIdx,:), Y(trainIdx), ...
        'Method','AdaBoostM2', ...
        'Learners', t, ...
        'NumLearningCycles', bestParams_ens.NumLearningCycles, ...
        'LearnRate', bestParams_ens.LearnRate, ...
        'Cost', costMatrix ...
    );
  
    Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end
confMat = confusionmat(Y, Ypred)
% Step 6B —Compute custom AlarmF1 loss
loss_ens = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (Ens): ', num2str(loss_ens)]);
% Compare models manually based on F1, NOT MATLAB’s error.

%Step7B-Train FINAL model (FULL DATA)

t = templateTree( ...
    'MaxNumSplits', bestParams_ens.MaxNumSplits, ...
    'MinLeafSize', bestParams_ens.MinLeafSize);

rng(999,"twister");
finalModel_ens = fitcensemble( ...
    X, Y, ...
    'Method','AdaBoostM2', ...
    'Learners', t, ...
    'NumLearningCycles', bestParams_ens.NumLearningCycles, ...
    'LearnRate', bestParams_ens.LearnRate, ...
    'Cost', costMatrix ...
);

% This is our final deployed model
disp(finalModel_ens.ClassNames)
%save finalModel_ens
finalModel=finalModel_ens;
%load finalModel_ens
% After this step for Ensemble Boosted, PASS TO STEP 8A, 
% DO NOT PASS to following STEP 4C
%% STEP 4C-Train Bilayered NN (Bayesian optimization + Group CV)

% Define COST matrix
costMatrix = [0 9 10;
7 0 1;
8 1 0];

% Bilayered NN hyperparameters
optimVars = [
optimizableVariable('Layer1',[10 100],'Type','integer')
optimizableVariable('Layer2',[5 50],'Type','integer')
optimizableVariable('Lambda',[1e-5 1],'Transform','log')
];

ObjFcn = @(params) bilayerNNObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
'MaxObjectiveEvaluations', 50, ...
'AcquisitionFunctionName','expected-improvement-plus', ...
'Verbose',1, ...
'UseParallel', false);

% Objective Function
function objective = bilayerNNObjectiveFcn(params, X, Y, costMatrix, cv_inner)
rng(1,"twister"); %Fix the global random seed
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets
rng(i,"twister"); %Fix the global random seed
trainIdx = training(cv_inner, i);
testIdx = test(cv_inner, i);

model = fitcnet( ...
X(trainIdx,:), Y(trainIdx), ...
'LayerSizes', [params.Layer1 params.Layer2], ...
'Lambda', params.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model, X(testIdx,:));
end

objective = alarmF1Loss(Y, Ypred);

end
%% STEP 5C - Group-Aware CV Evaluation

bestParams_bnn = resultsBO.XAtMinObjective;
%bestParams_bnn.Layer1=99;
%bestParams_bnn.Layer2=10;
%bestParams_bnn.Lambda=0.00077676;

Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
rng(i,"twister"); %Fix the global random seed
trainIdx = training(cv_outer, i);
testIdx = test(cv_outer, i);

model_fold = fitcnet( ...
X(trainIdx,:), Y(trainIdx), ...
'LayerSizes', [bestParams_bnn.Layer1 bestParams_bnn.Layer2], ...
'Lambda', bestParams_bnn.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end

confMat = confusionmat(Y, Ypred)

% Step6f-Compute custom loss
loss_bnn = alarmF1Loss(Y, Ypred);

disp(['Cross-validated Alarm F1 Loss (Bilayered NN): ', num2str(loss_bnn)]);

% Compare models manually based on F1, NOT MATLAB’s error.
% Select BEST based on LOWEST alarmF1Loss (i.e., highest F1)

% Step7f-Train FINAL model (FULL DATA)
rng(999,"twister");
finalModel_bnn = fitcnet( ...
X, Y, ...
'LayerSizes', [bestParams_bnn.Layer1 bestParams_bnn.Layer2], ...
'Lambda', bestParams_bnn.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

% This is our final deployed model
disp(finalModel_bnn.ClassNames)

%save finalModel_bnn

finalModel = finalModel_bnn;
%load finalModel_bnn
% After this step for Bilayered NN, PASS TO STEP 8A, 
% DO NOT PASS to following STEP 4D
%% STEP 4D-Train Trilayered NN (Bayesian optimization + Group CV)

% Define COST matrix
costMatrix = [0 9 10;
7 0 1;
8 1 0];

% Opt1 having tighter search ranges to prevent overfitting 
optimVars = [
    optimizableVariable('Layer1',[20 80],'Type','integer')
    optimizableVariable('Layer2',[10 50],'Type','integer')
    optimizableVariable('Layer3',[5 25],'Type','integer')
    optimizableVariable('Lambda',[1e-4 1],'Transform','log')
];

% Trilayered NN hyperparameters
%optimVars = [
%optimizableVariable('Layer1',[20 120],'Type','integer')
%optimizableVariable('Layer2',[10 80],'Type','integer')
%optimizableVariable('Layer3',[5 40],'Type','integer')
%optimizableVariable('Lambda',[1e-5 1],'Transform','log')
%];

ObjFcn = @(params) trilayerNNObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
'MaxObjectiveEvaluations', 50, ...
'AcquisitionFunctionName','expected-improvement-plus', ...
'Verbose',1, ...
'UseParallel', false);

% Objective Function
function objective = trilayerNNObjectiveFcn(params, X, Y, costMatrix, cv_inner)
rng(1,"twister"); %Fix the global random seed
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets
rng(i,"twister"); %Fix the global random seed
trainIdx = training(cv_inner, i);
testIdx = test(cv_inner, i);

model = fitcnet( ...
X(trainIdx,:), Y(trainIdx), ...
'LayerSizes', [params.Layer1 params.Layer2 params.Layer3], ...
'Lambda', params.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model, X(testIdx,:));
end

objective = alarmF1Loss(Y, Ypred);

end

%% STEP 5D - Group-Aware CV Evaluation

bestParams_tnn = resultsBO.XAtMinObjective;
%bestParams_tnn.Layer1=80;
%bestParams_tnn.Layer2=22;
%bestParams_tnn.Layer3=20;
%bestParams_tnn.Lambda=0.00065118;

Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
rng(i,"twister"); %Fix the global random seed
trainIdx = training(cv_outer, i);
testIdx = test(cv_outer, i);

model_fold = fitcnet( ...
X(trainIdx,:), Y(trainIdx), ...
'LayerSizes', [bestParams_tnn.Layer1 bestParams_tnn.Layer2 bestParams_tnn.Layer3], ...
'Lambda', bestParams_tnn.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end

confMat = confusionmat(Y, Ypred)

% Step6f-Compute custom loss
loss_tnn = alarmF1Loss(Y, Ypred);

disp(['Cross-validated Alarm F1 Loss (Trilayered NN): ', num2str(loss_tnn)]);

% Compare models manually based on F1, NOT MATLAB’s error.
% Select BEST based on LOWEST alarmF1Loss (i.e., highest F1)

% Step7f-Train FINAL model (FULL DATA)
rng(999,"twister");
finalModel_tnn = fitcnet( ...
X, Y, ...
'LayerSizes', [bestParams_tnn.Layer1 bestParams_tnn.Layer2 bestParams_tnn.Layer3], ...
'Lambda', bestParams_tnn.Lambda, ...
'Standardize', true, ...
'Cost', costMatrix ...
);

% This is our final deployed model
disp(finalModel_tnn.ClassNames)

%save finalModel_tnn

finalModel = finalModel_tnn;

% After this step for Trilayered NN, PASS TO STEP 8A, 
% DO NOT PASS to following STEP 4E
%% STEP 4E-Train Quadratic Discriminant
% Bayesian Optimization + Group CV

% Define COST matrix
costMatrix = [0 9 10;
              7 0 1;
              8 1 0];

% Optimize discriminant covariance/model structure
optimVars = [
    optimizableVariable('DiscrimType', ...
        {'linear','diaglinear','quadratic','diagquadratic'}, ...
        'Type','categorical')
];

ObjFcn = @(params) discrimObjectiveFcn( ...
    params, X, Y, costMatrix, cv_inner);

rng(10,"twister");

resultsBO = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations',50, ...
    'IsObjectiveDeterministic',true, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel',false);

function objective = discrimObjectiveFcn( ...
    params, X, Y, costMatrix, cv_inner)

Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets

    trainIdx = training(cv_inner,i);
    testIdx  = test(cv_inner,i);

    model = fitcdiscr( ...
        X(trainIdx,:), ...
        Y(trainIdx), ...
        'DiscrimType',char(params.DiscrimType), ...
        'Cost',costMatrix, ...
        'FillCoeffs','off');

    Ypred(testIdx) = predict( ...
        model,X(testIdx,:));

end

objective = alarmF1Loss(Y,Ypred);

end

%% STEP 5E - Group-Aware CV Evaluation

bestParams_discrim = resultsBO.XAtMinObjective;
%bestParams_discrim.DiscrimType='quadratic';
disp('Best Discriminant Type:');
disp(bestParams_discrim.DiscrimType);

Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets

    rng(i,"twister");
    trainIdx = training(cv_outer,i);
    testIdx  = test(cv_outer,i);

    model_fold = fitcdiscr( ...
        X(trainIdx,:), ...
        Y(trainIdx), ...
        'DiscrimType',char(bestParams_discrim.DiscrimType), ...
        'Cost',costMatrix, ...
        'FillCoeffs','off');

    Ypred(testIdx) = predict( ...
        model_fold,X(testIdx,:));

end

confMat = confusionmat(Y,Ypred)

loss_discrim = alarmF1Loss(Y,Ypred);

disp(['Cross-validated Alarm F1 Loss (Discriminant): ', ...
    num2str(loss_discrim)]);

% STEP7G-Train FINAL Discriminant Model

rng(999,"twister");

finalModel_discrim = fitcdiscr( ...
    X, ...
    Y, ...
    'DiscrimType',char(bestParams_discrim.DiscrimType), ...
    'Cost',costMatrix, ...
    'FillCoeffs','off');

disp(finalModel_discrim.ClassNames)
% save finalModel_discrim

finalModel = finalModel_discrim;


%load finalModel_discrim
% After this step for Quadratic Discriminant, PASS TO STEP 8A, 
% DO NOT PASS to following Validation & Test evaluation codes
%% Validation & Test evaluation (DO NOT RUN, just for comparison)
% (IF BOTH NO THRESHOLD TUNING and NO ENGINE-LEVEL DECISION, THEN APPLY THIS)

% Validation
Xval = ValidationData1sfnr(:, predictorNames);
Yval = ValidationData1sfnr.RUL;
Yval = categorical(lower(string(Yval)), classOrder);

Yval_pred = predict(finalModel, Xval);
Yval_pred = categorical(Yval_pred, classOrder);
Xp_val=tabulate(Yval_pred);
%Number of predicted classes in 3752 validation data
%Number of true classes (e.g. 260 alarms) in 3752 validation data
C_val=confusionmat(Yval,Yval_pred)
confusionchart(Yval,Yval_pred);
AlarmF1Loss_val=alarmF1Loss(Yval, Yval_pred)

% Test
Xtest = TestData1sfnr(:, predictorNames);
Ytest = TestData1sfnr.RUL;
Ytest = categorical(lower(string(Ytest)), classOrder);

Ytest_pred = predict(finalModel, Xtest);
Ytest_pred = categorical(Ytest_pred, classOrder);
Xp_test=tabulate(Ytest_pred);
%Number of predicted classes in 12096 test data

%Number of true classes (e.g. 33 alarms) in 12096 test data
%C_test=confusionmat(Ytest,Ytest_pred);
confusionchart(Ytest,Ytest_pred);
AlarmF1Loss_test=alarmF1Loss(Ytest, Ytest_pred)

%% STEP 8A-THRESHOLD TUNING for Alarm Class (VALIDATION)
classOrder = ["alarm","warning","normal"];
Xval = ValidationData1sfnr(:, predictorNames);
Yval = ValidationData1sfnr.RUL;
%Force category order
Yval = categorical(lower(string(Yval)), classOrder);

% Apply Threshold Tuning (Find best threshold for Alarm class
% with Recall ≥ Precision ≥ 0.7 constraint)
[label, score] = predict(finalModel, Xval);

% Control Class order
disp('Class order:'); disp(finalModel.ClassNames);
% Alarm class index
alarmIdx = find(finalModel.ClassNames == "alarm");
score_alarm = score(:, alarmIdx);

thresholds = quantile(score_alarm, 0.5:0.001:0.999);

%initialize
bestF1 = 0;
bestThreshold = 0.5;

results_thresh = [];

for t = thresholds
    
    Ypred_thresh = categorical(repmat("normal", size(Yval)), classOrder);
    % Threshold-based Alarm decision
    Ypred_thresh(score_alarm >= t) = "alarm";
    Ypred_thresh = categorical(Ypred_thresh, classOrder);
       
    % Metrics
    [recall, precision, F1] = alarmMetrics(Yval, Ypred_thresh);
    
    results_thresh = [results_thresh; t recall precision F1];
    
    if recall >= precision && precision >= 0.7 && F1 > bestF1
    bestF1 = F1;
    bestThreshold = t;
end
end

disp('Threshold tuning results:');
disp(array2table(results_thresh, ...
    'VariableNames', {'Threshold','Recall','Precision','F1'}));

disp(['Best Threshold: ', num2str(bestThreshold)]);
disp(['Best Validation F1: ', num2str(bestF1)]);

function [recall, precision, F1] = alarmMetrics(Ytrue, Ypred)

    classOrder = ["alarm","warning","normal"];
    Ytrue = categorical(lower(string(Ytrue)), classOrder);
    Ypred = categorical(lower(string(Ypred)), classOrder);

    alarmClass = "alarm";
  
    tp = sum((Ytrue == alarmClass) & (Ypred == alarmClass));
    fn = sum((Ytrue == alarmClass) & (Ypred ~= alarmClass));
    fp = sum((Ytrue ~= alarmClass) & (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);
    precision = tp / (tp + fp + eps);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

end

tAlarm_fixed = bestThreshold;
%tAlarm_fixed =0.5  %for No threshold calibration

%% STEP 8B-THRESHOLD TUNING FOR WARNING CLASS (VALIDATION)

classOrder = ["alarm","warning","normal"];

Xval = ValidationData1sfnr(:, predictorNames);
Yval = categorical(lower(string(ValidationData1sfnr.RUL)), classOrder);

[label, score] = predict(finalModel, Xval);

disp('Class order:'); 
disp(finalModel.ClassNames);

% Class Indexes
alarmIdx   = find(finalModel.ClassNames == "alarm");
warningIdx = find(finalModel.ClassNames == "warning");

score_alarm   = score(:, alarmIdx);
score_warning = score(:, warningIdx);

% Optimize only for warning 
% Search range
thresholdsW = quantile(score_warning, 0.3:0.001:0.95);

bestWF1 = 0;
bestTW = 0;

results_thresh = [];

for tW = thresholdsW
    % Initialize all observations as normal
    Ypred = categorical(repmat("normal", size(Yval)), classOrder);

    % Step 1: Alarm decisions (fixed)
    idxA = score_alarm >= tAlarm_fixed;
    Ypred(idxA) = "alarm";

    % Step 2: Warning decisions (outside the alarm region)
    idxW = (~idxA) & (score_warning >= tW);
    Ypred(idxW) = "warning";

    % Alarm metrics
    [recall, precision, F1] = classMetrics(Yval,Ypred,"alarm");

    % Warning metrics
    [warningRecall, warningPrecision, warningF1] = classMetrics(Yval,Ypred,"warning");

    results_thresh = [results_thresh;
    tW ...
    recall precision F1 warningRecall warningPrecision warningF1];

    % Preserve alarm performance
    if recall >= 0.8 && precision >= 0.70 && warningF1 > bestWF1
        bestWF1 = warningF1;
        bestTW = tW;
    end

end

disp(array2table(results_thresh,...
'VariableNames',...
{'tWarning',...
'AlarmRecall','AlarmPrecision','AlarmF1',...
'WarningRecall','WarningPrecision','WarningF1'}));

disp(['Fixed tAlarm: ',num2str(tAlarm_fixed)]);
disp(['Best tWarning: ',num2str(bestTW)]);
disp(['Best Warning F1: ',num2str(bestWF1)]);

function [recall,precision,F1] = classMetrics(Ytrue,Ypred,targetClass)

    classOrder = ["alarm","warning","normal"];

    Ytrue = categorical(lower(string(Ytrue)),classOrder);
    Ypred = categorical(lower(string(Ypred)),classOrder);

    tp = sum((Ytrue == targetClass) & (Ypred == targetClass));

    fn = sum((Ytrue == targetClass) & (Ypred ~= targetClass));

    fp = sum((Ytrue ~= targetClass) & (Ypred == targetClass));

    recall = tp/(tp+fn+eps);

    precision = tp/(tp+fp+eps);

    F1 = 2*(precision*recall)/(precision+recall+eps);

end

confusionchart(Yval, Ypred);
%Validation Confusion Matrix is appeared.
Xp_val=tabulate(Ypred)

%% STEP 9 — APPLY THRESHOLD ON TEST WITH WARNING CLASS 

%Calibrated Thresholds for Ensemble Bagged Trees
%tAlarm_fixed=0.39981;
%bestTW=0.36487;

% For No Threshold Calibration
%tAlarm_fixed=0.5;
%bestTW=0.5;

classOrder = ["alarm","warning","normal"];

Xtest = TestData1sfnr(:, predictorNames);
Ytest = categorical(lower(string(TestData1sfnr.RUL)), classOrder);

[label_test, score_test] = predict(finalModel, Xtest);


alarmIdx   = find(finalModel.ClassNames == "alarm");
warningIdx = find(finalModel.ClassNames == "warning");

score_alarm_test   = score_test(:, alarmIdx);
score_warning_test = score_test(:, warningIdx);

Ypred_test = categorical(repmat("normal", size(Ytest)), classOrder);

% Alarm
idxA = score_alarm_test >= tAlarm_fixed;
Ypred_test(idxA) = "alarm";

% Warning
idxW = (~idxA) & (score_warning_test >= bestTW);
Ypred_test(idxW) = "warning";


% Metrics
AlarmF1Loss_test = alarmF1Loss(Ytest, Ypred_test)

[recall_test, precision_test, F1_test] = alarmMetrics1(Ytest, Ypred_test)

function [recall, precision, F1] = alarmMetrics1(Ytrue, Ypred)

    classOrder = ["alarm","warning","normal"];
    Ytrue = categorical(lower(string(Ytrue)), classOrder);
    Ypred = categorical(lower(string(Ypred)), classOrder);

    alarmClass = "alarm";
  
    tp = sum((Ytrue == alarmClass) & (Ypred == alarmClass));
    fn = sum((Ytrue == alarmClass) & (Ypred ~= alarmClass));
    fp = sum((Ytrue ~= alarmClass) & (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);
    precision = tp / (tp + fp + eps);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

end

confusionchart(Ytest, Ypred_test);
%Test Confusion Matrix is appeared.
Xp_test=tabulate(Ypred_test)

% After this step, PASS TO STEP 10B, 
% DO NOT PASS to following Ablation Analysis codes
%% FRAMEWORK ABLATION ANALYSIS (DO NOT RUN, just for comparison)
% IF NO THRESHOLD TUNING THEN APPLY THIS for cycle-level test results
% Compare the result of this with the ones of selected baseline threshold values as 0.5
Ytest = TestData1sfnr.RUL;
Ytest = categorical(lower(string(Ytest)), classOrder);

Xtest = TestData1sfnr(:, predictorNames);

[label_test, score_test] = predict(finalModel, Xtest);

% Default classifier predictions
Ypred_test = categorical(label_test, classOrder);

Xp_test = tabulate(Ypred_test);

confusionchart(Ytest, Ypred_test);

AlarmF1Loss_test = alarmF1Loss(Ytest, Ypred_test);
%% 10A-ENGINE-LEVEL TRANSFORMATION (Basic worst-case/any-alarm engine aggregation)
% FRAMEWORK ABLATION ANALYSIS (WITHOUT TAEI)

engineIDs = unique(TestData1sfnr.Engine_ID);

engine_pred_noStrategy = strings(length(engineIDs),1);
engine_true = strings(length(engineIDs),1);

for i = 1:length(engineIDs)

    idx = TestData1sfnr.Engine_ID == engineIDs(i);

    % True engine label from final cycle
    engine_true(i) = string(Ytest(find(idx,1,'last')));

    % Basic worst-case aggregation using the SAME
    % threshold-calibrated cycle-level predictions
    if any(Ypred_test(idx) == "alarm")

        engine_pred_noStrategy(i) = "alarm";

    elseif any(Ypred_test(idx) == "warning")

        engine_pred_noStrategy(i) = "warning";

    else

        engine_pred_noStrategy(i) = "normal";

    end
end

engine_true = categorical(engine_true, classOrder);
engine_pred_noStrategy = categorical(engine_pred_noStrategy, classOrder);

[recall_e, precision_e, F1_e] = ...
    alarmMetric2(engine_true, engine_pred_noStrategy)

confusionchart(engine_true, engine_pred_noStrategy)
%Engine-level Test Confusion Matrix is appeared.
tabulate(engine_pred_noStrategy)

% Finding Predicted Alarm Engines
Engines_Compare=table(engineIDs,engine_true,engine_pred_noStrategy);
Engines_Alarm=Engines_Compare(engine_pred_noStrategy=='alarm',:)
%Predicted alarm engines' numbers are appeared without prioritization.

% Metric Function

function [recall, precision, F1] = alarmMetric2(Ytrue, Ypred)

    classOrder = ["alarm","warning","normal"];
    Ytrue = categorical(lower(string(Ytrue)), classOrder);
    Ypred = categorical(lower(string(Ypred)), classOrder);

    alarmClass = "alarm";
  
    tp = sum((Ytrue == alarmClass) & (Ypred == alarmClass));
    fn = sum((Ytrue == alarmClass) & (Ypred ~= alarmClass));
    fp = sum((Ytrue ~= alarmClass) & (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);
    precision = tp / (tp + fp + eps);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

end

%% STEP 10B — ENGINE LEVEL TRANSFORMATION (TAEI DECISION STAGE)

engineIDs = unique(TestData1sfnr.Engine_ID);

engine_pred = strings(length(engineIDs),1);
engine_true = strings(length(engineIDs),1);

for i = 1:length(engineIDs)

    idx = TestData1sfnr.Engine_ID == engineIDs(i);
    
    % Scores
    scores_e = score_alarm_test(idx);
    scores_w = score_warning_test(idx);
    
    % TRUE label (last cycle)
    true_label = Ytest(find(idx,1,'last'));
    engine_true(i) = string(true_label);

    % TIME-AWARE WINDOW 
    N = length(scores_e);
    last_window = max(1, round(0.2 * N));

    start_idx = max(1, N - last_window + 1);
    recent_scores = scores_e(start_idx:N);

    % ALARM Features

    % Top-K in recent window
    K = 2;
    recent_sorted = sort(recent_scores, 'descend');

    if length(recent_sorted) >= K
        top_recent = mean(recent_sorted(1:K));
    else
        top_recent = mean(recent_sorted);
    end

    % GLOBAL STRONG SIGNAL (fallback)
    K_global = 2;
    global_sorted = sort(scores_e, 'descend');

    if length(global_sorted) >= K_global
        top_global = mean(global_sorted(1:K_global));
    else
        top_global = mean(global_sorted);
    end

    % WARNING Features
    Kw = 2;
    scores_w_sorted = sort(scores_w, 'descend');

    if length(scores_w_sorted) >= Kw
        topW = mean(scores_w_sorted(1:Kw));
    else
        topW = mean(scores_w_sorted);
    end



    % CONSISTENCY FEATURE
    recent_ratio = sum(recent_scores >= tAlarm_fixed) / length(recent_scores);

    % FINAL DECISION 

    if (top_recent >= tAlarm_fixed) && (recent_ratio >= 0.05)
        % time-aware alarm
        engine_pred(i) = "alarm";

    elseif (top_global >= tAlarm_fixed * 1.2) && (recent_ratio >= 0.03)
        % very strong signal (rescues late spikes)
        engine_pred(i) = "alarm";

    elseif topW >= bestTW
        engine_pred(i) = "warning";

    else
        engine_pred(i) = "normal";
    end
   
end

% Metrics

engine_true = categorical(engine_true, ["alarm","warning","normal"]);
engine_pred = categorical(engine_pred, ["alarm","warning","normal"]);

[recall_e, precision_e, F1_e] = alarmMetric(engine_true, engine_pred)

confusionchart(engine_true, engine_pred)
%Engine-level Test Confusion Matrix is appeared.
tabulate(engine_pred)

% Metric Function

function [recall, precision, F1] = alarmMetric(Ytrue, Ypred)

    classOrder = ["alarm","warning","normal"];
    Ytrue = categorical(lower(string(Ytrue)), classOrder);
    Ypred = categorical(lower(string(Ypred)), classOrder);

    alarmClass = "alarm";
  
    tp = sum((Ytrue == alarmClass) & (Ypred == alarmClass));
    fn = sum((Ytrue == alarmClass) & (Ypred ~= alarmClass));
    fp = sum((Ytrue ~= alarmClass) & (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);
    precision = tp / (tp + fp + eps);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

end
% Finding Predicted Alarm Engines
Engines_Compare=table(engineIDs,engine_true,engine_pred);
Engines_Alarm=Engines_Compare(engine_pred=='alarm',:)
%Predicted alarm engines' numbers are appeared without prioritization.

%% ENGINE LEVEL DECISION STAGE+ BRANCH ACTIVATION AUDIT

% load alarmF1Loss_test %for Ensemble Bagged Trees

engineIDs = unique(TestData1sfnr.Engine_ID);
nEngines = length(engineIDs);

% Preallocate outputs
engine_pred = strings(nEngines,1);
engine_true = strings(nEngines,1);
decision_branch = strings(nEngines,1);

% Diagnostic variables
top_recent_values    = zeros(nEngines,1);
top_global_values    = zeros(nEngines,1);
recent_ratio_values  = zeros(nEngines,1);
top_warning_values   = zeros(nEngines,1);

% Optional diagnostic conditions
recent_alarm_condition  = false(nEngines,1);
global_alarm_condition  = false(nEngines,1);
warning_condition       = false(nEngines,1);

for i = 1:nEngines

    idx = TestData1sfnr.Engine_ID == engineIDs(i);
    
    % Scores
    scores_e = score_alarm_test(idx);
    scores_w = score_warning_test(idx);
    
    % TRUE label (last cycle)
    true_label = Ytest(find(idx,1,'last'));
    engine_true(i) = string(true_label);

    % TIME-AWARE WINDOW 
    N = length(scores_e);
    window_fraction = 0.20;
    last_window = max(1, round(window_fraction * N));

    start_idx = max(1, N - last_window + 1);
    recent_scores = scores_e(start_idx:N);

    % ALARM FEATURE 1:
    % TOP-K RECENT ALARM EVIDENCE (window)
    K = 2;
    recent_sorted = sort(recent_scores, 'descend');

    if length(recent_sorted) >= K
        top_recent = mean(recent_sorted(1:K));
    else
        top_recent = mean(recent_sorted);
    end

    % ALARM FEATURE 2:
    % GLOBAL STRONG SIGNAL (fallback)
    K_global = 2;
    global_sorted = sort(scores_e, 'descend');

    if length(global_sorted) >= K_global
        top_global = mean(global_sorted(1:K_global));
    else
        top_global = mean(global_sorted);
    end

    % WARNING Feature
    Kw = 2;
    scores_w_sorted = sort(scores_w, 'descend');

    if length(scores_w_sorted) >= Kw
        topW = mean(scores_w_sorted(1:Kw));
    else
        topW = mean(scores_w_sorted);
    end



    % CONSISTENCY FEATURE
    recent_ratio = sum(recent_scores >= tAlarm_fixed) / length(recent_scores);

    % STORE DIAGNOSTIC VALUES
    top_recent_values(i)   = top_recent;
    top_global_values(i)   = top_global;
    recent_ratio_values(i) = recent_ratio;
    top_warning_values(i)  = topW;

    % DECISION CONDITIONS
    recent_alarm_condition(i) = (top_recent >= tAlarm_fixed) && (recent_ratio >= 0.05);

    global_alarm_condition(i) = (top_global >= tAlarm_fixed * 1.2) && (recent_ratio >= 0.03);

    warning_condition(i) = (topW >= bestTW);

    % FINAL HIERARCHICAL DECISION
    if recent_alarm_condition(i)

        engine_pred(i) = "alarm";
        decision_branch(i) = "recent_alarm";

    elseif global_alarm_condition(i)

        engine_pred(i) = "alarm";
        decision_branch(i) = "global_fallback";

    elseif warning_condition(i)

        engine_pred(i) = "warning";
        decision_branch(i) = "warning";

    else

        engine_pred(i) = "normal";
        decision_branch(i) = "normal";

    end
end

% Metrics
engine_true_cat = categorical(engine_true, ...
    ["alarm","warning","normal"]);

engine_pred_cat = categorical(engine_pred, ...
    ["alarm","warning","normal"]);

[recall_e, precision_e, F1_e] = ...
    alarmMetricR(engine_true_cat, engine_pred_cat)

% CONFUSION MATRIX
figure;
confusionchart(engine_true_cat, engine_pred_cat);
%Engine-level Test Confusion Matrix is appeared.

%BRANCH ACTIVATION SUMMARY
disp('Predicted classes:')
tabulate(engine_pred)

disp('Decision branches:')
tabulate(decision_branch)

% DIAGNOSTIC TABLE
diagnosticTable = table( ...
    engineIDs, ...
    engine_true, ...
    engine_pred, ...
    top_recent_values, ...
    top_global_values, ...
    recent_ratio_values, ...
    top_warning_values, ...
    recent_alarm_condition, ...
    global_alarm_condition, ...
    warning_condition, ...
    decision_branch);

disp(diagnosticTable)

%table(engineIDs, engine_true, engine_pred, decision_branch);


% Metric Function

function [recall, precision, F1] = alarmMetricR(Ytrue, Ypred)

    classOrder = ["alarm","warning","normal"];
    Ytrue = categorical(lower(string(Ytrue)), classOrder);
    Ypred = categorical(lower(string(Ypred)), classOrder);

    alarmClass = "alarm";
  
    tp = sum((Ytrue == alarmClass) & (Ypred == alarmClass));
    fn = sum((Ytrue == alarmClass) & (Ypred ~= alarmClass));
    fp = sum((Ytrue ~= alarmClass) & (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);
    precision = tp / (tp + fp + eps);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

end

% Show he results 
fprintf('Recent alarm condition satisfied: %d\n', ...
    sum(recent_alarm_condition));

fprintf('Global alarm condition satisfied: %d\n', ...
    sum(global_alarm_condition));

fprintf('Global only (fallback uniquely needed): %d\n', ...
    sum(global_alarm_condition & ~recent_alarm_condition));

fprintf('Both recent and global conditions satisfied: %d\n', ...
    sum(global_alarm_condition & recent_alarm_condition));
diagnosticTable( ...
    global_alarm_condition & ~recent_alarm_condition, :)
diagnosticTable( ...
    global_alarm_condition & recent_alarm_condition, :)
%% FIVE-CONFIGURATION ABLATION STUDY
%clc;
%clear;
% load alarmF1Loss_test %for Ensemble Bagged Trees

% Components:
%
% R = Recent alarm strength
%     top_recent >= tAlarm_fixed
%
% C = Recent alarm consistency
%     recent_ratio >= 0.05
%
% G = Global strong-evidence fallback
%     top_global >= 1.2*tAlarm_fixed
%     AND recent_ratio >= 0.03
%
% Configurations:
%
% B0 = Neither R nor C nor G
% B1 = R only
% B2 = C only
% B3 = R + C
% B4 = R + C + G   --> Full proposed mechanism
%
% Warning branch is applied when the corresponding alarm
% condition is not satisfied and topW >= bestTW.
%

engineIDs = unique(TestData1sfnr.Engine_ID);
nEngines = length(engineIDs);

% ---------------------------------------------------------------
% PREALLOCATE
% ---------------------------------------------------------------

engine_true = strings(nEngines,1);

pred_B0 = strings(nEngines,1);
pred_B1 = strings(nEngines,1);
pred_B2 = strings(nEngines,1);
pred_B3 = strings(nEngines,1);
pred_B4 = strings(nEngines,1);

% Diagnostic variables
top_recent_values  = zeros(nEngines,1);
top_global_values  = zeros(nEngines,1);
recent_ratio_values = zeros(nEngines,1);
top_warning_values = zeros(nEngines,1);

recent_condition = false(nEngines,1);
consistency_condition = false(nEngines,1);
global_condition = false(nEngines,1);
warning_condition = false(nEngines,1);

% ---------------------------------------------------------------
% ENGINE-LEVEL LOOP
% ---------------------------------------------------------------

for i = 1:nEngines

    idx = TestData1sfnr.Engine_ID == engineIDs(i);

    % Scores
    scores_e = score_alarm_test(idx);
    scores_w = score_warning_test(idx);

    % TRUE ENGINE LABEL
    true_label = Ytest(find(idx,1,'last'));
    engine_true(i) = string(true_label);

    % -----------------------------------------------------------
    % TIME-AWARE WINDOW
    % ------------------------------------------------------------

    N = length(scores_e);

    last_window = max(1, round(0.20 * N));

    start_idx = max(1, N - last_window + 1);

    recent_scores = scores_e(start_idx:N);

    % -----------------------------------------------------------
    % R — TOP-K RECENT ALARM STRENGTH
    % ------------------------------------------------------------

    K = 2;

    recent_sorted = sort(recent_scores,'descend');

    if length(recent_sorted) >= K
        top_recent = mean(recent_sorted(1:K));
    else
        top_recent = mean(recent_sorted);
    end

    % -----------------------------------------------------------
    % GLOBAL ALARM STRENGTH
    % ------------------------------------------------------------

    K_global = 2;

    global_sorted = sort(scores_e,'descend');

    if length(global_sorted) >= K_global
        top_global = mean(global_sorted(1:K_global));
    else
        top_global = mean(global_sorted);
    end

    % -----------------------------------------------------------
    % WARNING STRENGTH
    % ------------------------------------------------------------

    Kw = 2;

    scores_w_sorted = sort(scores_w,'descend');

    if length(scores_w_sorted) >= Kw
        topW = mean(scores_w_sorted(1:Kw));
    else
        topW = mean(scores_w_sorted);
    end

    % -----------------------------------------------------------
    % C — RECENT ALARM CONSISTENCY
    % ------------------------------------------------------------

    recent_ratio = ...
        sum(recent_scores >= tAlarm_fixed) / length(recent_scores);

    % -----------------------------------------------------------
    % STORE DIAGNOSTIC VALUES
    % ------------------------------------------------------------

    top_recent_values(i) = top_recent;
    top_global_values(i) = top_global;
    recent_ratio_values(i) = recent_ratio;
    top_warning_values(i) = topW;

    % -----------------------------------------------------------
    % THREE BASIC CONDITIONS
    % ------------------------------------------------------------

    R = (top_recent >= tAlarm_fixed);

    C = (recent_ratio >= 0.05);

    % Global fallback condition
    G = (top_global >= tAlarm_fixed * 1.2) && ...
        (recent_ratio >= 0.03);

    W = (topW >= bestTW);

    recent_condition(i) = R;
    consistency_condition(i) = C;
    global_condition(i) = G;
    warning_condition(i) = W;

    % ===========================================================
    % B0 — NEITHER R NOR C NOR G
    % ===========================================================

    if W
        pred_B0(i) = "warning";
    else
        pred_B0(i) = "normal";
    end

    % ===========================================================
    % B1 — RECENT STRENGTH ONLY
    % ===========================================================

    if R
        pred_B1(i) = "alarm";

    elseif W
        pred_B1(i) = "warning";

    else
        pred_B1(i) = "normal";
    end

    % ===========================================================
    % B2 — CONSISTENCY ONLY
    % ===========================================================

    if C
        pred_B2(i) = "alarm";

    elseif W
        pred_B2(i) = "warning";

    else
        pred_B2(i) = "normal";
    end

    % ===========================================================
    % B3 — RECENT STRENGTH + CONSISTENCY
    % ===========================================================

    if R && C
        pred_B3(i) = "alarm";

    elseif W
        pred_B3(i) = "warning";

    else
        pred_B3(i) = "normal";
    end

    % ===========================================================
    % B4 — FULL PROPOSED:
    %      RECENT + CONSISTENCY + GLOBAL FALLBACK
    % ===========================================================

    if R && C

        % Primary time-aware alarm
        pred_B4(i) = "alarm";

    elseif G

        % Global strong-evidence rescue
        pred_B4(i) = "alarm";

    elseif W

        pred_B4(i) = "warning";

    else

        pred_B4(i) = "normal";

    end

end

% ---------------------------------------------------------------
% CONVERT TO CATEGORICAL
% ---------------------------------------------------------------

classOrder = ["alarm","warning","normal"];

Ytrue = categorical(engine_true,classOrder);

Y_B0 = categorical(pred_B0,classOrder);
Y_B1 = categorical(pred_B1,classOrder);
Y_B2 = categorical(pred_B2,classOrder);
Y_B3 = categorical(pred_B3,classOrder);
Y_B4 = categorical(pred_B4,classOrder);

% ---------------------------------------------------------------
% CALCULATE METRICS
% ---------------------------------------------------------------

[Recall_B0, Precision_B0, F1_B0] = ...
    alarmMetricR5(Ytrue,Y_B0);

[Recall_B1, Precision_B1, F1_B1] = ...
    alarmMetricR5(Ytrue,Y_B1);

[Recall_B2, Precision_B2, F1_B2] = ...
    alarmMetricR5(Ytrue,Y_B2);

[Recall_B3, Precision_B3, F1_B3] = ...
    alarmMetricR5(Ytrue,Y_B3);

[Recall_B4, Precision_B4, F1_B4] = ...
    alarmMetricR5(Ytrue,Y_B4);

% ---------------------------------------------------------------
% ALARM COUNTS
% ---------------------------------------------------------------

AlarmCount_B0 = sum(Y_B0 == "alarm");
AlarmCount_B1 = sum(Y_B1 == "alarm");
AlarmCount_B2 = sum(Y_B2 == "alarm");
AlarmCount_B3 = sum(Y_B3 == "alarm");
AlarmCount_B4 = sum(Y_B4 == "alarm");

% ---------------------------------------------------------------
% RESULTS TABLE
% ---------------------------------------------------------------

Configuration = [
    "B0_Neither"
    "B1_RecentOnly"
    "B2_ConsistencyOnly"
    "B3_Both"
    "B4_Both_GlobalFallback"
    ];

EngineRecall = [
    Recall_B0
    Recall_B1
    Recall_B2
    Recall_B3
    Recall_B4
    ];

EnginePrecision = [
    Precision_B0
    Precision_B1
    Precision_B2
    Precision_B3
    Precision_B4
    ];

EngineF1 = [
    F1_B0
    F1_B1
    F1_B2
    F1_B3
    F1_B4
    ];

AlarmCount = [
    AlarmCount_B0
    AlarmCount_B1
    AlarmCount_B2
    AlarmCount_B3
    AlarmCount_B4
    ];

ablationResults = table( ...
    Configuration, ...
    EngineRecall, ...
    EnginePrecision, ...
    EngineF1, ...
    AlarmCount);

disp(ablationResults);

% ---------------------------------------------------------------
% ENGINE-LEVEL DIAGNOSTIC TABLE
% ---------------------------------------------------------------

diagnosticTable_B = table( ...
    engineIDs, ...
    engine_true, ...
    pred_B0, ...
    pred_B1, ...
    pred_B2, ...
    pred_B3, ...
    pred_B4, ...
    top_recent_values, ...
    top_global_values, ...
    recent_ratio_values, ...
    top_warning_values, ...
    recent_condition, ...
    consistency_condition, ...
    global_condition, ...
    warning_condition);

disp(diagnosticTable_B);

% ---------------------------------------------------------------
% METRIC FUNCTION
% ---------------------------------------------------------------

function [recall, precision, F1] = alarmMetricR5(Ytrue,Ypred)

    classOrder = ["alarm","warning","normal"];

    Ytrue = categorical(lower(string(Ytrue)),classOrder);
    Ypred = categorical(lower(string(Ypred)),classOrder);

    alarmClass = "alarm";

    tp = sum((Ytrue == alarmClass) & ...
             (Ypred == alarmClass));

    fn = sum((Ytrue == alarmClass) & ...
             (Ypred ~= alarmClass));

    fp = sum((Ytrue ~= alarmClass) & ...
             (Ypred == alarmClass));

    recall = tp / (tp + fn + eps);

    precision = tp / (tp + fp + eps);

    F1 = 2 * (precision * recall) / ...
         (precision + recall + eps);

end
% B1 vs B2

idx_B1_B2 = pred_B1 ~= pred_B2;

fprintf('B1 vs B2 disagreement count: %d\n', ...
    sum(idx_B1_B2));

diagnosticTable_B(idx_B1_B2,:)

% B1 vs B3

idx_B1_B3 = pred_B1 ~= pred_B3;

fprintf('B1 vs B3 disagreement count: %d\n', ...
    sum(idx_B1_B3));

diagnosticTable_B(idx_B1_B3,:)

% B2 vs B3

idx_B2_B3 = pred_B2 ~= pred_B3;

fprintf('B2 vs B3 disagreement count: %d\n', ...
    sum(idx_B2_B3));

diagnosticTable_B(idx_B2_B3,:)

% B3 vs B4 — EFFECT OF GLOBAL FALLBACK

idx_B3_B4 = pred_B3 ~= pred_B4;

fprintf('B3 vs B4 disagreement count: %d\n', ...
    sum(idx_B3_B4));

diagnosticTable_B(idx_B3_B4,:)

% Which engines are rescued by global fallback?

rescued = ...
    (pred_B3 ~= "alarm") & ...
    (pred_B4 == "alarm");

EnginesRescuedByGlobalFallback=diagnosticTable_B(rescued,:)


% Which global-fallback alarms are false?

global_false_alarm = ...
    (pred_B3 ~= "alarm") & ...
    (pred_B4 == "alarm") & ...
    (engine_true ~= "alarm");


% Which global-fallback alarms are true alarms?

global_true_alarm = ...
    (pred_B3 ~= "alarm") & ...
    (pred_B4 == "alarm") & ...
    (engine_true == "alarm");

TrueGlobalFallbackAlarms=diagnosticTable_B(global_true_alarm,:)

FalseGlobalFallbackAlarms=diagnosticTable_B(global_false_alarm,:)


