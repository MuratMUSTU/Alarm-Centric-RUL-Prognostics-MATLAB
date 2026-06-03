%% SECOND PIPELINE
clc;
clear;
load TrainData1sfnr

%For Ablation Study of no preprocessing (using raw data)
%TrainData1sfnr=TrainData1_raw;
%ValidationData1sfnr=ValidationData1_raw;
%TestData1sfnr=TestData1_raw;


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


%% Step4a-Train Ensemble Bag (Global Bayesian optimization + Group-Aware CV)

% Define related COST matrix (the best one)
costMatrix = [0 4 5;
              3  0  1;
              4  1  0];

% Global Bayesian optimization for AlarmF1Loss
%Bag
optimVars = [
    optimizableVariable('NumLearningCycles',[100 400],'Type','integer')
    optimizableVariable('MaxNumSplits',[5 150],'Type','integer')
    optimizableVariable('MinLeafSize',[1 30],'Type','integer')
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

 
%% STEP5a-Group-Aware CV Evaluation

%Take best hyperparameters
%bestParams_ens = resultsBO.XAtMinObjective;
bestParams_ens.NumLearningCycles=176;
bestParams_ens.MaxNumSplits=147;
bestParams_ens.MinLeafSize=1;

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
% Step 5d —Compute custom AlarmF1 loss
loss_ens = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (Ens): ', num2str(loss_ens)]);
% Compare models manually based on F1, NOT MATLAB’s error.
% LOWEST alarmF1Loss means highest F1

%Step7-Train FINAL model (FULL DATA)

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
save finalModel_ensb
finalModel=finalModel_ensb;
%load finalModel_ensb

%% Step4b-Train Ensemble Boosted (Global Bayesian optimization + Group-Aware CV)

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

 
%% STEP5b-Group-Aware CV Evaluation

%Take best hyperparameters
%bestParams_ens = resultsBO.XAtMinObjective;
bestParams_ens.NumLearningCycles=219;
bestParams_ens.LearnRate=0.29124;
bestParams_ens.MaxNumSplits=50;
bestParams_ens.MinLeafSize=16;
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
% Step 5d —Compute custom AlarmF1 loss
loss_ens = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (Ens): ', num2str(loss_ens)]);
% Compare models manually based on F1, NOT MATLAB’s error.

%Step7-Train FINAL model (FULL DATA)

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
save finalModel_ens
finalModel=finalModel_ens;
%load finalModel_ens

%% Step4c-Train Narrow NN (Bayesian optimization + Group CV)

% Define COST matrix
costMatrix = [0 4 5;
              3  0  1;
              4  1  0];

% Narrow NN hyperparameters
optimVars = [
    optimizableVariable('LayerSize',[5 15],'Type','integer')   % Narrow network
    optimizableVariable('Lambda',[1e-5 1],'Transform','log')   % Regularization
];

ObjFcn = @(params) nnObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 50, ...
    'IsObjectiveDeterministic', true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel', false);

%Objective Function
function objective = nnObjectiveFcn(params, X, Y, costMatrix, cv_inner)
rng(1,"twister"); %Fix the global random seed
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets
    rng(i,"twister"); %Fix the global random seed
    trainIdx = training(cv_inner, i);
    testIdx  = test(cv_inner, i);
    
    model = fitcnet( ...
        X(trainIdx,:), Y(trainIdx), ...
        'LayerSizes', params.LayerSize, ...
        'Lambda', params.Lambda, ...
        'Standardize', true, ...
        'Cost', costMatrix ...
    );
    
    Ypred(testIdx) = predict(model, X(testIdx,:));
end

objective = alarmF1Loss(Y, Ypred);

end
%% Step5c-Group-Aware CV Evaluation

%bestParams_nn = resultsBO.XAtMinObjective;
bestParams_nn.LayerSize=10;
bestParams_nn.Lambda=1.0044e-05;
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
    rng(i,"twister"); %Fix the global random seed
    trainIdx = training(cv_outer, i);
    testIdx  = test(cv_outer, i);
    
    model_fold = fitcnet( ...
        X(trainIdx,:), Y(trainIdx), ...
        'LayerSizes', bestParams_nn.LayerSize, ...
        'Lambda', bestParams_nn.Lambda, ...
        'Standardize', true, ...
        'Cost', costMatrix ...
    );
    
    Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end


confMat = confusionmat(Y, Ypred)

% Step 6c —Compute custom loss
loss_nn = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (NN): ', num2str(loss_nn)]);
% Compare models manually based on F1, NOT MATLAB’s error.

%Step7-Train FINAL model (FULL DATA)
rng(999,"twister");
finalModel_nn = fitcnet( ...
    X, Y, ...
    'LayerSizes', bestParams_nn.LayerSize, ...
    'Lambda', bestParams_nn.Lambda, ...
    'Standardize', true, ...
    'Cost', costMatrix ...
);
% This is our final deployed model
disp(finalModel_nn.ClassNames)
%save finalModel_nn
finalModel=finalModel_nn;
%load finalModel_nn

%% Step4d-Train Coarse KNN (Bayesian optimization + Group CV)

% Define COST matrix
costMatrix = [0 4 5;
3 0 1;
4 1 0];

% Coarse KNN hyperparameters
optimVars = [

optimizableVariable('NumNeighbors',[5 100],'Type','integer')

optimizableVariable('DistanceMetric', ...
{'euclidean','cityblock','cosine','correlation'}, ...
'Type','categorical')

optimizableVariable('DistanceWeight', ...
{'equal','inverse','squaredinverse'}, ...
'Type','categorical')

];



ObjFcn = @(params) knnObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
'MaxObjectiveEvaluations', 50, ...
'IsObjectiveDeterministic', true,...
'AcquisitionFunctionName','expected-improvement-plus', ...
'Verbose',1, ...
'UseParallel', false);

% Objective Function
function objective = knnObjectiveFcn(params, X, Y, costMatrix, cv_inner)

Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets

trainIdx = training(cv_inner, i);
testIdx = test(cv_inner, i);

model = fitcknn( ...
X(trainIdx,:), Y(trainIdx), ...
'NumNeighbors', params.NumNeighbors, ...
'Distance', char(params.DistanceMetric), ...
'DistanceWeight', char(params.DistanceWeight), ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model, X(testIdx,:));
end

objective = alarmF1Loss(Y, Ypred);

end

%% Step5d-Group-Aware CV Evaluation

%bestParams_knn = resultsBO.XAtMinObjective;
bestParams_knn.NumNeighbors=5;
bestParams_knn.DistanceMetric='cityblock';
bestParams_knn.DistanceWeight='squaredinverse';
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets

trainIdx = training(cv_outer, i);
testIdx = test(cv_outer, i);

model_fold = fitcknn( ...
X(trainIdx,:), Y(trainIdx), ...
'NumNeighbors', bestParams_knn.NumNeighbors, ...
'Distance', char(bestParams_knn.DistanceMetric), ...
'DistanceWeight', char(bestParams_knn.DistanceWeight), ...
'Standardize', true, ...
'Cost', costMatrix ...
);

Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end

confMat = confusionmat(Y, Ypred)

% Step6d-Compute custom loss
loss_knn = alarmF1Loss(Y, Ypred);

disp(['Cross-validated Alarm F1 Loss (KNN): ', num2str(loss_knn)]);

% Step7d-Train FINAL model (FULL DATA)
rng(999,"twister");
finalModel_knn = fitcknn( ...
X, Y, ...
'NumNeighbors', bestParams_knn.NumNeighbors, ...
'Distance', char(bestParams_knn.DistanceMetric), ...
'DistanceWeight', char(bestParams_knn.DistanceWeight), ...
'Standardize', true, ...
'Cost', costMatrix ...
);

disp(finalModel_knn.ClassNames)

%save finalModel_knn

finalModel = finalModel_knn;

%% Step4e-Train Medium Neural Network (Bayesian optimization + Group CV)

% Define COST matrix
costMatrix = [0 4 5;
              3  0  1;
              4  1  0];

% Medium NN hyperparameters
optimVars = [
    optimizableVariable('LayerSize',[15 50],'Type','integer')   % Medium network
    optimizableVariable('Lambda',[1e-5 1],'Transform','log')   % Regularization
];

ObjFcn = @(params) mnnObjectiveFcn(params, X, Y, costMatrix, cv_inner);
rng(10,"twister");
resultsBO = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 50, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'Verbose',1, ...
    'UseParallel', false);

%Objective Function
function objective = mnnObjectiveFcn(params, X, Y, costMatrix, cv_inner)
rng(1,"twister");
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_inner.NumTestSets
    rng(i,"twister");
    trainIdx = training(cv_inner, i);
    testIdx  = test(cv_inner, i);
    
    model = fitcnet( ...
        X(trainIdx,:), Y(trainIdx), ...
        'LayerSizes', params.LayerSize, ...
        'Lambda', params.Lambda, ...
        'Standardize', true, ...
        'Cost', costMatrix ...
    );
    
    Ypred(testIdx) = predict(model, X(testIdx,:));
end

objective = alarmF1Loss(Y, Ypred);

end

%% Step5e-Group-Aware CV Evaluation

%bestParams_mnn = resultsBO.XAtMinObjective;
bestParams_mnn.LayerSize=45;
bestParams_mnn.Lambda=1.1213e-05;
Ypred = Y;
Ypred(:) = missing;

for i = 1:cv_outer.NumTestSets
    rng(i,"twister");
    trainIdx = training(cv_outer, i);
    testIdx  = test(cv_outer, i);
    
    model_fold = fitcnet( ...
        X(trainIdx,:), Y(trainIdx), ...
        'LayerSizes', bestParams_mnn.LayerSize, ...
        'Lambda', bestParams_mnn.Lambda, ...
        'Standardize', true, ...
        'Cost', costMatrix ...
    );
    
    Ypred(testIdx) = predict(model_fold, X(testIdx,:));
end


confMat = confusionmat(Y, Ypred)

% Step 6e —Compute custom loss
loss_mnn = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (MNN): ', num2str(loss_mnn)]);
% Compare models manually based on F1, NOT MATLAB’s error.

%Step7e-Train FINAL model (FULL DATA)
rng(999,"twister");
finalModel_mnn = fitcnet( ...
    X, Y, ...
    'LayerSizes', bestParams_mnn.LayerSize, ...
    'Lambda', bestParams_mnn.Lambda, ...
    'Standardize', true, ...
    'Cost', costMatrix ...
);
% This is our final deployed model
disp(finalModel_mnn.ClassNames)
save finalModel_mnn
finalModel=finalModel_mnn;
%load finalModel_mnn

%% Validation & Test evaluation 
% (IF NO THRESHOLD TUNING and NO ENGINE-LEVEL DECISION, THEN APPLY THIS)

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

%% Step8a-THRESHOLD TUNING for Alarm Class (VALIDATION)
classOrder = ["alarm","warning","normal"];
Xval = ValidationData1sfnr(:, predictorNames);
Yval = ValidationData1sfnr.RUL;
%Force category order
Yval = categorical(lower(string(Yval)), classOrder);

% Apply Threshold Tuning (Find best threshold for Alarm class
% with Recall ≥ 0.95 and Precision ≥ 0.7 constraint)
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
    
    if recall >= 0.95 && precision >= 0.7 && F1 > bestF1
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

%% Step8b-THRESHOLD TUNING FOR WARNING CLASS (VALIDATION)

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
thresholdsW = quantile(score_warning, 0.3:0.001:0.95);

bestF1 = 0;
bestTW = 0;

results_thresh = [];

for tW = thresholdsW
    
    Ypred = categorical(repmat("normal", size(Yval)), classOrder);

    % 1. Alarm (FIXED)
    idxA = score_alarm >= tAlarm_fixed;
    Ypred(idxA) = "alarm";

    % 2. Warning (outside the alarm region)
    idxW = (~idxA) & (score_warning >= tW);
    Ypred(idxW) = "warning";

    % Metrics
    [recall, precision, F1] = alarmMetrics1(Yval, Ypred);

    results_thresh = [results_thresh; tW recall precision F1];

    if recall >= 0.7 && precision >= 0.7 && F1 > bestF1
        bestF1 = F1;
        bestTW = tW;
    end
end

disp(array2table(results_thresh, ...
    'VariableNames', {'tWarning','Recall','Precision','F1'}));

disp(['Fixed tAlarm: ', num2str(tAlarm_fixed)]);
disp(['Best tWarning: ', num2str(bestTW)]);
disp(['Best Validation F1: ', num2str(bestF1)]);

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

confusionchart(Yval, Ypred);
%Validation Confusion Matrix is appeared.
Xp_val=tabulate(Ypred)

%% STEP 9 — APPLY THRESHOLD ON TEST WITH WARNING CLASS 
tAlarm_fixed=0.28515;
bestTW=7.2331e-19;
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

confusionchart(Ytest, Ypred_test);
%Test Confusion Matrix is appeared.
Xp_test=tabulate(Ypred_test)
%% ABLATION ANALYSIS (IF NO THRESHOLD TUNING THEN APPLY THIS)
Ytest = TestData1sfnr.RUL;
Ytest = categorical(lower(string(Ytest)), classOrder);

Xtest = TestData1sfnr(:, predictorNames);

[label_test, score_test] = predict(finalModel, Xtest);

% Default classifier predictions
Ypred_test = categorical(label_test, classOrder);

Xp_test = tabulate(Ypred_test);

confusionchart(Ytest, Ypred_test);

AlarmF1Loss_test = alarmF1Loss(Ytest, Ypred_test);
%% ENGINE-LEVEL ANALYSIS for ABLATION ANALYSIS (IF NO THRESHOLD TUNING APPLY THIS)
% Native classifier argmax decision instead of threshold tuning decision

engineIDs = unique(TestData1sfnr.Engine_ID);

engine_pred = strings(length(engineIDs),1);
engine_true = strings(length(engineIDs),1);

for i = 1:length(engineIDs)

    idx = TestData1sfnr.Engine_ID == engineIDs(i);

    % True label from last cycle
    true_label = Ytest(find(idx,1,'last'));
    engine_true(i) = string(true_label);

    % Worst-case engine prediction
    if any(label_test(idx) == "alarm")

        engine_pred(i) = "alarm";

    elseif any(label_test(idx) == "warning")

        engine_pred(i) = "warning";

    else

        engine_pred(i) = "normal";

    end
end

engine_true = categorical(engine_true, classOrder);
engine_pred = categorical(engine_pred, classOrder);

[recall_e, precision_e, F1_e] = ...
    alarmMetric2(engine_true, engine_pred);

confusionchart(engine_true, engine_pred);

tabulate(engine_pred)

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

%% STEP 10 — ENGINE LEVEL DECISION STAGE

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
%% STEP11-Finding Predicted Alarm Engines
Engines_Compare=table(engineIDs,engine_true,engine_pred);
Engines_Alarm=Engines_Compare(engine_pred=='alarm',:)
%Predicted alarm engines' numbers are appeared without prioritization.

%% Step13-Evaluate Test Results
mukayese=table(TestData1sfnr.Engine_ID,TestData1sfnr.Time,Ytest,Ypred_test,TestData1sfnr.TTF);
mukayese.Properties.VariableNames=[{'Engine_ID'} {'Time'} {'True_Class'} {'Predicted_Class'} {'True_RUL'}];
mukayese1=mukayese(mukayese.True_Class=='alarm',:);

%Predicting RULs of the alarm classes
%Find the predicted alarms rows.
muk1_alarm=mukayese(mukayese.Predicted_Class=='alarm',:);
filtered_Alarms = muk1_alarm(ismember(muk1_alarm.Engine_ID, Engines_Alarm.engineIDs), :);

RULr_alarm=zeros(length(filtered_Alarms.Engine_ID),1);
for i=1:(length(filtered_Alarms.Engine_ID)-1) 
    RULr_alarm(1,:)=12;
    if filtered_Alarms.Engine_ID(i+1,:)==filtered_Alarms.Engine_ID(i,:)
       RULr_alarm(i+1,:)=RULr_alarm(i,:)-1;
    else 
       RULr_alarm(i+1,:)=12;
    end
end
filtered_Alarms.Predicted_RUL=RULr_alarm;
filtered_Alarms.Predicted_FailureTime=filtered_Alarms.Time+filtered_Alarms.Predicted_RUL;
% Evaluate and Prioritize Alarm Predictions
predr_alarm=filtered_Alarms(:,[1 2 6 7 5]);
n=1;
for i=1:(length(filtered_Alarms.Engine_ID)-1)
    if predr_alarm.Engine_ID(i,:)~=predr_alarm.Engine_ID(i+1,:)
        predr_alarm1(n,:)=predr_alarm(i,:);
        n=n+1;
    elseif predr_alarm.Engine_ID(i,:)==predr_alarm.Engine_ID(i+1,:)
        while predr_alarm.Engine_ID==predr_alarm.Engine_ID(i,:)
            idx=(min(predr_alarm.Predicted_RUL));
            predr_alarm1(n,:)=predr_alarm(idx,:);
            n=n+1;
        end
    end
end
predr_alarm1=[predr_alarm1;predr_alarm(end,:)];
predr_alarm1=sortrows(predr_alarm1,"Predicted_RUL")
%At the end of this pipeline2, predicted RULs of predicted each alarm
% engines are appeared with maintenance priority.