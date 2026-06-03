%% Regression Baseline Model (Random Forest)
%Load and use the same training, and test data
clc;
clear;
load TrainData1sfnr
rng(1,"twister"); %Fix the global random seed

% Prepare training inputs and output
% Input features (3–22 columns)
X_train = TrainData1sfnr{:, 3:22};
% Output (TTF → column 23)
y_train = TrainData1sfnr{:, 23};

% Prepare test inputs and output
% Input features (3–22 columns)
X_test = TestData1sfnr{:, 3:22};
% Output (TTF → column 23)
y_test = TestData1sfnr{:, 23};


%% Train Random Forest Regression
rng(42); % reproducibility

% Put cost weights to warning and alarm classes
weights = ones(size(y_train));
weights(y_train < 38) = 2;
weights(y_train < 13) = 150;

RF_Model = fitrensemble(X_train, y_train, 'Method','Bag','Weights',weights);

%% Prediction (Test)

y_pred_test = predict(RF_Model, X_test);

% Regression Metrics

RMSE_test = sqrt(mean((y_test - y_pred_test).^2));
MAE_test  = mean(abs(y_test - y_pred_test));

fprintf('Test RMSE: %.3f\n', RMSE_test);
fprintf('Test MAE : %.3f\n', MAE_test);

% Convert Regression → Alarm Class

% Alarm threshold (same as classification study)
alarm_threshold = 13;

y_true_alarm = y_test < alarm_threshold;
y_pred_alarm = y_pred_test < alarm_threshold;

TestData1sfnr.pred_RUL= y_pred_test;
TestData_RUL=TestData1sfnr(:,[1,2,23,24,25]);
TestData_RUL=TestData_RUL(TestData_RUL.pred_RUL<13,:);
Alarm_Engines=unique(TestData_RUL.Engine_ID)
%% Classification Metrics (ALARM)

TP = sum((y_pred_alarm == 1) & (y_true_alarm == 1));
FP = sum((y_pred_alarm == 1) & (y_true_alarm == 0));
FN = sum((y_pred_alarm == 0) & (y_true_alarm == 1));

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
F1        = 2 * (precision * recall) / (precision + recall + eps);

fprintf('Alarm Precision: %.3f\n', precision);
fprintf('Alarm Recall   : %.3f\n', recall);
fprintf('Alarm F1-score : %.3f\n', F1);

