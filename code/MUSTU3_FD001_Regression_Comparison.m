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
nTrees = 100; % enough for baseline

RF_Model = TreeBagger(nTrees, X_train, y_train, ...
    'Method', 'regression', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', 5);

%% Prediction (Test)

y_pred_test = predict(RF_Model, X_test);

% %Regression Metrics

RMSE_test = sqrt(mean((y_test - y_pred_test).^2));
MAE_test  = mean(abs(y_test - y_pred_test));

fprintf('Test RMSE: %.3f\n', RMSE_test);
fprintf('Test MAE : %.3f\n', MAE_test);

% Convert Regression → Alarm Class

% Alarm threshold (same as classification study)
alarm_threshold = 13;

y_true_alarm = y_test < alarm_threshold;
y_pred_alarm = y_pred_test < alarm_threshold;


%% Classification Metrics (ALARM_Cycle-Level)

TP = sum((y_pred_alarm == 1) & (y_true_alarm == 1));
FP = sum((y_pred_alarm == 1) & (y_true_alarm == 0));
FN = sum((y_pred_alarm == 0) & (y_true_alarm == 1));

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
F1        = 2 * (precision * recall) / (precision + recall + eps);

fprintf('Alarm Precision: %.3f\n', precision);
fprintf('Alarm Recall   : %.3f\n', recall);
fprintf('Alarm F1-score : %.3f\n', F1);

%% ENGINE-LEVEL ALARM EVALUATION
% Worst-case trajectory aggregation
%
% An engine is classified as ALARM if at least one cycle is
% predicted as ALARM (predicted RUL < 13).
%
% The true engine condition is determined from the last cycle
% of each engine, consistent with the C-MAPSS test-set labeling.

engineIDs = unique(TestData1sfnr.Engine_ID);

nEngines = length(engineIDs);

engine_true_alarm = false(nEngines,1);
engine_pred_alarm = false(nEngines,1);

for i = 1:nEngines

    % Current engine
    idx = TestData1sfnr.Engine_ID == engineIDs(i);

    % True engine-level alarm status
    % True label is determined from the last cycle
    last_idx = find(idx,1,'last');

    engine_true_alarm(i) = ...
        y_test(last_idx) < alarm_threshold;

    % Predicted engine-level alarm status
    % Worst-case trajectory aggregation:
    % if ANY cycle is predicted as alarm, engine = alarm
    engine_pred_alarm(i) = ...
        any(y_pred_test(idx) < alarm_threshold);

end

% Engine-level confusion counts

TP_engine = sum(engine_pred_alarm & engine_true_alarm);

FP_engine = sum(engine_pred_alarm & ~engine_true_alarm);

FN_engine = sum(~engine_pred_alarm & engine_true_alarm);

TN_engine = sum(~engine_pred_alarm & ~engine_true_alarm);

% Engine-level alarm metrics

precision_engine = TP_engine / ...
    (TP_engine + FP_engine + eps);

recall_engine = TP_engine / ...
    (TP_engine + FN_engine + eps);

F1_engine = 2 * ...
    (precision_engine * recall_engine) / ...
    (precision_engine + recall_engine + eps);

% Display results

fprintf('\n========================================\n');
fprintf('ENGINE-LEVEL ALARM PERFORMANCE\n');
fprintf('Worst-case trajectory aggregation\n');
fprintf('========================================\n');

fprintf('Number of engines       : %d\n', nEngines);
fprintf('True alarm engines      : %d\n', sum(engine_true_alarm));
fprintf('Predicted alarm engines : %d\n', sum(engine_pred_alarm));

fprintf('TP                      : %d\n', TP_engine);
fprintf('FP                      : %d\n', FP_engine);
fprintf('FN                      : %d\n', FN_engine);
fprintf('TN                      : %d\n', TN_engine);

fprintf('Engine Alarm Recall     : %.4f\n', recall_engine);
fprintf('Engine Alarm Precision  : %.4f\n', precision_engine);
fprintf('Engine Alarm F1-score   : %.4f\n', F1_engine);

%% ENGINE-LEVEL ALARM ENGINE IDs

True_Alarm_Engine_IDs = engineIDs(engine_true_alarm);

Pred_Alarm_Engine_IDs = engineIDs(engine_pred_alarm);

Correct_Alarm_Engine_IDs = ...
    engineIDs(engine_true_alarm & engine_pred_alarm);

Missed_Alarm_Engine_IDs = ...
    engineIDs(engine_true_alarm & ~engine_pred_alarm);

False_Alarm_Engine_IDs = ...
    engineIDs(~engine_true_alarm & engine_pred_alarm);

fprintf('\nTrue alarm engines:\n');
disp(True_Alarm_Engine_IDs');

fprintf('Predicted alarm engines:\n');
disp(Pred_Alarm_Engine_IDs');

fprintf('Correctly detected alarm engines:\n');
disp(Correct_Alarm_Engine_IDs');

fprintf('Missed alarm engines:\n');
disp(Missed_Alarm_Engine_IDs');

fprintf('False alarm engines:\n');
disp(False_Alarm_Engine_IDs');


