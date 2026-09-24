%% Regression Baseline Model (Random Forest)
%Load and use the same training, validation, and test data
clc;
clear;
load TrainData1sfnr
rng(1,"twister"); %Fix the global random seed

% Prepare training inputs and output
% Input features (3–22 columns)
X_train = TrainData1sfnr{:, 3:22};
% Output (TTF → column 23)
y_train = TrainData1sfnr{:, 23}; % Numerical TTF

% Prepare validation inputs and output
% Input features (3–22 columns)
X_valid = ValidationData1sfnr{:, 3:22};
% Output (TTF → column 23)
y_valid = ValidationData1sfnr{:, 23}; % Numerical TTF

% Prepare test inputs and output
% Input features (3–22 columns)
X_test = TestData1sfnr{:, 3:22};
% Output (TTF → column 23)
y_test = TestData1sfnr{:, 23}; % Numerical TTF

alarm_threshold = 13;
warning_threshold = 38;

%% 1.Candidate low-RUL weighting schemes
% Define a pre-specified candidate set.
% Format: [AlarmWeight WarningWeight NormalWeight]

WeightSchemes = [
     1   1   1
     5   2   1
    10   2   1
    20   2   1
    35   2   1
    50   2   1
   100   2   1
   150   2   1
];

nSchemes = size(WeightSchemes,1);

%% 2.Pseudo-truncated validation protocol
%
% ValidationData1sfnr contains complete run-to-failure trajectories.
% The TTF variable represents the remaining cycles to failure.
%
% Therefore, a pseudo-truncated trajectory at a specified true RUL
% contains all observations with TTF >= the selected target RUL.
%
% Example:
% targetRUL = 30
%   observed prefix: TTF = 168, 167, ..., 31, 30
%   hidden future  : TTF = 29, 28, ..., 1
%
% The independent test set is NOT used in this procedure.

pseudoRUL = [40 30 15 13 10 5];
validEngineIDs = unique(ValidationData1sfnr.Engine_ID);

nValidEngines = numel(validEngineIDs);
nCutoffs      = numel(pseudoRUL);

%% 3.Check TTF range of every validation engine

fprintf('\nValidation engine TTF ranges:\n');

for i = 1:nValidEngines

    engineID = validEngineIDs(i);

    idx = ValidationData1sfnr.Engine_ID == engineID;

    ttf_engine = ValidationData1sfnr.TTF(idx);

    fprintf('Engine %d: max TTF = %d, min TTF = %d, observations = %d\n', ...
        engineID, ...
        max(ttf_engine), ...
        min(ttf_engine), ...
        numel(ttf_engine));

end

%% 4.Construct pseudo-truncated validation cases

PseudoCases = struct();

caseCounter = 0;

for i = 1:nValidEngines

    engineID = validEngineIDs(i);

    % Rows belonging to current validation engine
    idxEngine = ValidationData1sfnr.Engine_ID == engineID;

    engineRows = find(idxEngine);

    % TTF values for this engine
    engineTTF = ValidationData1sfnr.TTF(engineRows);

    for j = 1:nCutoffs

        targetRUL = pseudoRUL(j);

        % Select the observed prefix.
        % Since TTF decreases toward failure, observations with
        % TTF >= targetRUL occur before or at the artificial cutoff.
        prefixRows = engineRows(engineTTF >= targetRUL);

        if isempty(prefixRows)
            continue;
        end

        caseCounter = caseCounter + 1;

        PseudoCases(caseCounter).EngineID = engineID;
        PseudoCases(caseCounter).TargetRUL = targetRUL;

        % The last observed row represents the artificial cutoff.
        PseudoCases(caseCounter).CutoffRow = prefixRows(end);

        % Store all observations available before/at cutoff.
        PseudoCases(caseCounter).Rows = prefixRows;

        % Ground-truth alarm status at the artificial cutoff
        PseudoCases(caseCounter).TrueAlarm = ...
            targetRUL < alarm_threshold;

    end
end

nPseudoCases = caseCounter;

fprintf('\nNumber of pseudo-truncated validation cases: %d\n', ...
    nPseudoCases);


%  Verify PseudoCases

if ~exist("PseudoCases","var")
    error("PseudoCases does not exist. Run the pseudo-truncation construction code first.");
end

nPseudoCases = numel(PseudoCases);

fprintf("\n=============================================\n");
fprintf("Pseudo-truncated validation cases: %d\n",nPseudoCases);
fprintf("=============================================\n\n");

% Verify that all three horizons are present

for h = 1:nCutoffs

    targetRUL = pseudoRUL(h);

    countHorizon = sum([PseudoCases.TargetRUL] == targetRUL);

    fprintf("Target RUL = %3d : %d pseudo-cases\n", ...
        targetRUL,countHorizon);

end

% Preallocate numerical result arrays

EngineRecall    = zeros(nSchemes,1);
EnginePrecision = zeros(nSchemes,1);
EngineF1        = zeros(nSchemes,1);

TP_all = zeros(nSchemes,1);
FP_all = zeros(nSchemes,1);
FN_all = zeros(nSchemes,1);
TN_all = zeros(nSchemes,1);

FalseAlarm_40 = zeros(nSchemes,1);
FalseAlarm_30 = zeros(nSchemes,1);
FalseAlarm_15 = zeros(nSchemes,1);
FalseAlarm_13 = zeros(nSchemes,1);

DetectedAlarm_10 = zeros(nSchemes,1);
DetectedAlarm_5 = zeros(nSchemes,1);
%% 5.EVALUATE WEIGHTING SCHEMES 

% Evaluation is made using pseudo-truncated validation engine-level
% alarm F1-score

for s = 1:nSchemes

   % Correct interpretation= Alarm : Warning : Normal
   
    wAlarm   = WeightSchemes(s,1);
    wWarning = WeightSchemes(s,2);
    wNormal  = WeightSchemes(s,3);

    fprintf("\n=============================================\n");
    fprintf("Weighting scheme: %g-%g-%g\n", ...
        wAlarm,wWarning,wNormal);
    fprintf("=============================================\n");

    fprintf("Alarm weight   : %g\n",wAlarm);
    fprintf("Warning weight : %g\n",wWarning);
    fprintf("Normal weight  : %g\n",wNormal);
   
    % Construct observation-level training weights
    weights = ones(size(y_train));

    % Normal region
    weights(y_train >= warning_threshold) = wNormal;

    % Warning region
    weights(y_train >= alarm_threshold & ...
            y_train < warning_threshold) = wWarning;

    % Alarm region
    weights(y_train < alarm_threshold) = wAlarm;

   
    % Optional diagnostic: verify assigned weights
    fprintf("\nTraining-weight distribution:\n");

    fprintf("Normal observations : %d, weight = %g\n", ...
        sum(y_train >= warning_threshold),wNormal);

    fprintf("Warning observations: %d, weight = %g\n", ...
        sum(y_train >= alarm_threshold & ...
            y_train < warning_threshold),wWarning);

    fprintf("Alarm observations  : %d, weight = %g\n", ...
        sum(y_train < alarm_threshold),wAlarm);

   
    % Train weighted Random Forest regression model
    rng(42,"twister");

    RF_Model = fitrensemble( ...
        X_train, ...
        y_train, ...
        "Method","Bag", ...
        "Weights",weights);

   
    % Predict all validation observations
    y_pred_valid = predict(RF_Model,X_valid);

 
    % Initialize pooled confusion matrix
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;

    FA40 = 0;
    FA30 = 0;
    FA15 = 0;
    FA13 = 0;
    Alarm10 = 0;
    Alarm5 = 0;

   
    % Evaluate all pseudo-truncated cases
   for c = 1:nPseudoCases

        targetRUL = PseudoCases(c).TargetRUL;
        rows      = PseudoCases(c).Rows;

        % True engine-level alarm state

        trueAlarm = targetRUL < alarm_threshold;

        % Worst-case engine alarm aggregation
        %
        % If ANY observed prefix cycle is predicted as
        % RUL < 13, the engine is classified as alarm.
        % -----------------------------------------------------

        predictedAlarm = any( ...
            y_pred_valid(rows) < alarm_threshold);

        % Update confusion matrix

        if trueAlarm && predictedAlarm

            TP = TP + 1;

        elseif ~trueAlarm && predictedAlarm

            FP = FP + 1;

        elseif trueAlarm && ~predictedAlarm

            FN = FN + 1;

        else

            TN = TN + 1;

        end

        % Horizon-specific diagnostics

       if targetRUL == 40

            if predictedAlarm
                FA40 = FA40 + 1;
            end   

        elseif targetRUL == 30

            if predictedAlarm
                FA30 = FA30 + 1;
            end

         elseif targetRUL == 15

            if predictedAlarm
                FA15 = FA15 + 1;
            end

          elseif targetRUL == 13

            if predictedAlarm
                FA13 = FA13 + 1;
            end

        elseif targetRUL == 10

            if predictedAlarm
                Alarm10 = Alarm10 + 1;
            end

        elseif targetRUL == 5

            if predictedAlarm
                Alarm5 = Alarm5 + 1;
            end

        end

    end

    % --------------------------------------------------------
    % Calculate engine-level metrics
    % ---------------------------------------------------------

    if TP + FN > 0
        recall = TP / (TP + FN);
    else
        recall = NaN;
    end

    if TP + FP > 0
        precision = TP / (TP + FP);
    else
        precision = NaN;
    end

    if ~isnan(precision) && ~isnan(recall) && ...
            (precision + recall) > 0

        F1 = 2 * precision * recall / ...
            (precision + recall);

    else

        F1 = NaN;

    end

    % --------------------------------------------------------
    % Store results
    % ---------------------------------------------------------

    EngineRecall(s)    = recall;
    EnginePrecision(s) = precision;
    EngineF1(s)        = F1;

    TP_all(s) = TP;
    FP_all(s) = FP;
    FN_all(s) = FN;
    TN_all(s) = TN;

    FalseAlarm_40(s) = FA40;
    FalseAlarm_30(s) = FA30;
    FalseAlarm_15(s) = FA15;
    FalseAlarm_13(s) = FA13;

    DetectedAlarm_10(s) = Alarm10;
    DetectedAlarm_5(s) = Alarm5;

    % --------------------------------------------------------
    % Display results
    % ---------------------------------------------------------

    fprintf("\nTP                : %d\n",TP);
    fprintf("FP                : %d\n",FP);
    fprintf("FN                : %d\n",FN);
    fprintf("TN                : %d\n",TN);

    fprintf("\nEngine Recall     : %.4f\n",recall);
    fprintf("Engine Precision  : %.4f\n",precision);
    fprintf("Engine F1         : %.4f\n",F1);

    fprintf("\nTarget RUL = 40:\n");
    fprintf("  False alarms    : %d / 20\n",FA40);

    fprintf("Target RUL = 30:\n");
    fprintf("  False alarms    : %d / 20\n",FA30);

    fprintf("Target RUL = 15:\n");
    fprintf("  False alarms    : %d / 20\n",FA15);

    fprintf("Target RUL = 13:\n");
    fprintf("  False alarms    : %d / 20\n",FA13);

    fprintf("Target RUL = 10:\n");
    fprintf("  Detected alarms : %d / 20\n",Alarm10);

    fprintf("Target RUL = 5:\n");
    fprintf("  Detected alarms : %d / 20\n",Alarm5);

end

%% 6.Construct final results table and Select best weighting scheme

SchemeLabel = strings(nSchemes,1);

for s = 1:nSchemes

    SchemeLabel(s) = sprintf("%g-%g-%g", ...
        WeightSchemes(s,1), ...
        WeightSchemes(s,2), ...
        WeightSchemes(s,3));

end

Results = table( ...
    SchemeLabel, ...
    WeightSchemes(:,1), ...
    WeightSchemes(:,2), ...
    WeightSchemes(:,3), ...
    TP_all, ...
    FP_all, ...
    FN_all, ...
    TN_all, ...
    EngineRecall, ...
    EnginePrecision, ...
    EngineF1, ...
    FalseAlarm_40, ...
    FalseAlarm_30, ...
    FalseAlarm_15, ...
    FalseAlarm_13, ...
    DetectedAlarm_10, ...
    DetectedAlarm_5, ...
    'VariableNames',{ ...
    'WeightScheme', ...
    'AlarmWeight', ...
    'WarningWeight', ...
    'NormalWeight', ...
    'TP', ...
    'FP', ...
    'FN', ...
    'TN', ...
    'EngineRecall', ...
    'EnginePrecision', ...
    'EngineF1', ...
    'FalseAlarm_RUL40', ...
    'FalseAlarm_RUL30', ...
    'FalseAlarm_RUL15', ...
    'FalseAlarm_RUL13', ...
    'DetectedAlarm_RUL10'...
    'DetectedAlarm_RUL5'});

% ------------------------------------------------------------
% Sort by validation engine F1
% ------------------------------------------------------------

Results = sortrows(Results,'EngineF1','descend');

fprintf("\n\n=============================================\n");
fprintf("FINAL VALIDATION WEIGHTING-SCHEME RESULTS\n");
fprintf("=============================================\n\n");

disp(Results);

% ============================================================
% Select best weighting scheme
% ============================================================

validF1 = Results.EngineF1;
validF1(isnan(validF1)) = -Inf;

[~,bestIdx] = max(validF1);

bestScheme = Results(bestIdx,:);

fprintf("\n=============================================\n");
fprintf("SELECTED WEIGHTING SCHEME\n");
fprintf("=============================================\n");

fprintf("Alarm weight   : %g\n",bestScheme.AlarmWeight);
fprintf("Warning weight : %g\n",bestScheme.WarningWeight);
fprintf("Normal weight  : %g\n",bestScheme.NormalWeight);

fprintf("\nValidation Engine Recall    : %.4f\n", ...
    bestScheme.EngineRecall);

fprintf("Validation Engine Precision : %.4f\n", ...
    bestScheme.EnginePrecision);

fprintf("Validation Engine F1        : %.4f\n", ...
    bestScheme.EngineF1);

fprintf("\nSelection was performed using pseudo-truncated\n");
fprintf("VALIDATION data only.\n");

%% Train Random Forest Regression
rng(42); % reproducibility

% Put cost weights to warning and alarm classes
weights = ones(size(y_train));
weights(y_train < 38) = 2;
weights(y_train < 13) = 5; %Best Weighting after pseudo-truncated validation

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

