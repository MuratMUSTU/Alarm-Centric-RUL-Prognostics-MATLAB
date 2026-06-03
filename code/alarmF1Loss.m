% Step5-Define custom F1 evaluation function
function loss = alarmF1Loss(Ytrue, Ypred)

    confMat = confusionmat(Ytrue, Ypred);
    classNames = categories(Ytrue);
    
    alarm_idx = find(classNames == "alarm");

    TP = confMat(alarm_idx, alarm_idx);
    FP = sum(confMat(:, alarm_idx)) - TP;
    FN = sum(confMat(alarm_idx, :)) - TP;

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
% Compute Alarm F1 manually
    F1 = 2 * (precision * recall) / (precision + recall + eps);

    loss = 1 - F1;  % minimize loss
end