%% DATA LOADING (Load Raw Data to MATLAB's Workspace)
%The 260 different engine is operating normally at the beginning,
%and develops a fault (HPC degredation) during at some point the series.
%In training set, the fault grows in magnitude until failure. In test set,
% having truncated trajectories we aim to predict RUL
clc;
clear;
load engine_data2.mat
% read in only one file data, assuming here that data file can fit in the memory. 
engine_testdata2=readtable('C:\Users\asusssd\Documents\MATLAB\PdM\Data_All_Eng\test_FD002.csv');
%To add variablenames to training data;
engine_data2.Properties.VariableNames = ["Engine_ID","Time","Op_Set1","Op_Set2","Op_Set3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PcNfR_dmd","W31","W32"];
TrainData2=engine_data2;
%To add variablenames to test data;
engine_testdata2.Properties.VariableNames = ["Engine_ID","Time","Op_Set1","Op_Set2","Op_Set3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PcNfR_dmd","W31","W32"];
TestData2=engine_testdata2;
%% DATA ANALYSIS
%Checkpoint1
head(TrainData2,3) %To see the first 3 rows of training data
summary(TrainData2);
Checkpoint1=grpstats(TrainData2, "Engine_ID", "numel");
%Checkpoint results indicate that no data corruption, missing values, 
%wrong dataset split nor sensor inconsistency
%Checkpoint2
Checkpoint2=grpstats(TrainData2, "Engine_ID", @(x) numel(unique(x)), ...
    'DataVars', {'Op_Set1','Op_Set2','Op_Set3'});
% Each engine operates under MANY different conditions during its life.
% So, apply regime-aware smoothing and feature engineering
% Visualize the number of flights per engine with a histogram to make pre-evaluation
figure
histogram(TrainData2.Engine_ID,'BinMethod','integers')
xlabel('Engine No')
ylabel('Number of Flights (Cycles)')
title('Number of Flights of Engines to Failure')
Cycle_min=min(Checkpoint2.GroupCount);
Cycle_max=max(Checkpoint2.GroupCount);
disp(['Lowest Cycle: ', num2str(Cycle_min)]);
disp(['Highest Cycle: ', num2str(Cycle_max)]);
% The lowest cycle is 128 of Engine Nu.244, and the highest cycle is 378 of Engine Nu.112

%View Subset of sensor signals of the first engine, determine useful data
ID1=TrainData2(TrainData2.Engine_ID==1,:);
% We have a total 21 sensors for each engine. Since that's a lot to put
% their plots on the screen at once, lets look at the first 9 sensors and last 12 sensors seperately. 
figure
for i=1:9
    subplot(3,3,i)
    plot(ID1.Time, ID1{:,5+i})
    title(ID1.Properties.VariableNames{5+i})
    xlabel('Time')
end
%We can see that the signals of the sensors don't show any degradation
%trend, so we can not understand how the condition of the system is
%changing. The reason for that is the existence of different operation conditions.
figure
for i=1:12
    subplot(4,3,i)
    plot(ID1.Time, ID1{:,14+i})
    title(ID1.Properties.VariableNames{14+i})
    xlabel('Time')
end

%% RUL CLASS DEFINITION AND GENERATION
%Revise training table data including TTF(RUL) variables.
IDs = TrainData2{:,1};
n = length(unique(IDs));
for i=1:n
    TTF(TrainData2.Engine_ID==i)=max(TrainData2.Time(TrainData2.Engine_ID==i))-TrainData2.Time(TrainData2.Engine_ID==i);
end
TrainData2.TTF=TTF';

%3.Define Classification Thresholds
%To solve this as a classification problem, we need to define what the
%classes are and where the boundaries are between them. This typically is
%something you cannot do purely from the equipment sensor data.Here we drew
%arbitrary boundaries to create three different classes. We will attempt to
%classify each point as being alarm (urgently) in need of maintenance, or having a
% warning (short) time until maintenance is needed, or normal (no maintenance need).
Thresh_alarm=round(10*Cycle_min/100);
Thresh_warning=round(30*Cycle_min/100);
catThreshold=[Thresh_alarm,Thresh_warning,Cycle_max];
%If RUL<13 Alarm, 13=<RUL<38 Warning, 38=<RUL Normal
%To convert numerical TTF data to the categorical RUL data
TTFc = discretize(TrainData2.TTF,[0 Thresh_alarm Thresh_warning Cycle_max],"categorical",["alarm" "warning" "normal"]);
TrainData2.RUL=TTFc;
%Look at spread of classes
tabulate(TrainData2.RUL);
%Alarm classes of training data are rare (%6.29). 
% If Alarm is <10% of total data, you definitely need imbalance handling.
histogram(TrainData2.RUL)

% Revise test data including TTF(RUL) variables in the same way.
engine_RULdata2=readtable('C:\Users\asusssd\Documents\MATLAB\PdM\Data_RUL_Eng\RUL_FD002.csv');
%load engine_RULdata2
IDs = TestData2{:,1};
n = length(unique(IDs));
for i=1:n
    y=max(TestData2.Time(TestData2.Engine_ID==i));
    x=y+engine_RULdata2{i,:};
    RTF(TestData2.Engine_ID==i)=x-TestData2.Time(TestData2.Engine_ID==i);
end
TestData2.TTF=RTF';
TTF_testc = discretize(RTF',[0 Thresh_alarm Thresh_warning Cycle_max],"categorical",["alarm" "warning" "normal"]);
TestData2.RUL=TTF_testc;

%% ENGINE LEVEL DATA PARTITIONING (%80 for training and %20 for validation)
%To convert table data to cell array data
IDs = TrainData2{:,1};
nID = unique(IDs);
degradationData = cell(numel(nID),1);
for ct=1:numel(nID)
    idx = IDs == nID(ct);
    degradationData{ct} = TrainData2(idx,:);
end
%Split the degradation data into a training data set and a validation data set
rng('default')  % To make sure the results are repeatable by fixing the global random seed
numEnsemble = length(degradationData);
%Partition the data (%80 for training and %20 for validation)
numFold = 5;
cv = cvpartition(numEnsemble, 'KFold', numFold);
TrainData = degradationData(training(cv, 1)); %cell array data of 80 engines
TrainData2= vertcat(TrainData{:}) %raw table data of 80 engines
ValidationData = degradationData(test(cv, 1)); %cell array data of 20 engines
ValidationData2= vertcat(ValidationData{:}) %raw table data of 20 engines
% Now, raw training, validation, and test datasets are ready to save.
save TrainData2

%% Visualize All Sensor Data in Categories
figure
for ii=1:9
h(ii)=subplot(3,3,ii);
scatter(h(ii), TrainData2.Time, TrainData2{:,5+ii},[], TrainData2.RUL, 'filled' );
title(h(ii), TrainData2.Properties.VariableNames{5+ii})
xlabel(h(ii),'Time')

end
figure
for ii=1:12
    h(ii)=subplot(3,4,ii);
    scatter(h(ii), TrainData2.Time, TrainData2{:,14+ii},[], TrainData2.RUL, 'filled' );
    title(h(ii), TrainData2.Properties.VariableNames{14+ii})
    xlabel(h(ii),'Time')
end
%The signals of the 21 sensors don't show any degradation trend due to the
%different operation conditions. But we should remove the same 7 sensor data of FD001
%% DATA PREPROCESSING (REMOVE UNUSABLE SAME DATA)
load TrainData2
%Select the useful variables of training and test data. (Feature Selection)
TrainData2=TrainData2(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);
ValidationData2=ValidationData2(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);
TestData2=TestData2(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);

%% DATA PREPROCESSING-SMOOTHING (REMOVE NOISE)

% Step1-Define operating regime segments (cluster) before regime-aware smoothing
% Operating condition variables
opVars = {'Op_Set1','Op_Set2','Op_Set3'};
Xop = TrainData2{:, opVars};

% K-means clustering (6 for FD002)
numClusters = 6;

[idx, C] = kmeans(Xop, numClusters, 'Replicates',10, 'MaxIter',1000);

% Add Cluster label to dataset
TrainData2.ClusterID = idx;

%Step 2 — Both engine-level and regime-aware smoothing WITHIN segments
engines = unique(TrainData2.Engine_ID);
% We must split continuous segments, not just cluster groups.
sensorVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'}; % 14 sensor as input data

for e = 1:length(engines)
    idxE = TrainData2.Engine_ID == engines(e);
    
    dataE = TrainData2(idxE,:);
    dataE = sortrows(dataE, 'Time');
    % find regime change points
    clusterSeq = dataE.ClusterID;
    changeIdx = [1; find(diff(clusterSeq) ~= 0) + 1; height(dataE)+1];
    
    for i = 1:length(changeIdx)-1
        segIdx = changeIdx(i):changeIdx(i+1)-1;
        
        segment = dataE(segIdx,:);
        
        % smoothing ONLY within continuous segment
        % Pad with the first value repeated 4 times at the beginning
        segment1 = [repmat(segment(1,:),4,1); segment];
        % Apply causal moving average [4,0] or a trailing moving average to smooth the signals slightly
        if height(segment) >= 5
            segment1{:, sensorVars} = movmean(segment1{:, sensorVars}, [4,0]);
        end
        % Remove padding (shift back)
        segment1(1:4,:) = []; %Delete the first 4 rows
        
        dataE(segIdx,:) = segment1;
    end
    
    TrainData2(idxE,:) = dataE;
end
TrainData2s=TrainData2

% Step3-Validation Data regime-aware smoothing
Xop_val = ValidationData2{:, opVars};
idx_val = knnsearch(C, Xop_val); % closest centroid
% Add Cluster label to dataset
ValidationData2.ClusterID = idx_val;
%Step 2 — Smooth WITHIN segments
engines = unique(ValidationData2.Engine_ID);
for e = 1:length(engines)
    idxE = ValidationData2.Engine_ID == engines(e);
    
    dataE = ValidationData2(idxE,:);
    dataE = sortrows(dataE, 'Time');
    % find regime change points
    clusterSeq = dataE.ClusterID;
    changeIdx = [1; find(diff(clusterSeq) ~= 0) + 1; height(dataE)+1];
    
    for i = 1:length(changeIdx)-1
        segIdx = changeIdx(i):changeIdx(i+1)-1;
        
        segment = dataE(segIdx,:);
        
        % smoothing ONLY within continuous segment
        % Pad with the first value repeated 4 times at the beginning
        segment1 = [repmat(segment(1,:),4,1); segment];
        % Apply causal moving average [4,0] or a trailing moving average to smooth the signals slightly
        if height(segment) >= 5
            segment1{:, sensorVars} = movmean(segment1{:, sensorVars}, [4,0]);
        end
        % Remove padding (shift back)
        segment1(1:4,:) = []; %Delete the first 4 rows
        
        dataE(segIdx,:) = segment1;
    end
    
    ValidationData2(idxE,:) = dataE;
end
ValidationData2s=ValidationData2

% Step4-Test Data regime-aware smoothing
Xop_test = TestData2{:, opVars};
idx_test = knnsearch(C, Xop_test); % closest centroid
% Add Cluster label to dataset
TestData2.ClusterID = idx_test;
%Step 2 — Smooth WITHIN segments
engines = unique(TestData2.Engine_ID);
for e = 1:length(engines)
    idxE = TestData2.Engine_ID == engines(e);
    
    dataE = TestData2(idxE,:);
    dataE = sortrows(dataE, 'Time');
    % find regime change points
    clusterSeq = dataE.ClusterID;
    changeIdx = [1; find(diff(clusterSeq) ~= 0) + 1; height(dataE)+1];
    
    for i = 1:length(changeIdx)-1
        segIdx = changeIdx(i):changeIdx(i+1)-1;
        
        segment = dataE(segIdx,:);
        
        % smoothing ONLY within continuous segment
        % Pad with the first value repeated 4 times at the beginning
        segment1 = [repmat(segment(1,:),4,1); segment];
        % Apply causal moving average [4,0] or a trailing moving average to smooth the signals slightly
        if height(segment) >= 5
            segment1{:, sensorVars} = movmean(segment1{:, sensorVars}, [4,0]);
        end
        % Remove padding (shift back)
        segment1(1:4,:) = []; %Delete the first 4 rows
        
        dataE(segIdx,:) = segment1;
    end
    
    TestData2(idxE,:) = dataE;
end
TestData2s=TestData2
save TrainData2s

ID1=TrainData2s(TrainData2s.Engine_ID==1,:);
% We have 14 sensors for each engine.
figure
for i=1:15
    subplot(5,3,i)
    plot(ID1.Time, ID1{:,5+i})
    title(ID1.Properties.VariableNames{5+i})
    xlabel('Time')
end
%% FEATURE ENGINEERING (ENGINE-LEVEL ONLY)
load TrainData2s
% For training data, add features by making feature engineering
% Long-term features are intentionally excluded to improve robustness
% under multiple operating conditions (FD002).

% 1.Add cycle-to-cycle change (sensor(t) − sensor(t−1)) of 14 sensors 
% 2.Add short (5 cycles) average moving trends (average slopes) of 14 sensors
% (sensor(t) − sensor(t−5)) / 5. This makes slopes scale-consistent.
% 3.Add mid (10 cycles) average moving trends (average slopes) of 14 sensors 
% Trend features must be calculated within each engine trajectory.

sensorVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'};

%Only engine-level feature engineering WITHIN continuous segments
engines = unique(TrainData2s.Engine_ID);

% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    TrainData2s.(["d" + sensor]) = NaN(height(TrainData2s),1);
    TrainData2s.(sensor + "_trend5") = NaN(height(TrainData2s),1);
    TrainData2s.(sensor + "_trend10") = NaN(height(TrainData2s),1);
end

% TRAINING DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = TrainData2s.Engine_ID == engines(e);
    
    dataE = TrainData2s(idxE,:);
    dataE = sortrows(dataE, 'Time');

   for s = 1:length(sensorVars)
        sensor = sensorVars{s};
        x = dataE.(sensor);
        n = length(x);
        
        % First difference
        dx = [NaN; diff(x)];   
        
        % Short-term 5-step trend
        trend5 = NaN(n,1);
        if n >= 6
            trend5(6:end) = (x(6:end) - x(1:end-5)) / 5;
        end
        % Mid-term 10-step trend
        trend10 = NaN(n,1);
        if n >= 11
            trend10(11:end) = (x(11:end) - x(1:end-10)) / 10;
        end

        %Assign
        dataE.(["d" + sensor]) = dx;
        dataE.(sensor + "_trend5") = trend5;
        dataE.(sensor + "_trend10") = trend10; 
    end
    
    % Write back preserving order
    [~, order] = sort(dataE.Time);
    TrainData2s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(TrainData2s),1);
engines = unique(TrainData2s.Engine_ID);

for e = 1:length(engines)
    idxE = find(TrainData2s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(TrainData2s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

TrainData2sf = TrainData2s(rowsToKeep,:);

% For validation data, add the same features. 
engines = unique(ValidationData2s.Engine_ID);
% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    ValidationData2s.(["d" + sensor]) = NaN(height(ValidationData2s),1);
    ValidationData2s.(sensor + "_trend5") = NaN(height(ValidationData2s),1);
    ValidationData2s.(sensor + "_trend10") = NaN(height(ValidationData2s),1);
end

% VALIDATION DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = ValidationData2s.Engine_ID == engines(e);
    
    dataE = ValidationData2s(idxE,:);
    dataE = sortrows(dataE, 'Time');
        
    for s = 1:length(sensorVars)
        sensor = sensorVars{s};
        x = dataE.(sensor);
        n = length(x);
        
         % First difference
        dx = [NaN; diff(x)];   
        
        % Short-term 5-step trend
        trend5 = NaN(n,1);
        if n >= 6
            trend5(6:end) = (x(6:end) - x(1:end-5)) / 5;
        end

        % Mid-term 10-step trend
        trend10 = NaN(n,1);
        if n >= 11
            trend10(11:end) = (x(11:end) - x(1:end-10)) / 10;
        end
        
        %Assign
        dataE.(["d" + sensor]) = dx;
        dataE.(sensor + "_trend5") = trend5;
        dataE.(sensor + "_trend10") = trend10;
    end
    
    % Write back preserving order
    [~, order] = sort(dataE.Time);
    ValidationData2s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(ValidationData2s),1);
engines = unique(ValidationData2s.Engine_ID);

for e = 1:length(engines)
    idxE = find(ValidationData2s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(ValidationData2s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

ValidationData2sf = ValidationData2s(rowsToKeep,:);

% For test data, add the same features. 
engines = unique(TestData2s.Engine_ID);

% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    TestData2s.(["d" + sensor]) = NaN(height(TestData2s),1);
    TestData2s.(sensor + "_trend5") = NaN(height(TestData2s),1);
    TestData2s.(sensor + "_trend10") = NaN(height(TestData2s),1);
end

% TEST DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = TestData2s.Engine_ID == engines(e);
    
    dataE = TestData2s(idxE,:);
    dataE = sortrows(dataE, 'Time');
       
    for s = 1:length(sensorVars)
        sensor = sensorVars{s};
        x = dataE.(sensor);
        n = length(x);
        
        % First difference
        dx = [NaN; diff(x)];   
        
        % Short-term 5-step trend
        trend5 = NaN(n,1);
        if n >= 6
            trend5(6:end) = (x(6:end) - x(1:end-5)) / 5;
        end

        % Mid-term 10-step trend
        trend10 = NaN(n,1);
        if n >= 11
            trend10(11:end) = (x(11:end) - x(1:end-10)) / 10;
        end
  
         %Assign
         dataE.(["d" + sensor]) = dx;
         dataE.(sensor + "_trend5") = trend5;
         dataE.(sensor + "_trend10") = trend10;
    end
    
    % Write back preserving order
    [~, order] = sort(dataE.Time);
    TestData2s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(TestData2s),1);
engines = unique(TestData2s.Engine_ID);

for e = 1:length(engines)
    idxE = find(TestData2s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(TestData2s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

TestData2sf = TestData2s(rowsToKeep,:);

save TrainData2sf
ID1=TrainData2sf(TrainData2sf.Engine_ID==1,:)

%% TRAINING DATA NORMALIZATION
%load TrainData2sf
% STEP 2 — Cluster-based normalization (TRAIN)
sensorVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32',...
    'dT24','dT30','dT50','dP30','dNf','dNc','dPs30','dphi','dNRf','dNRc','dBPR','dhtBleed','dW31','dW32',...
    'T24_trend5','T30_trend5','T50_trend5','P30_trend5','Nf_trend5','Nc_trend5','Ps30_trend5',...
    'phi_trend5','NRf_trend5','NRc_trend5','BPR_trend5','htBleed_trend5','W31_trend5','W32_trend5',...
    'T24_trend10','T30_trend10','T50_trend10','P30_trend10','Nf_trend10','Nc_trend10','Ps30_trend10',...
    'phi_trend10','NRf_trend10','NRc_trend10','BPR_trend10','htBleed_trend10','W31_trend10','W32_trend10'}; % 14 sensor + 42 features as input data
numClusters=6;
% Calculate mean & std for each cluster
clusterStats = struct();

for k = 1:numClusters
    idx = TrainData2sf.ClusterID == k;
    
    data_k = TrainData2sf{idx, sensorVars};
    
    mu = mean(data_k, 1);
    sigma = std(data_k, 0, 1);
    
    % zero std protection
    sigma(sigma == 0) = 1;
    
    clusterStats(k).mu = mu;
    clusterStats(k).sigma = sigma;
end

% Normalize TRAIN
Xnorm = zeros(size(TrainData2sf{:, sensorVars}));

for k = 1:numClusters
    idx = TrainData2sf.ClusterID == k;
    
    X = TrainData2sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm(idx,:) = (X - mu) ./ sigma;
end

TrainData2sf{:, sensorVars} = Xnorm;
TrainData2sfn=TrainData2sf

%% VALIDATION AND TEST DATA NORMALIZATION
% Validation normalization
Xnorm_val = zeros(size(ValidationData2sf{:, sensorVars}));

for k = 1:numClusters
    idx = ValidationData2sf.ClusterID == k;
    
    X = ValidationData2sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm_val(idx,:) = (X - mu) ./ sigma;
end

ValidationData2sf{:, sensorVars} = Xnorm_val;
ValidationData2sfn=ValidationData2sf

% Test normalization
Xnorm_test = zeros(size(TestData2sf{:, sensorVars}));

for k = 1:numClusters
    idx = TestData2sf.ClusterID == k;
    
    X = TestData2sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm_test(idx,:) = (X - mu) ./ sigma;
end

TestData2sf{:, sensorVars} = Xnorm_test;
TestData2sfn=TestData2sf
%save TrainData2sfn
ID1=TrainData2sfn(TrainData2sfn.Engine_ID==1,:);
% We have 14 sensors and 42 features for each engine.
figure
for i=1:15
    subplot(5,3,i)
    plot(ID1.Time, ID1{:,5+i})
    title(ID1.Properties.VariableNames{5+i})
    xlabel('Time')
end
figure
for i=1:15
    subplot(5,3,i)
    plot(ID1.Time, ID1{:,22+i})
    title(ID1.Properties.VariableNames{22+i})
    xlabel('Time')
end
figure
for i=1:15
    subplot(5,3,i)
    plot(ID1.Time, ID1{:,37+i})
    title(ID1.Properties.VariableNames{37+i})
    xlabel('Time')
end
figure
for i=1:12
    subplot(4,3,i)
    plot(ID1.Time, ID1{:,52+i})
    title(ID1.Properties.VariableNames{52+i})
    xlabel('Time')
end
%After normalization, basic 14 sensor signals showed trend.
%% CREATING HEALTH INDEX AS A NEW FEATURE
load TrainData2sfn

rng(1,"twister"); %Fix the global random seed
basic_vars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'}; %Basic 14 sensors
% FOR TRAINING DATA
Xtrain = TrainData2sfn{:, basic_vars};
%Apply PCA (Fit PCA ONLY on training data)
[coeff, scoreTrain, latent, tsquared, explained, mu] = pca(Xtrain);
% Use first principal component (PC1) as Health Index
HI_train = scoreTrain(:,1);
%Ensure degradation direction (HI should decrease with time)
corr_val = corr(HI_train, TrainData2sfn.Time);
if corr_val > 0
    HI_train = -HI_train;
    coeff(:,1) = -coeff(:,1); % ensure consistency for projection
end

% GLOBAL NORMALIZATION
% Use training statistics ONLY
mu_HI  = mean(HI_train);
std_HI = std(HI_train);

% Avoid division by zero
if std_HI == 0
    std_HI = 1;
end

HI_train_norm = (HI_train - mu_HI) / std_HI; %HI has absolute meaning across engines

%Add HI as new feature
TrainData2sfn.HealthIndex = HI_train_norm;

%FOR VALIDATION DATA
Xval = ValidationData2sfn{:, basic_vars};
% Project validation onto TRAIN PCA space
scoreVal = (Xval - mu) * coeff;

HI_val = scoreVal(:,1);

% Apply TRAIN normalization
HI_val_norm = (HI_val - mu_HI) / std_HI;

ValidationData2sfn.HealthIndex = HI_val_norm;

%FOR TEST DATA
Xtest = TestData2sfn{:, basic_vars};
% Project test onto TRAIN PCA space
scoreTest = (Xtest - mu) * coeff;

HI_test = scoreTest(:,1);

% Apply TRAIN normalization
HI_test_norm = (HI_test - mu_HI) / std_HI;

TestData2sfn.HealthIndex = HI_test_norm;
% We have totally 57 features consist of 14 sensor data and 43 generated data
%save TestData2sfn
%% FEATURE RANKING and SELECTION OF TOP 20 PREDICTORS
load TestData2sfn
rng(1,"twister"); %Fix the global random seed for reproducibility

% Define feature set (same as used in normalization + HI)
featureVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi',...
    'NRf','NRc','BPR','htBleed','W31','W32',...
    'dT24','dT30','dT50','dP30','dNf','dNc','dPs30','dphi','dNRf','dNRc','dBPR','dhtBleed','dW31','dW32',...
    'T24_trend5','T30_trend5','T50_trend5','P30_trend5','Nf_trend5','Nc_trend5','Ps30_trend5',...
    'phi_trend5','NRf_trend5','NRc_trend5','BPR_trend5','htBleed_trend5','W31_trend5','W32_trend5',...
    'T24_trend10','T30_trend10','T50_trend10','P30_trend10','Nf_trend10','Nc_trend10','Ps30_trend10',...
    'phi_trend10','NRf_trend10','NRc_trend10','BPR_trend10','htBleed_trend10','W31_trend10','W32_trend10',...
    'HealthIndex'};

X = TrainData2sfn{:, featureVars};

% Replace with your actual class label variable
Y = TrainData2sfn.RUL;

% Feature ranking using mRMR on training data
[idx, scores] = fscmrmr(X, Y);
rankedFeatures = featureVars(idx);

% Select top-k features
k = 20;  
topFeatures = rankedFeatures(1:k);
topScores   = scores(idx(1:k));

T = table(topFeatures', topScores', ...
    'VariableNames', {'Feature','mRMR_Score'});

disp(T)
save selectedFeatures2.mat topFeatures

%% Select Top 20 common predictors for features.
load TestData2sfn
%load selectedFeatures1
%load selectedFeatures2
% Here, the 20 common predictors of FD001 and FD002 datasetare selected.
rng(1,"twister"); %Fix the global random seed for reproducibility
%Prepare the datasets for next 2ndstage pipeline according to the selected predictors
%Mixed20= HI+12 sensor+3 mid-term +4 short-term (Replace 'BPR_trend10' with 'P30_trend10')
columns_top_pred= {'Engine_ID','Time','HealthIndex','dW32','NRf_trend5','T24_trend5','NRc_trend5','Nc','Nf','T24','T30','T50','W32','htBleed','P30','Ps30','phi','BPR','W31','NRc_trend10', 'P30_trend10','NRf_trend10','TTF','RUL'};

TrainData2sfnr=TrainData2sfn(:, columns_top_pred);
TestData2sfnr=TestData2sfn(:, columns_top_pred);
ValidationData2sfnr=ValidationData2sfn(:, columns_top_pred);
%save TrainData2sfnr
%Now, preprocessed same 20 predictors of FD002 are ready for second pipeline. 
%% SECOND PIPELINE
clc;
clear;
load TrainData2sfnr
rng(1,"twister"); %Fix the global random seed
% Step1-Extract variables
predictorNames = TrainData2sfnr.Properties.VariableNames;
predictorNames = setdiff(predictorNames, {'Engine_ID', 'Time','TTF','RUL'});

X = TrainData2sfnr(:, predictorNames);
Y = TrainData2sfnr.RUL;

classOrder = ["alarm","warning","normal"];
Y = categorical(lower(string(Y)), classOrder);

groups = TrainData2sfnr.Engine_ID; % Create group vector
TTF = TrainData2sfnr.TTF;

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
% Already defined

%% Step4-Train Narrow NN with optimized hyperparameters + Group CV

% Define the same COST matrix
costMatrix = [0 4 5;
              3  0  1;
              4  1  0];


% Group-Aware CV Evaluation

%Define the same parameters from Narrow NN Bayesian optimization of FD001;
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

% Compute custom loss
loss_nn = alarmF1Loss(Y, Ypred);
disp(['Cross-validated Alarm F1 Loss (NN): ', num2str(loss_nn)]);
% Compare models manually based on F1, NOT MATLAB’s error.

%Train FINAL model (FULL DATA)
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

%% STEP 5 — APPLY THRESHOLD ON TEST WITH WARNING CLASS 
%Define the same calibrated thresholds of Narrow NN FD001
tAlarm_fixed=0.28515;
bestTW=7.2331e-19;

classOrder = ["alarm","warning","normal"];

Xtest = TestData2sfnr(:, predictorNames);
Ytest = categorical(lower(string(TestData2sfnr.RUL)), classOrder);

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
%Cycle-level test Confusion Matrix is appeared
Xp_test=tabulate(Ypred_test)

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
%% STEP 6 — ENGINE LEVEL DECISION

engineIDs = unique(TestData2sfnr.Engine_ID);

engine_pred = strings(length(engineIDs),1);
engine_true = strings(length(engineIDs),1);

for i = 1:length(engineIDs)

    idx = TestData2sfnr.Engine_ID == engineIDs(i);
    
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
%Engine-level test Confusion Matrix is appeared
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
%% Step7 Finding Predicted Alarm Engines
Engines_Compare=table(engineIDs,engine_true,engine_pred);
Engines_Alarm=Engines_Compare(engine_pred=='alarm',:)

%% Step8-Evaluate Test Results
mukayese=table(TestData2sfnr.Engine_ID,TestData2sfnr.Time,Ytest,Ypred_test,TestData2sfnr.TTF);
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