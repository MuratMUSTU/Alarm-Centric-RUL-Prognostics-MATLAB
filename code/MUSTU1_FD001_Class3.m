%% DATA LOADING (Load Raw Data to MATLAB's Workspace)
%The 100 different engine is operating normally at the beginning,
%and develops a fault (HPC degredation) during at some point the series.
%In training set, the fault grows in magnitude until failure. In test set,
% having truncated trajectories we aim to predict RUL
clc;
clear;
load engine_data1.mat %Training data of FD001 loaded
load engine_testdata1_Unhead.mat %Test data of FD001 loaded

%To add variablenames to training data;
engine_data1.Properties.VariableNames = ["Engine_ID","Time","Op_Set1","Op_Set2","Op_Set3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PcNfR_dmd","W31","W32"];
TrainData1=engine_data1;
%To add variablenames to test data;
engine_testdata1_Unhead.Properties.VariableNames = ["Engine_ID","Time","Op_Set1","Op_Set2","Op_Set3","T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi","NRf","NRc","BPR","farB","htBleed","Nf_dmd","PcNfR_dmd","W31","W32"];
TestData1=engine_testdata1_Unhead;
%% DATA ANALYSIS
head(TrainData1,3) %To see the first 3 rows of training data
summary(TrainData1);
Checkpoint1=grpstats(TrainData1, "Engine_ID", "numel");
%Checkpoint results indicate that no data corruption, missing values, 
%wrong dataset split nor sensor inconsistency

%Checkpoint2
Checkpoint2=grpstats(TrainData1, "Engine_ID", @(x) numel(unique(x)), ...
    'DataVars', {'Op_Set1','Op_Set2','Op_Set3'});
% Each engine operates under one condition during its life.
% While Op1=0 and Op2=0, Op3=100 There's only one operation condition. (Sea Level Flight)

% Visualize the number of flights per engine with a histogram to make pre-evaluation
figure
histogram(TrainData1.Engine_ID,'BinMethod','integers')
xlabel('Engine No')
ylabel('Number of Flights (Cycles)')
title('Number of Flights of Engines to Failure')
Cycle_min=min(Checkpoint2.GroupCount);
Cycle_max=max(Checkpoint2.GroupCount);
disp(['Lowest Cycle: ', num2str(Cycle_min)]);
disp(['Highest Cycle: ', num2str(Cycle_max)]);
% The lowest cycle is 128 of Engine Nu.39, and the highest cycle is 362 of Engine Nu.69

%View Subset of sensor signals of the first engine, determine useful data
ID1=TrainData1(TrainData1.Engine_ID==1,:);
% We have a total 21 sensors for each engine. Since that's a lot to put
% their plots on the screen at once, lets look at the first 9 sensors. 
figure
for i=1:9
    subplot(3,3,i)
    plot(ID1.Time, ID1{:,5+i})
    title(ID1.Properties.VariableNames{5+i})
    xlabel('Time')
end
%We can see that some of the signals are flat, so they won't be useful for
%understanding how the condition of the system is changing. Others need a
%bit of smoothing to remove some noises. Lets look at the other 12 sensors.
figure
for i=1:12
    subplot(4,3,i)
    plot(ID1.Time, ID1{:,14+i})
    title(ID1.Properties.VariableNames{14+i})
    xlabel('Time')
end
%Variable 6, 10, 11, 15, 21, 23 and 24 are not useful.
% We can select the useable 14 variables (feature selection) and remove the 
% 7 sensor data that were flat and unuseful.

%% RUL CLASS DEFINITION AND GENERATION
%Revise training table data including TTF(RUL) variables.
IDs = TrainData1{:,1};
n = length(unique(IDs));
for i=1:n
    TTF(TrainData1.Engine_ID==i)=max(TrainData1.Time(TrainData1.Engine_ID==i))-TrainData1.Time(TrainData1.Engine_ID==i);
end
TrainData1.TTF=TTF';

%Define Classification Thresholds
%To solve this as a classification problem, we need to define what the
%classes are and where the boundaries are between them. This typically is
%something you cannot do purely from the equipment sensor data.Here we drew
%arbitrary boundaries to create three different classes. We will attempt to
%classify each point as being alarm (urgently) in need of maintenance, or having a
% warning (short) time until maintenance is needed, or normal (no maintenance need).
Thresh_alarm=round(10*Cycle_min/100);

%Alarm Threshold Sensivity Analysis
%Thresh_alarm=15;

Thresh_warning=round(30*Cycle_min/100);
catThreshold=[Thresh_alarm,Thresh_warning,Cycle_max];
%If RUL<13 Alarm, 13=<RUL<38 Warning, 38=<RUL Normal
%To convert numerical TTF data to the categorical RUL data
TTFc = discretize(TrainData1.TTF,[0 Thresh_alarm Thresh_warning Cycle_max],"categorical",["alarm" "warning" "normal"]);
TrainData1.RUL=TTFc;

%Look at spread of classes
tabulate(TrainData1.RUL);
%Alarm classes of training data are rare (%6.30). 
% If Alarm is < 10% of total data, you definitely need imbalance handling.
histogram(TrainData1.RUL)

% Revise test data including TTF(RUL) variables in the same way.
load engine_RULdata1
for i=1:100
    y=max(TestData1.Time(TestData1.Engine_ID==i));
    x=y+engine_RULdata1{i,:};
    RTF(TestData1.Engine_ID==i)=x-TestData1.Time(TestData1.Engine_ID==i);
end
TestData1.TTF=RTF';
TTF_testc = discretize(RTF',[0 Thresh_alarm Thresh_warning Cycle_max],"categorical",["alarm" "warning" "normal"]);
TestData1.RUL=TTF_testc;

%% ENGINE LEVEL DATA PARTITIONING (%80 for training and %20 for validation)
%To convert table data to cell array data
IDs = TrainData1{:,1};
nID = unique(IDs);
degradationData = cell(numel(nID),1);
for ct=1:numel(nID)
    idx = IDs == nID(ct);
    degradationData{ct} = TrainData1(idx,:);
end
%Split the degradation data into a training data set and a validation data set
rng('default')  % To make sure the results are repeatable by fixing the global random seed
numEnsemble = length(degradationData);
%Partition the data (%80 for training and %20 for validation)
numFold = 5;
cv = cvpartition(numEnsemble, 'KFold', numFold);
TrainData = degradationData(training(cv, 1)); %cell array data of 80 engines
TrainData1= vertcat(TrainData{:}) %raw table data of 80 engines
ValidationData = degradationData(test(cv, 1)); %cell array data of 20 engines
ValidationData1= vertcat(ValidationData{:}) %raw table data of 20 engines
% Now, raw training, validation, and test datasets are ready to save.
%save TrainData1

%% Visualize All Sensor Data in Categories
figure
for ii=1:9
h(ii)=subplot(3,3,ii);
scatter(h(ii), TrainData1.Time, TrainData1{:,5+ii},[], TrainData1.RUL, 'filled' );
title(h(ii), TrainData1.Properties.VariableNames{5+ii})
xlabel(h(ii),'Time')
end

figure
for ii=1:12
    h(ii)=subplot(3,4,ii);
    scatter(h(ii), TrainData1.Time, TrainData1{:,14+ii},[], TrainData1.RUL, 'filled' );
    title(h(ii), TrainData1.Properties.VariableNames{14+ii})
    xlabel(h(ii),'Time')
end
%The signals of the 7 sensors should be removed because of having no
%degradation trend.
%% DATA PREPROCESSING (REMOVE UNUSABLE DATA)
%load TrainData1
%Select the useful variables of training and test data. (Feature Selection)
TrainData1_raw=TrainData1(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);
ValidationData1_raw=ValidationData1(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);
TestData1_raw=TestData1(:,[1 2 3 4 5 7 8 9 12 13 14 16 17 18 19 20 22 25 26 27 28]);
%save TrainData1_raw

%% DATA PREPROCESSING-SMOOTHING (REMOVE NOISE)
%load TrainData1_raw
% Step1-Define operating regime segments (cluster) before regime-aware smoothing
% Operating condition variables
opVars = {'Op_Set1','Op_Set2','Op_Set3'};
Xop = TrainData1_raw{:, opVars};

% K-means clustering (1 for FD001)
numClusters = 1;

[idx, C] = kmeans(Xop, numClusters, 'Replicates',10, 'MaxIter',1000);

% Add Cluster label to dataset
TrainData1_raw.ClusterID = idx;

%Step 2 — Both engine-level and regime-aware smoothing WITHIN segments
engines = unique(TrainData1_raw.Engine_ID);
% We must split continuous segments, not just cluster groups.
sensorVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'}; % 14 sensor as input data

for e = 1:length(engines)
    idxE = TrainData1_raw.Engine_ID == engines(e);
    
    dataE = TrainData1_raw(idxE,:);
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
    
    TrainData1_raw(idxE,:) = dataE;
end
TrainData1s=TrainData1_raw

% Step3-Validation Data regime-aware smoothing
Xop_val = ValidationData1_raw{:, opVars};
idx_val = knnsearch(C, Xop_val); % closest centroid
% Add Cluster label to dataset
ValidationData1_raw.ClusterID = idx_val;
%Step 2 — Smooth WITHIN segments
engines = unique(ValidationData1_raw.Engine_ID);
for e = 1:length(engines)
    idxE = ValidationData1_raw.Engine_ID == engines(e);
    
    dataE = ValidationData1_raw(idxE,:);
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
    
    ValidationData1_raw(idxE,:) = dataE;
end
ValidationData1s=ValidationData1_raw

% Step4-Test Data regime-aware smoothing
Xop_test = TestData1_raw{:, opVars};
idx_test = knnsearch(C, Xop_test); % closest centroid
% Add Cluster label to dataset
TestData1_raw.ClusterID = idx_test;
%Step 2 — Smooth WITHIN segments
engines = unique(TestData1_raw.Engine_ID);
for e = 1:length(engines)
    idxE = TestData1_raw.Engine_ID == engines(e);
    
    dataE = TestData1_raw(idxE,:);
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
    
    TestData1_raw(idxE,:) = dataE;
end
TestData1s=TestData1_raw
%save TrainData1s

ID1=TrainData1s(TrainData1s.Engine_ID==1,:);
% We have 14 sensors for each engine.
figure
for i=1:15
    subplot(5,3,i)
    plot(ID1.Time, ID1{:,5+i})
    title(ID1.Properties.VariableNames{5+i})
    xlabel('Time')
end

%% FEATURE ENGINEERING (ENGINE-LEVEL ONLY)
%load TrainData1s
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
engines = unique(TrainData1s.Engine_ID);

% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    TrainData1s.(["d" + sensor]) = NaN(height(TrainData1s),1);
    TrainData1s.(sensor + "_trend5") = NaN(height(TrainData1s),1);
    TrainData1s.(sensor + "_trend10") = NaN(height(TrainData1s),1);
end

% TRAINING DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = TrainData1s.Engine_ID == engines(e);
    
    dataE = TrainData1s(idxE,:);
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
    TrainData1s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(TrainData1s),1);
engines = unique(TrainData1s.Engine_ID);

for e = 1:length(engines)
    idxE = find(TrainData1s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(TrainData1s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

TrainData1sf = TrainData1s(rowsToKeep,:);

% For validation data, add the same features. 
engines = unique(ValidationData1s.Engine_ID);
% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    ValidationData1s.(["d" + sensor]) = NaN(height(ValidationData1s),1);
    ValidationData1s.(sensor + "_trend5") = NaN(height(ValidationData1s),1);
    ValidationData1s.(sensor + "_trend10") = NaN(height(ValidationData1s),1);
end

% VALIDATION DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = ValidationData1s.Engine_ID == engines(e);
    
    dataE = ValidationData1s(idxE,:);
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
    ValidationData1s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(ValidationData1s),1);
engines = unique(ValidationData1s.Engine_ID);

for e = 1:length(engines)
    idxE = find(ValidationData1s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(ValidationData1s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

ValidationData1sf = ValidationData1s(rowsToKeep,:);


% For test data, add the same features. 
engines = unique(TestData1s.Engine_ID);

% Initialize feature columns
for s = 1:length(sensorVars)
    sensor = sensorVars{s};
    
    TestData1s.(["d" + sensor]) = NaN(height(TestData1s),1);
    TestData1s.(sensor + "_trend5") = NaN(height(TestData1s),1);
    TestData1s.(sensor + "_trend10") = NaN(height(TestData1s),1);
end

% TEST DATA FEATURE ENGINEERING
for e = 1:length(engines)
    
    idxE = TestData1s.Engine_ID == engines(e);
    
    dataE = TestData1s(idxE,:);
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
    TestData1s(idxE,:) = dataE(order,:);
    
end

%Delete first 10 rows of each engine
rowsToKeep = true(height(TestData1s),1);
engines = unique(TestData1s.Engine_ID);

for e = 1:length(engines)
    idxE = find(TestData1s.Engine_ID == engines(e));
    % time order
    [~, order] = sort(TestData1s.Time(idxE));
    idxE = idxE(order);
    
    nRemove = min(10, length(idxE));
    rowsToKeep(idxE(1:nRemove)) = false;
end

TestData1sf = TestData1s(rowsToKeep,:);


%save TrainData1sf
ID1=TrainData1sf(TrainData1sf.Engine_ID==1,:)
%% TRAINING DATA NORMALIZATION
%load TrainData1sf
% STEP 2 — Cluster-based normalization (TRAIN)
sensorVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32',...
    'dT24','dT30','dT50','dP30','dNf','dNc','dPs30','dphi','dNRf','dNRc','dBPR','dhtBleed','dW31','dW32',...
    'T24_trend5','T30_trend5','T50_trend5','P30_trend5','Nf_trend5','Nc_trend5','Ps30_trend5',...
    'phi_trend5','NRf_trend5','NRc_trend5','BPR_trend5','htBleed_trend5','W31_trend5','W32_trend5',...
    'T24_trend10','T30_trend10','T50_trend10','P30_trend10','Nf_trend10','Nc_trend10','Ps30_trend10',...
    'phi_trend10','NRf_trend10','NRc_trend10','BPR_trend10','htBleed_trend10','W31_trend10','W32_trend10'}; % 14 sensor + 42 features as input data
numClusters=1;

% Calculate mean & std for each cluster
clusterStats = struct();

for k = 1:numClusters
    idx = TrainData1sf.ClusterID == k;
    
    data_k = TrainData1sf{idx, sensorVars};
    
    mu = mean(data_k, 1);
    sigma = std(data_k, 0, 1);
    
    % zero std protection
    sigma(sigma == 0) = 1;
    
    clusterStats(k).mu = mu;
    clusterStats(k).sigma = sigma;
end

% Normalize TRAIN
Xnorm = zeros(size(TrainData1sf{:, sensorVars}));

for k = 1:numClusters
    idx = TrainData1sf.ClusterID == k;
    
    X = TrainData1sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm(idx,:) = (X - mu) ./ sigma;
end

TrainData1sf{:, sensorVars} = Xnorm;
TrainData1sfn=TrainData1sf

%% VALIDATION AND TEST DATA NORMALIZATION
% Validation normalization
Xnorm_val = zeros(size(ValidationData1sf{:, sensorVars}));

for k = 1:numClusters
    idx = ValidationData1sf.ClusterID == k;
    
    X = ValidationData1sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm_val(idx,:) = (X - mu) ./ sigma;
end

ValidationData1sf{:, sensorVars} = Xnorm_val;
ValidationData1sfn=ValidationData1sf

% Test normalization
Xnorm_test = zeros(size(TestData1sf{:, sensorVars}));

for k = 1:numClusters
    idx = TestData1sf.ClusterID == k;
    
    X = TestData1sf{idx, sensorVars};
    
    mu = clusterStats(k).mu;
    sigma = clusterStats(k).sigma;
    
    Xnorm_test(idx,:) = (X - mu) ./ sigma;
end

TestData1sf{:, sensorVars} = Xnorm_test;
TestData1sfn=TestData1sf
%save TrainData1sfn
ID1=TrainData1sfn(TrainData1sfn.Engine_ID==1,:);
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
%% CREATING HEALTH INDEX AS A NEW FEATURE
%load TrainData1sfn
rng(1,"twister"); %Fix the global random seed for reproducibility
basic_vars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'}; %Basic 14 sensors

% FOR TRAINING DATA
Xtrain = TrainData1sfn{:, basic_vars};
%Apply PCA (Fit PCA ONLY on training data)
[coeff, scoreTrain, latent, tsquared, explained, mu] = pca(Xtrain);
% Use first principal component (PC1) as Health Index
HI_train = scoreTrain(:,1);
%Ensure degradation direction (HI should decrease with time)
corr_val = corr(HI_train, TrainData1sfn.Time);
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
TrainData1sfn.HealthIndex = HI_train_norm;

%FOR VALIDATION DATA
Xval = ValidationData1sfn{:, basic_vars};
% Project validation onto TRAIN PCA space
scoreVal = (Xval - mu) * coeff;

HI_val = scoreVal(:,1);

% Apply TRAIN normalization
HI_val_norm = (HI_val - mu_HI) / std_HI;

ValidationData1sfn.HealthIndex = HI_val_norm;

%FOR TEST DATA
Xtest = TestData1sfn{:, basic_vars};
% Project test onto TRAIN PCA space
scoreTest = (Xtest - mu) * coeff;

HI_test = scoreTest(:,1);

% Apply TRAIN normalization
HI_test_norm = (HI_test - mu_HI) / std_HI;

TestData1sfn.HealthIndex = HI_test_norm;
% We have totally 57 features consist of 14 sensor data and 43 generated data
%save TestData1sfn
%% FEATURE RANKING and SELECTION OF TOP 20 PREDICTORS
%load TestData1sfn
rng(1,"twister"); %Fix the global random seed fr reproducibility

% Define feature set (same as used in normalization + HI)
featureVars = {'T24','T30','T50','P30','Nf','Nc','Ps30','phi',...
    'NRf','NRc','BPR','htBleed','W31','W32',...
    'dT24','dT30','dT50','dP30','dNf','dNc','dPs30','dphi','dNRf','dNRc','dBPR','dhtBleed','dW31','dW32',...
    'T24_trend5','T30_trend5','T50_trend5','P30_trend5','Nf_trend5','Nc_trend5','Ps30_trend5',...
    'phi_trend5','NRf_trend5','NRc_trend5','BPR_trend5','htBleed_trend5','W31_trend5','W32_trend5',...
    'T24_trend10','T30_trend10','T50_trend10','P30_trend10','Nf_trend10','Nc_trend10','Ps30_trend10',...
    'phi_trend10','NRf_trend10','NRc_trend10','BPR_trend10','htBleed_trend10','W31_trend10','W32_trend10',...
    'HealthIndex'};

X = TrainData1sfn{:, featureVars};

% Replace with your actual class label variable
Y = TrainData1sfn.RUL;

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
save selectedFeatures1.mat topFeatures

%% Select Top 20 common predictors from FD001 and FD002.
%load TestData1sfn
%load selectedFeatures1
%load selectedFeatures2
rng(1,"twister"); %Fix the global random seed fr reproducibility
% Here, the 20 common predictors of FD001 and FD002 datasetare selected.
%Prepare the datasets for next 2ndstage pipeline according to the selected predictors
%Mixed20= HI+12 sensor+3 mid-term +4 short-term (Replace 'BPR_trend10' with 'P30_trend10')
columns_top_pred= {'Engine_ID','Time','HealthIndex','dW32','NRf_trend5','T24_trend5','NRc_trend5','Nc','Nf','T24','T30','T50','W32','htBleed','P30','Ps30','phi','BPR','W31','NRc_trend10', 'P30_trend10','NRf_trend10','TTF','RUL'};

TrainData1sfnr=TrainData1sfn(:, columns_top_pred);
TestData1sfnr=TestData1sfn(:, columns_top_pred);
ValidationData1sfnr=ValidationData1sfn(:, columns_top_pred);
%save TrainData1sfnr

%% Using classificationLearner for generating function of model
load TrainData1sfnr
rng(1,"twister"); %Fix the global random seed
classificationLearner
%33 ML models available in the classificationLearner app. are trained with
% cost sensitive learning by using TrainData1sfnr
%% Use Generated Function from App for External Training of models
% Add rng(1,"twister") inside the generated function to fix the global
% random seed, then save it as trainClassifier.m, and run this code for
% external training.

rng(1,"twister"); %Fix the global random seed
[trainedClassifier, validationAccuracy] = trainClassifier(TrainData1sfnr)
% After externally training data with a specific model's generated function,
% apply external validation process below.

%% EXTERNAL VALIDATION PROCESS of models
rng(1,"twister"); %Fix the global random seed
[yfit_all,scores] = trainedClassifier.predictFcn(ValidationData1sfnr);
Xp_all=tabulate(yfit_all)
%Number of predicted classes in 3752 validation data
Xr_all=tabulate(ValidationData1sfnr.RUL);
%Number of true classes (e.g. 260 alarms) in 3752 validation data
C_nn1=confusionmat(ValidationData1sfnr.RUL,yfit_all);
confusionchart(ValidationData1sfnr.RUL,yfit_all);
% According to the opening confuion matrix, Alarm recall, precision and 
% F1-score are calculated.

%Evaluate Validation Results
mukayese=table(ValidationData1sfnr.Engine_ID,ValidationData1sfnr.Time,ValidationData1sfnr.RUL,yfit_all,ValidationData1sfnr.TTF);
mukayese.Properties.VariableNames=[{'Engine_ID'} {'Time'} {'True_Class'} {'Predicted_Class'} {'True_RUL'}];
mukayese1=mukayese(mukayese.True_Class=='alarm',:);

% Finally, top 5 candidate models are selected according to the validation
% F1-score, at the end of this pipeline1. Candidate models are as follows:
% 2.22. Ensemble Bossted Trees with cost 9-10-8-9
% 25. Coarse KNN with cost 4-5-3-4
% 24. Ensemble Bagged Trees with cost 4-5-3-4
% 18. Narrow NN with cost 4-5-3-4
% 19. Medium NN with cost 4-5-3-4

