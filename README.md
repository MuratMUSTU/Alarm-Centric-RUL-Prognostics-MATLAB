# Alarm-Centric Prognostic Framework for Aircraft Turbofan Engines (MATLAB Implementation)
This repository contains the MATLAB implementation of the alarm-centric, decision-oriented prognostic framework for aircraft turbofan engines. The proposed methodology reformulates remaining useful life (RUL) estimation as a three-state degradation classification problem (normal, warning, alarm) aligned with maintenance decision thresholds.
The framework is developed and validated on the publicly available NASA C-MAPSS FD001 and FD002 datasets, and integrates feature engineering with ranking, cost-sensitive classification, group-aware cross validation, Bayesian hyperparameter optimization, hierarchical threshold calibration, engine-level decision aggregation, and maintenance-oriented evaluation strategies for predictive maintenance (PdM) applications.
# 1. Repository Structure
# 1.1 MATLAB Implementation Files
# •	MUSTU1_FD001_Class3.m
Model screening stage of alarm-centric classification pipeline to determine top five candidate models for FD001 dataset, including preprocessing, feature selection, model training in Classification Learner App, external validation and evaluation.
# •	MUSTU2_FD001_Class3.m
Decision-Oriented Optimization stage of alarm-centric classification pipeline to select final best model for FD001 dataset, including cost-sensitive learning, group-aware cross validation, Bayesian hyperparameter optimization, hierarchical threshold calibration, engine-level decision aggregation, and RUL prediction of alarm engines.
# •	MUSTU5_FD002_Class3.m
FD002 deployment pipeline for cross-condition evaluation using FD001-optimized configuration, with model retraining on FD002 and fixed decision thresholds for performance assessment.
# •	MUSTU3_FD001_Regression_Comparison.m
Conventional regression-based RUL baseline model for FD001.
# •	MUSTU4_FD001_CostAware_Regression_Comparison.m
Cost-aware regression baseline incorporating weighted penalties for alarm and warning classes.
# •	alarmF1Loss.m
Custom evaluation function implementing alarm-oriented F1-score for imbalanced classification and failure detection assessment.
# 1.2 Data Files
# •	engine_data1.mat – FD001 training dataset
# •	engine_testdata1_Unhead.mat – FD001 test dataset (truncated trajectories)
# •	engine_RULdata1.mat – Ground-truth RUL values for FD001 test engines
# •	engine_data2.mat – FD002 training dataset
# •	selectedFeatures1.mat – mRMR-based feature ranking (FD001, top-20 predictors)
# •	selectedFeatures2.mat – mRMR-based feature ranking (FD002, top-20 predictors)
# 2. System Requirements
# •	MATLAB R2024a (or later recommended)
# •	Statistics and Machine Learning Toolbox
# •	Predictive Maintenance Toolbox
No external or third-party dependencies are required.
# 3. Methodological Overview
The implemented framework consists of a two-stage alarm-centric prognostic architecture:
1.	Stage I – Model Screening:
Multiple (33) supervised machine learning classifiers are evaluated under cost-sensitive learning to address severe class imbalance in the alarm state.
2.	Stage II – Decision-Oriented Optimization:
Selected models are refined using:
o	Bayesian hyperparameter optimization
o	Nested engine-level cross-validation
o	Hierarchical threshold calibration
o	Engine-level decision aggregation
The final output is both cycle-level and engine-level maintenance-oriented health-state predictions (normal / warning / alarm) rather than continuous RUL regression. Therefore, the framework emphasizes maintenance-oriented decision support.
# 4. Execution Workflow
To reproduce the results reported in the manuscript, the following pipeline should be executed:
4.1 FD001 – Model Development and Selection
1.	Load FD001 dataset from .mat files
2.	Execute preprocessing and feature engineering pipeline
3.	Stage I – Model screening:
Run multiple supervised classifiers under cost-sensitive learning using MATLAB Classification Learner App
4.	Select top-performing model families based on validation alarm-class F1-score
5.	Stage II – Model optimization:
Apply Bayesian hyperparameter optimization, nested engine-level cross-validation, and cost-sensitive learning
6.	Apply hierarchical threshold calibration (alarm and warning thresholds)
7.	Perform engine-level decision aggregation and compute final cycle-level and engine-level performance metrics
Example (FD001 full pipeline):
run MUSTU1_FD001_Class3.m (Stage I – model screening)
run MUSTU2_FD001_Class3.m (Stage II – optimization & final model)
4.2 Baseline Comparisons (FD001)
run MUSTU3_FD001_Regression_Comparison.m
run MUSTU4_FD001_CostAware_Regression_Comparison.m
4.3 FD002 – Cross-Condition Deployment Evaluation
1.	Load FD002 dataset 
2.	Apply identical preprocessing pipeline and feature set selected from FD001 
3.	Retrain the Narrow NN model using the same architecture, cost matrix, and hyperparameter configuration optimized on FD001 
4.	Apply the same calibrated decision thresholds derived from FD001 without re-tuning 
5.	Perform cycle-level prediction and engine-level aggregation 
6.	Evaluate performance under multiple operating conditions and regime variations
Example (FD002 full pipeline):
run MUSTU5_FD002_Class3.m
# 5. Outputs
The implementation generates the following outputs:
# •	Cycle-level health-state predictions (normal / warning / alarm)
# •	Engine-level alarm detection results
# •	Alarm-oriented precision, recall, and F1-score
# •	Maintenance prioritization rankings based on alarm proximity
# •	Cost-sensitive evaluation metrics
# •	Comparative results for regression and classification baselines
# 6. Dataset Information
This study uses the NASA C-MAPSS turbofan engine degradation dataset:
# •	FD001: Single operating condition, single fault mode
# •	FD002: Multiple (6) operating conditions, multiple flight regimes
The dataset is publicly available and is partially redistributed in this repository.
# 7. Reproducibility Notes
# •	All experiments are fully reproducible using the provided MATLAB scripts
# •	Feature selection rankings are precomputed and stored in .mat files
# •	Random seeds are fixed where applicable to ensure deterministic results
# •	Engine-level partitioning is used to prevent data leakage across trajectories
# 8. License
This repository is intended for academic and research purposes only.
# 9. Contact
For questions, collaborations, or research inquiries:
Murat MUSTU
Kocaeli University, Türkiye
Email: [226164002@kocaeli.edu.tr]
ORCID: 0000-0003-1246-0074
