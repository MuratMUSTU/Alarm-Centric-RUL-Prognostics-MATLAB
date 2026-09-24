# Decision-Oriented Prognostics of Aircraft Turbofan Engines (MATLAB Implementation)
This repository provides the MATLAB implementation accompanying the study:
“Decision-Oriented Prognostics of Aircraft Turbofan Engines: From Failure-Proximity Learning to Engine-Level Alarm Decisions”
The repository implements an experimentally decomposed decision-oriented prognostic architecture for aircraft turbofan engines. Rather than treating prognostics exclusively as continuous remaining useful life (RUL) estimation, the framework formulates engine health in three failure-proximity states—normal, warning, and alarm—and explicitly separates:
1.	Failure-proximity learning
2.	Probability-based decision calibration
3.	Cycle-to-engine decision integration

The implementation combines cost-sensitive multi-state classification, engine-grouped model development, Bayesian hyperparameter optimization, probability-threshold calibration, and Trajectory-Aware Evidence Integration (TAEI) as a trajectory-aware hierarchical mechanism for integrating sequential alarm evidence into engine-level decisions.
Experiments are conducted on the publicly available NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001 and FD002 datasets. FD001 is used for the principal model-development and evaluation workflow, while FD002 is used to examine cross-condition robustness under alternative predictor-set strategies.

# 1. Repository Structure
# 1.1 MATLAB Implementation Files
# •	MUSTU1_FD001_Class3.m
Stage I – Model Screening
Implements the initial FD001 model-development workflow, including:

•	Data preprocessing

•	Feature engineering

•	Health-index construction

•	mRMR-based feature ranking and predictor selection

•	Cost-sensitive training

•	Evaluation of 33 classification models using MATLAB Classification Learner

•	External validation on the held-out validation engines

•	Selection of candidate model families based on alarm-class F1-score

# •	MUSTU2_FD001_Class3.m
Stage II – Decision-Oriented Model Development and Final Evaluation
Implements the main FD001 decision-oriented workflow, including:

•	Cost-sensitive model refinement

•	Engine-grouped cross-validation

•	Bayesian hyperparameter optimization

•	Alarm-oriented optimization using alarmF1Loss.m

•	Hierarchical probability-threshold calibration

•	Cycle-level failure-proximity prediction

•	TAEI-based cycle-to-engine decision integration

•	Engine-level alarm evaluation

The final selected FD001 model is an Ensemble Bagged Trees classifier.

# •	MUSTU3_FD001_Regression_Comparison.m
Implements the Random Forest (RF) regression baseline used for the controlled comparison between continuous RUL prediction and failure-proximity classification.

# •	MUSTU4_FD001_CostAware_Regression_Comparison.m
Implements the weighted Random Forest (Weighted RF) regression baseline using alarm-oriented cost weighting.

# •	MUSTU5_FD002_Class3.m
Implements the FD002 cross-condition robustness experiments. The classifier is retrained on FD002 while alternative predictor-set strategies and decision configurations are evaluated according to the experimental design reported in the manuscript.

# •	alarmF1Loss.m
Custom loss function implementing the alarm-class F1-based optimization objective used during decision-oriented model optimization.

# 1.2 Feature-Selection Files
The repository includes precomputed mRMR feature-selection results:

•	selectedFeatures1.mat — FD001 training-derived mRMR ranking and selected top-20 predictors

•	selectedFeatures2.mat — FD002 training-derived mRMR ranking and selected top-20 predictors

•	selectedFeatures3.mat — common top-20 predictors derived from the FD001 and FD002 training sets

These files support reproducibility of the predictor-set strategies examined in the manuscript.

# 2. System Requirements
•	MATLAB R2024a or later

•	Machine Learning Toolbox or Predictive Maintenance Toolbox

•	Deep Learning Toolbox for the deep-learning regression comparison models, where applicable

No external third-party software dependencies are required for the core MATLAB implementation.

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
1.	Load FD001 dataset
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
The dataset is publicly available and is not redistributed in this repository.
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
