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
# 1.1 MATLAB Implementation Files (code)
# •	MUSTU1_FD001_Class3.m
Stage I – Model Screening

Implements the initial FD001 model-development workflow, including:

•	Data preprocessing

•	Feature engineering

•	Health-index construction

•	mRMR-based feature ranking and predictor selection

•	Cost-sensitive training

•	Evaluation of 33 classification models using MATLAB Classification Learner (models)

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

# 1.2 Feature-Selection Files (data)
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
The repository implements a two-stage model-development workflow within a broader three-function decision architecture.
# 3.1. Failure-Proximity Learning
Engine health is represented using three failure-proximity states:
•	Normal
•	Warning
•	Alarm

The baseline FD001 decision boundary defines:
•	Alarm: RUL < 13 cycles
•	Warning: 13 ≤ RUL < 38 cycles
•	Normal: RUL ≥ 38 cycles

The classification formulation is designed to align the predictive objective with failure-proximity recognition rather than relying exclusively on numerical RUL accuracy.
# 3.2. Stage I – Model Screening
Stage I evaluates 33 supervised classification models using cost-sensitive learning.
The workflow includes:

•	Training-data preprocessing
•	Feature engineering
•	Health-index construction
•	mRMR feature ranking
•	Selection of 20 predictors
•	Cost-sensitive classification
•	External validation using engine-disjoint validation data
•	Alarm-oriented model comparison

Candidate model families are selected primarily according to alarm-class F1-score on the held-out validation engines.
# 3.3. Stage II – Decision-Oriented Model Development
Selected candidate models are further developed using:

•	Engine-grouped cross-validation
•	Cost-sensitive learning
•	Bayesian hyperparameter optimization
•	An alarm-oriented objective based on 1 − F1_alarm
•	Deterministic random seeds where applicable
•	Hierarchical probability-threshold calibration

The probability calibration stage determines separate decision thresholds for the alarm and warning states using the held-out validation data.
# 3.4. Cycle-to-Engine Decision Integration
Cycle-level predictions are subsequently transformed into engine-level maintenance decisions.
Two cycle-to-engine decision transformations are considered:

1.	Common worst-case aggregation, used as a model-independent benchmark.

2.	Trajectory-Aware Evidence Integration (TAEI), the proposed engine-level decision layer.

TAEI is implemented as a trajectory-aware hierarchical decision mechanism that integrates sequential alarm evidence according to its temporal location, intensity, and consistency along the engine trajectory.
TAEI is a post-classifier decision layer. It does not modify the underlying cycle-level predictions or classifier parameters.
Its decision-policy configuration is fixed before independent-test evaluation and is not optimized using the independent test set.

# 4. Execution Workflow (FD001 Model Development and Evaluation)
Step 1 — Load the FD001 Dataset

Load the publicly available C-MAPSS FD001 training and test data.

Step 2 — Define Failure-Proximity States

Generate the normal, warning, and alarm labels according to the predefined RUL decision boundaries.

Step 3 — Engine-Level Data Partitioning

Partition the FD001 training engines at the engine level into development and held-out validation subsets.
This prevents observations from the same engine trajectory from appearing in both subsets.

Step 4 — Preprocessing and Feature Engineering

Execute the preprocessing pipeline, including:
•	Operating-regime processing
•	Causal smoothing
•	Sensor-derived feature engineering
•	Trend and cumulative features
•	Normalization
•	PCA-based health-index construction
•	Candidate predictor generation

The resulting candidate feature space contains 57 predictors, from which 20 predictors are selected using mRMR.

Step 5 — Stage I Model Screening

Run:
MUSTU1_FD001_Class3.m

The script supports the Stage I model-screening workflow. The 33 classification models are evaluated under cost-sensitive learning using MATLAB Classification Learner.
Following training, the selected model functions can be exported for external retraining and validation.

Step 6 — Candidate Model Selection

Select candidate model families using the held-out validation alarm-class F1-score.

Step 7 — Stage II Model Optimization

Run:
MUSTU2_FD001_Class3.m

The selected models are refined using engine-grouped cross-validation and Bayesian hyperparameter optimization.

Step 8 — Probability-Threshold Calibration

Calibrate the alarm and warning probability thresholds using the held-out validation engines.

Step 9 — Cycle-to-Engine Decision Integration

Apply the common worst-case aggregation and the TAEI-based engine-level decision layer to the cycle-level predictions.

Step 10 — Final Evaluation

Calculate:
•	Cycle-level precision, recall, and F1-score
•	Engine-level alarm recall, precision, and F1-score
•	Detected and missed alarm engines
•	False-alarm engines

The independent C-MAPSS test set is reserved for final evaluation after model-development decisions have been fixed.

# 5. FD001 Regression Comparisons
The repository includes the MATLAB implementations used for the principal regression-based comparison:

Random Forest Regression

Run:
MUSTU3_FD001_Regression_Comparison.m

Weighted Random Forest Regression

Run:
MUSTU4_FD001_CostAware_Regression_Comparison.m

The regression models are evaluated under a common alarm-oriented protocol using the same FD001-derived predictor set and predefined failure-proximity boundary.
The comparison is intended as a controlled methodological comparison, rather than as a comprehensive benchmark of all contemporary RUL prediction methods.
The manuscript additionally reports results for LSTM- and CNN-based regression models.

# 6. FD002 Cross-Condition Robustness
Run:
MUSTU5_FD002_Class3.m

FD002 contains six operating conditions and is used to examine cross-condition robustness within the   C-MAPSS benchmark.
The FD002 experiments evaluate three predictor-set strategies:
# 6.1 Fixed-Feature Strategy
The 20 predictor identities selected from FD001 are retained, while the classifier is retrained using FD002 training data.
# 6.2 Domain-Adapted Strategy
Predictors are selected using FD002 training data, and the classifier configuration is adapted to the FD002 development environment.
# 6.3 Common-Feature Strategy
A common set of 20 predictors is defined using training-data-based feature-selection procedures for FD001 and FD002.
The FD002 experiments use FD002-specific preprocessing and normalization derived from the FD002 training data.
FD002 is therefore not treated as a zero-shot transfer experiment or as external validation. Instead, it evaluates robustness under a specific cross-condition shift represented by the C-MAPSS benchmark.

# 7. TAEI: Trajectory-Aware Evidence Integration
TAEI is the engine-level decision mechanism proposed in the framework.
TAEI integrates sequential cycle-level alarm evidence using a hierarchical decision structure involving:

•	Recent trajectory evidence
•	Alarm-evidence consistency
•	Global trajectory evidence
•	A selective global fallback pathway

The baseline TAEI decision-policy configuration is fixed before independent-test evaluation.
TAEI is deliberately treated as a decision-layer component rather than an additional predictive model. Consequently, applying TAEI does not alter the cycle-level predictions or cycle-level performance metrics.
The repository therefore supports separate assessment of:

•	Predictive-model performance
•	Probability-threshold effects
•	Engine-level decision-layer effects
•	TAEI pathway activation
•	TAEI sensitivity to alternative fixed parameterizations (Supplementary Material under docs)

# 8. Outputs
The implementation generates or supports analysis of:

•	Cycle-level normal/warning/alarm predictions

•	Calibrated failure-proximity probabilities

•	Engine-level alarm decisions

•	Alarm precision, recall, and F1-score

•	Engine-level missed-alarm and false-alarm counts

•	Cost-sensitive classification results

•	Regression-versus-classification comparisons

•	TAEI pathway activation statistics

•	TAEI sensitivity and ablation results

•	FD002 cross-condition robustness results

# 9. Dataset Information
The experiments use the publicly available NASA C-MAPSS turbofan engine degradation benchmark.

FD001

•	100 training/test engine trajectories
•	Single operating condition
•	Single fault mode

FD002

•	260 training/259 test engine trajectories
•	6 operating conditions
•	Multiple operating regimes

The C-MAPSS datasets are not redistributed in this repository. Users should obtain the datasets from their official public source and place them in the appropriate local data directories before execution.

# 10. Reproducibility Notes
The repository is designed to support reproduction of the principal experiments reported in the manuscript.

•	Feature-selection results are provided in .mat files.

•	Engine-level partitioning is used to prevent trajectory-level data leakage.

•	Model-development decisions are separated from independent-test evaluation.

•	Random seeds are fixed where applicable.

•	FD001-derived model configurations and FD002-specific adaptation strategies follow the experimental protocols described in the manuscript.

•	The independent test data are not used for feature selection, hyperparameter optimization, probability-threshold calibration, or TAEI decision-policy modification.

•	TAEI parameters are treated as fixed decision-policy parameters for the primary evaluation.

Because MATLAB Classification Learner models are trained interactively during Stage I, users should follow the workflow described in the manuscript and in the corresponding MATLAB scripts when reproducing the model-screening stage.

# 11. License
This repository is intended for academic and research purposes only.

The C-MAPSS datasets are not included in the repository and remain subject to their respective terms of use.

# 12. Citation
If you use this repository or the associated methodology in your research, please cite the corresponding manuscript:

MUSTU, M., ARAS, F., & INAL, M.

Decision-Oriented Prognostics of Aircraft Turbofan Engines: From Failure-Proximity Learning to Engine-Level Alarm Decisions.

A DOI and formal citation will be added when available.

# 13. Contact
For questions, collaborations, or research inquiries:
Murat MUSTU
Kocaeli University, Türkiye
Email: [226164002@kocaeli.edu.tr]
ORCID: 0000-0003-1246-0074
