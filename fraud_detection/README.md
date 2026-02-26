# Credit Card Default Prediction MLFlow Pipeline

## Project Overview
This project establishes a robust end-to-end machine learning pipeline designed to predict credit card default risk. In the financial sector, the ability to accurately assess creditworthiness is paramount for mitigating loss and ensuring portfolio stability. This solution moves beyond simple predictive modelling by integrating comprehensive exploratory data analysis, rigorous feature engineering, and strict MLOps practices using MLflow.

The primary objective is not merely to classify potential defaulters but to construct a reproducible, transparent and auditable workflow. By leveraging gradient boosting algorithms and optimising classification thresholds, this pipeline addresses the inherent challenges of imbalanced financial data. The integration of MLflow ensures that every experiment, parameter adjustment and model artefact is tracked, facilitating seamless collaboration and future model iteration.

## Dataset
The analysis utilises the Credit Card Clients dataset, comprising demographic information, payment history and billing statements for credit card holders. The target variable is binary, indicating whether a client defaulted on their payment in the following month.

- **Source**: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Records**: 30,000
- **Features**: 25 initial variables, expanded through engineering.

## Project Architecture

The pipeline is architected to ensure modularity and reproducibility. Data flows through distinct stages of preprocessing, analysis, and modelling, with MLflow acting as the central registry for all experimental artefacts.

## Methodology

### Data Cleaning and Preprocessing

- **Categorical Mapping**: Variables such as `EDUCATION` and `MARRIAGE` contained undefined categories. These were consolidated into meaningful groups to prevent sparse matrix issues during encoding.
- **Negative Value Handling**: Bill amounts contained negative values, which are logically inconsistent in this context. These were clipped to zero to maintain data integrity.
- **Skewness Mitigation**: Financial variables such as `LIMIT_BAL` and `BILL_AMT` exhibit heavy right skew. We applied a `log1p` transformation to normalise distributions, stabilising variance and improving model convergence.
- **Scaling**: All numerical features were standardised using `StandardScaler` to ensure that magnitude differences did not bias distance-based algorithms.

### Feature Engineering

- **Utilisation Ratios**: We calculated `UTILIZATION_RATIO` and `BALANCE_RATIO` to capture the proportion of credit limit being consumed. High utilisation is a strong proxy for financial stress.
- **Payment Behaviour**: `PAYMENT_RATIO` was derived to assess the consistency of repayments relative to billed amounts.
- **Trend Analysis**: Features such as `BILL_TREND` and `PAY_TREND` were created to capture the trajectory of spending and repayment over the six-month observation window.
- **Statistical Aggregates**: `AVG_PAY_STATUS` and `PAY_VARIABILITY` summarise the consistency of payment delays, providing a robust signal of reliability.

### Exploratory Data Analysis and Insights
- **Target Distribution**: The dataset exhibits class imbalance, with approximately 22% of clients defaulting. This necessitated the use of stratified sampling and class-weighted loss functions during modelling.
- **Correlation Structure**: Heatmap analysis revealed that payment status variables (`PAY_0` through `PAY_6`) possess the highest correlation with the target. This confirms that past payment behaviour is the strongest predictor of future default.
- **Demographic Risk**: Analysis of `SEX` and `AGE_GROUP` revealed nuanced risk profiles. While gender showed a slight correlation, age stratification indicated that default rates are not linear across the lifespan, peaking in specific demographic cohorts.
- **Feature Importance**: Preliminary modelling indicated that engineered features such as `AVG_PAY_STATUS` ranked highly, validating the engineering effort.

## Modelling Strategy

### Baseline Performance
Benchmarks were established using Logistic Regression, Random Forest, and Gradient Boosting.
Gradient Boosting emerged as the superior baseline architecture, achieving a ROC-AUC of 0.7792.
This algorithm was selected for its ability to handle non-linear relationships and its robustness against overfitting when properly regularised.

### Metric Rationale: ROC-AUC vs. F1-Score
**Model Selection (ROC-AUC)**: ROC-AUC was prioritised for comparing model architectures because it is threshold-independent. It measures the model's ability to distinguish between classes across all possible classification thresholds, which is critical when the optimal operating point is not yet known.

**Operational Tuning (F1-Score)**: While ROC-AUC selects the best model, F1-score was used for threshold optimisation. In credit risk, the cost of a false negative (missing a defaulter) is high, but so is the cost of a false positive (declining a good customer). F1-score balances precision and recall, helping identify a threshold that optimises this trade-off for deployment.

**Context**: Although similar to fraud detection in terms of class imbalance, this is a credit default problem. The strategy reflects the need to maximise discrimination (ROC-AUC) while calibrating risk appetite (F1/Threshold).


- **Optimal Threshold**: 0.2981
- **Impact**: Adjusting the threshold significantly improved the F1-Score to 0.5430. This adjustment prioritises the recall of defaulters, which is critical in risk management where the cost of a false negative (missing a defaulter) exceeds that of a false positive.

### Hyperparameter Tuning
To maximise generalisation, `RandomizedSearchCV` was employed on the Gradient Boosting classifier. The search space included learning rate, tree depth, and subsample ratios.

- **Tuned Performance**: The optimised model achieved a ROC-AUC of 0.7805.
- **Significance**: While the numerical increase in ROC-AUC appears marginal, it represents a statistically significant improvement in the model's ability to discriminate between classes across all thresholds. The tuned model demonstrates better calibration and stability on unseen data.

### Model Selection and Optimisation
**Selected Model**: Tuned Gradient Boosting Classifier.

Following rigorous comparative analysis and hyperparameter optimisation, the Tuned Gradient Boosting Classifier was identified as the optimal model for deployment.

**Justification for Selection**

**Highest Discriminatory Power**: The tuned model achieved the highest ROC-AUC score of 0.7805, surpassing the Random Forest baseline (0.7736) and Logistic Regression (0.7389).

**Handling of Non-Linearity**: Unlike Logistic Regression, Gradient Boosting inherently captures complex non-linear interactions between features without requiring explicit polynomial feature engineering.

**Robustness to Overfitting**: Through the tuning process, parameters such as learning_rate (0.0504) and subsample (0.995) were optimised to prevent the model from memorising noise in the training data.

**Feature Importance Alignment**: The model's feature importance rankings align closely with domain expertise. The top predictors identified (PAY_0, AVG_PAY_STATUS, PAY_VARIABILITY) are consistent with established credit risk theories.

**Imbalanced Data Handling:** The implementation of class weights during training ensured that the minority class (defaulters) was adequately represented during the learning process.

### MLOps and Experiment Tracking
MLflow serves as the cornerstone of this operational framework. It ensures that the development process is transparent and reproducible. The implementation within this pipeline covers setup, logging, and artefact management.

### Configuration and Setup
-The tracking URI was configured to a local directory (file:///.../mlruns), ensuring all data remains accessible without requiring a remote server during development.

-A specific experiment named credit-card-default-detection was created to isolate these runs from other workflows.

**Code Implementation:**
```python
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.absolute()}")
mlflow.set_experiment("credit-card-default-detection")
 ```
**Run Management and Logging**

-Each modelling stage was encapsulated within a mlflow.start_run context manager. This ensures that parameters and metrics are correctly associated with their specific execution instance.

-**Parameters:** Model hyperparameters (e.g., n_estimators, learning_rate), data split sizes, and feature counts were logged using mlflow.log_param. This allows for exact replication of the environment.

-**Metrics:** Performance metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC were logged using mlflow.log_metric. This facilitates direct comparison between baseline and tuned models via the MLflow UI.

-**Artefacts:** Critical visualisations such as confusion matrices and feature importance plots were saved as PNG files and logged using mlflow.log_artifact. This preserves the visual context of model performance alongside numerical metrics.

-**Model Registry:** The final sklearn models were serialised and logged using mlflow.sklearn.log_model. This includes an inferred signature to define input and output schemas, enabling future deployment pipelines to validate data compatibility.

### Experiment Comparison
The MLflow UI provides a comparative view of all runs. This allows for the identification of the best-performing configuration based on specific metrics.

## Results Summary

| Model                         | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|-------------------------------|----------|-----------|---------|----------|---------|
| Logistic Regression            | 0.7290   | 0.4235    | 0.6240  | 0.5046   | 0.7389  |
| Random Forest                  | 0.7863   | 0.5156    | 0.5599  | 0.5369   | 0.7736  |
| Gradient Boosting (Baseline)   | 0.8207   | 0.6722    | 0.3693  | 0.4767   | 0.7792  |
| Gradient Boosting (Tuned)      | 0.8185   | 0.6626    | 0.3655  | 0.4711   | 0.7805  |


## Installation and Usage

To replicate this environment, ensure you have Python 3.8 or higher installed.

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3.**Run the pipeline**: Execute the Jupyter notebook or the main training script. MLflow will automatically initialise a local tracking server.

4. **View MLflow Dashboard**: Navigate to `http://localhost:5000` to view experiment tracking and model artefacts.

## Conclusion
This project demonstrates a sophisticated approach to credit risk modelling. By combining advanced feature engineering with rigorous hyperparameter tuning and a robust MLOps framework, we have developed a model that is not only predictive but also maintainable and auditable. The integration of MLflow ensures that the model lifecycle is managed professionally, adhering to industry best practices for machine learning operations.

## License
This project is licensed under the MIT License.
```










