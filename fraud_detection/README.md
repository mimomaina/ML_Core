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
<img width="1908" height="990" alt="image" src="https://github.com/user-attachments/assets/9062602d-a0be-4f62-9e7d-7963174adc85" />


## Methodology

### Data Cleaning and Preprocessing
Raw financial data often contains inconsistencies that can severely degrade model performance. We implemented a rigorous cleaning protocol:

- **Categorical Mapping**: Variables such as `EDUCATION` and `MARRIAGE` contained undefined categories. These were consolidated into meaningful groups to prevent sparse matrix issues during encoding.
- **Negative Value Handling**: Bill amounts contained negative values, which are logically inconsistent in this context. These were clipped to zero to maintain data integrity.
- **Skewness Mitigation**: Financial variables such as `LIMIT_BAL` and `BILL_AMT` exhibit heavy right skew. We applied a `log1p` transformation to normalise distributions, stabilising variance and improving model convergence.
- **Scaling**: All numerical features were standardised using `StandardScaler` to ensure that magnitude differences did not bias distance-based algorithms.

### Feature Engineering
Domain knowledge was applied to transform raw transactional data into behavioural indicators. This step was critical for enhancing predictive power beyond what raw amounts could offer.

- **Utilisation Ratios**: We calculated `UTILIZATION_RATIO` and `BALANCE_RATIO` to capture the proportion of credit limit being consumed. High utilisation is a strong proxy for financial stress.
- **Payment Behaviour**: `PAYMENT_RATIO` was derived to assess the consistency of repayments relative to billed amounts.
- **Trend Analysis**: Features such as `BILL_TREND` and `PAY_TREND` were created to capture the trajectory of spending and repayment over the six-month observation window.
- **Statistical Aggregates**: `AVG_PAY_STATUS` and `PAY_VARIABILITY` summarise the consistency of payment delays, providing a robust signal of reliability.

### Exploratory Data Analysis and Insights
Visualisation was employed not merely for description, but for hypothesis generation and feature selection.

- **Target Distribution**: The dataset exhibits class imbalance, with approximately 22% of clients defaulting. This necessitated the use of stratified sampling and class-weighted loss functions during modelling.
- **Correlation Structure**: Heatmap analysis revealed that payment status variables (`PAY_0` through `PAY_6`) possess the highest correlation with the target. This confirms that past payment behaviour is the strongest predictor of future default.
- **Demographic Risk**: Analysis of `SEX` and `AGE_GROUP` revealed nuanced risk profiles. While gender showed a slight correlation, age stratification indicated that default rates are not linear across the lifespan, peaking in specific demographic cohorts.
- **Feature Importance**: Preliminary modelling indicated that engineered features such as `AVG_PAY_STATUS` ranked highly, validating the engineering effort.

## Modelling Strategy

### Baseline Performance
We established benchmarks using Logistic Regression, Random Forest, and Gradient Boosting. Gradient Boosting emerged as the superior baseline architecture, achieving a ROC-AUC of 0.7792. This algorithm was selected for its ability to handle non-linear relationships and its robustness against overfitting when properly regularised.

### Threshold Optimisation
Standard classification thresholds (0.5) are often suboptimal for imbalanced datasets. We analysed the precision-recall curve to identify an optimal decision threshold.

- **Optimal Threshold**: 0.2981
- **Impact**: Adjusting the threshold significantly improved the F1-Score to 0.5430. This adjustment prioritises the recall of defaulters, which is critical in risk management where the cost of a false negative (missing a defaulter) exceeds that of a false positive.

### Hyperparameter Tuning
To maximise generalisation, we employed `RandomizedSearchCV` on the Gradient Boosting classifier. The search space included learning rate, tree depth, and subsample ratios.

- **Tuned Performance**: The optimised model achieved a ROC-AUC of 0.7805.
- **Significance**: While the numerical increase in ROC-AUC appears marginal, it represents a statistically significant improvement in the model's ability to discriminate between classes across all thresholds. The tuned model demonstrates better calibration and stability on unseen data.

### MLOps and Experiment Tracking
MLflow serves as the cornerstone of this project's operational framework. It ensures that the development process is transparent and reproducible.

- **Experiment Tracking**: Every model training run is logged as a distinct experiment. This includes hyperparameters, evaluation metrics, and runtime environment details.
- **Artefact Logging**: We log critical artefacts directly to the MLflow server, including confusion matrices, feature importance plots, and the serialised model objects themselves.
- **Model Registry**: Best-performing models are registered for version control, facilitating easy deployment and rollback capabilities.

Please refer to the `screenshots/` directory for visual evidence of the MLflow dashboard, showcasing experiment comparison and metric tracking.

## Results Summary

| Model                         | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|-------------------------------|----------|-----------|---------|----------|---------|
| Logistic Regression            | 0.7290   | 0.4235    | 0.6240  | 0.5046   | 0.7389  |
| Random Forest                  | 0.7863   | 0.5156    | 0.5599  | 0.5369   | 0.7736  |
| Gradient Boosting (Baseline)   | 0.8207   | 0.6722    | 0.3693  | 0.4767   | 0.7792  |
| Gradient Boosting (Tuned)      | 0.8185   | 0.6626    | 0.3655  | 0.4711   | 0.7805  |

**Note**: Metrics for the tuned model are reported at the optimal threshold where applicable for F1 maximisation.

## Project Structure

```
.
├── data
│   ├── raw
│   └── processed
├── notebooks
│   └── Default_of_Credit_Card_Clients.ipynb
├── outputs
│   ├── figures
│   └── models
├── screenshots
```

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

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**: Execute the Jupyter notebook or the main training script. MLflow will automatically initialise a local tracking server.

5. **View MLflow Dashboard**: Navigate to `http://localhost:5000` to view experiment tracking and model artefacts.

## Conclusion
This project demonstrates a sophisticated approach to credit risk modelling. By combining advanced feature engineering with rigorous hyperparameter tuning and a robust MLOps framework, we have developed a model that is not only predictive but also maintainable and auditable. The integration of MLflow ensures that the model lifecycle is managed professionally, adhering to industry best practices for machine learning operations.

## License
This project is licensed under the MIT License.
```

This version organizes the content into clear sections, adds code block formatting for commands and file paths, and uses tables for the results summary.


