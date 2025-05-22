# AI Preventive Maintenance Classifier

**Author:** Matheus Vilar Mota Santos
**Contact:** vilarmms@gmail.com
**LinkedIn:** https://www.linkedin.com/in/matheus-vilar-03708292/

## Introduction

This project addresses a logistics company's challenge of rising vehicle maintenance expenses, specifically targeting the air system. The core objective is to develop a supervised classification AI solution to automate preventive repair decisions, aiming to significantly reduce costs associated with delayed failure detection and unnecessary maintenance.

Historical data indicates a substantial increase in air system maintenance costs over the past three years. If current trends persist, these costs are projected to escalate further. This AI solution aims to predict and prevent costly air system failures by analyzing vehicle condition data, thereby optimizing maintenance schedules and reducing overall expenses.

The project leverages a dataset of historical and current vehicle data, including encoded features representing various vehicle conditions and corresponding labels indicating the necessity of repairs.

## Project Structure


.
├── Project_notebook.ipynb # Main Jupyter Notebook with analysis and modeling
├── datasets/
│ ├── air_system_previous_years.csv # Training data
│ └── air_system_present_year.csv # Test/current year data
├── models/
│ └── lightboost/
│ └── model.pkl # Saved trained LightGBM model
├── src/
│ ├── ExploratoryDataAnalysis.py # Python module for EDA utilities
│ └── Classifier.py # Python module for LightGBM classifier and optimization
├── README.md # This README file
└── .venv/ # (Optional) Virtual environment

## Exploratory Data Analysis (EDA)

The EDA phase involved:
1.  **Data Loading:** Importing training (`air_system_previous_years.csv`) and test (`air_system_present_year.csv`) datasets.
2.  **Initial Inspection:**
    *   Training set: 60,000 samples × 171 features.
    *   Test set: 16,000 samples × 171 features.
    *   Features are predominantly numeric (170) with one categorical target variable (`class`).
3.  **Missing Value Analysis:** Both datasets exhibit a similar percentage of missing values (around 8.3%). Features like `br_000`, `bq_000`, `bp_000`, `bo_000`, and `cr_000` have a high proportion of missing data (over 75%).
4.  **Missing Value Imputation:** Missing numerical values were imputed using the median for skewed distributions and the mean for normally distributed features.
5.  **Feature Skewness:** Several numeric features show high right-skewness (e.g., `cs_009`, `cf_000`).
6.  **Target Variable Analysis:** The target variable `class` is binary (`pos`, `neg`) and highly imbalanced, with the 'neg' class (no repair needed) being dominant (approx. 98%).
7.  **Correlation Analysis:** Spearman's rank correlation was used to identify and remove highly correlated features (threshold |r| >= 0.7), reducing redundancy and potential overfitting. 12 features were dropped from the training set based on this.
8.  **t-SNE Visualization:** After feature cleaning, t-SNE was employed to visualize the data distribution in a lower-dimensional space. The visualization revealed a non-linear separation between classes, suggesting that a non-linear model would be more appropriate.

## Modeling

### Model Choice
Based on the t-SNE visualization indicating non-linear class separation and the nature of the dataset, a **LightGBM (Light Gradient Boosting Machine)** model was selected. LightGBM is known for its efficiency and ability to handle large datasets and capture complex, non-linear relationships.

### Cost Function
A custom cost function was defined to align the model's optimization with the client's business objective of minimizing maintenance-related expenses:
*   True Positive (TP - vehicle inspection detects an error): **$25**
*   False Positive (FP - vehicle inspection does not detect an error, but inspection was done): **$10**
*   False Negative (FN - vehicle damaged and not sent for repair): **$500**
*   True Negative (TN - vehicle not damaged, no inspection): **$0**

The goal is to minimize: `Cost = (TP * 25) + (FP * 10) + (FN * 500)`

### Hyperparameter Optimization
Optuna was used for hyperparameter optimization of the LightGBM model, performing 200 trials to find the parameters that minimize the custom cost function on a validation set (derived from the test set in this notebook).

### Threshold Analysis
Given the significant cost disparity, particularly the high cost of false negatives, the classification threshold for predicting a 'pos' (repair needed) class is critical. A threshold analysis was performed to identify the optimal threshold that minimizes the overall cost function on the test set.

## Results

*   **Optimal Threshold:** 0.021
*   **Cost Function Value (on test set):** $19,295
*   **Accuracy:** 0.961
*   **ROC AUC:** 0.995
*   **Precision (Class 'pos'):** 0.375
*   **Recall (Class 'pos'):** 0.979
*   **F1 Score (Class 'pos'):** 0.542

The model achieves a very high recall for the 'pos' class, meaning it correctly identifies most vehicles requiring repair. This comes at the cost of lower precision, indicating some unnecessary inspections (false positives). However, this trade-off is desirable given the high cost of missing a necessary repair (false negative).

## Savings Analysis

By implementing this AI solution, the client is projected to achieve significant savings:
*   **Current Year (2025) Projected Savings:** Approximately $17,000 (compared to an estimated $41,000 cost without the AI model).
*   **Future Savings:** Savings are expected to increase in subsequent years, assuming the AI model's predictive performance is maintained and cost trends continue as projected.

It is crucial to monitor the model's performance in production for potential data drift, which might necessitate retraining to ensure continued accuracy and cost-effectiveness.

## How to Run

1.  **Setup Environment:**
    *   It's recommended to use a virtual environment.
    *   Install necessary packages:
        ```bash
        pip install pandas numpy matplotlib scipy seaborn scikit-learn lightgbm optuna tqdm
        ```
2.  **Place Data:** Ensure the `datasets` folder is in the root directory with `air_system_previous_years.csv` and `air_system_present_year.csv`.
3.  **Run Notebook:** Open and run the `Project_notebook.ipynb` Jupyter Notebook.
    *   The notebook contains options to either train a new model or load the pre-trained model from `models/lightboost/model.pkl`. Set the `train_model` variable accordingly.

## Future Work & Considerations

*   **Data Drift Monitoring:** Implement a system to monitor for data drift in production and trigger alerts for model retraining.
*   **Feature Engineering:** Explore further feature engineering techniques to potentially improve model performance.
*   **Alternative Models:** Investigate other advanced classification models, although LightGBM has shown strong performance.
*   **Explainability:** Incorporate model explainability techniques (e.g., SHAP) to understand feature importance and model decisions better.
*   **Continuous Feedback Loop:** Establish a feedback loop with the client to incorporate new data and adjust the cost function parameters as business priorities evolve.
