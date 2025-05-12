Here is an updated version of your README file, now including all the files you've listed from your project directory. I've also clarified and organized some points for better structure and professionalism.

---

# Loan Prediction System

## Project Overview

This project develops a machine learning model to predict loan approval outcomes for **Motz Financial Services**. The model leverages applicant data such as gender, marital status, income, and credit history to determine loan eligibility. A key focus is identifying the most important features influencing loan approval. Preprocessed training and testing datasets are saved for inspection, and a dashboard provides interactive visualizations.

## Dataset

The dataset (`loan_data_set.csv`) contains applicant details and loan status. Key features include:

* `ApplicantIncome`
* `CoapplicantIncome`
* `LoanAmount`
* `Credit_History`
* `Loan_Status` (target)

## Methodology

1. **Data Cleaning**: Missing values were handled using mode for categorical features and median for numerical features. Outliers were removed using the IQR method.
2. **Preprocessing**: A `Total_Income` feature was engineered; categorical variables were encoded and numerical features were scaled.
3. **Data Splitting**: The dataset was randomly split into 80% training and 20% testing.
4. **Modeling**: Logistic Regression was chosen due to its performance and interpretability.
5. **Model Tuning**: GridSearchCV was used to optimize hyperparameters.
6. **Feature Importance**: Top contributing features were identified and ranked.
7. **Dashboard**: An interactive Streamlit dashboard was developed to visualize results and insights.

## Files Included

### Main Files

* `loanprediction.ipynb` – Jupyter notebook with full exploratory data analysis and modeling steps
* `dashboard.py` – Streamlit dashboard app
* `tuned_logistic_regression.pkl` – Trained and tuned Logistic Regression model

### Datasets

* `loan_data_set.csv` – Original dataset
* `loan_data_cleaned_missing_values.csv` – Cleaned dataset with missing values handled
* `loan_data_cleaned_outliers.csv` – Dataset after outlier removal
* `loan_data_with_total_income_emi.csv` – Feature-engineered dataset
* `loan_data_preprocessed.csv` – Fully preprocessed dataset
* `model_training_results.csv` – Summary of model performance
* `model_tuning_logistic_regression_results.csv` – Grid search results

### Visualizations

* `boxplots_before_outlier_removal.png`
* `boxplots_after_outlier_removal.png`
* `confusion_matrix.png`
* `confusion_matrix_logistic_regression.png`
* `credit_history_vs_loan_status.png`
* `gender_vs_loan_status.png`
* `income_distribution.png`
* `loan_amount_distribution.png`
* `loan_amount_term_distribution.png`
* `loan_status_distribution.png`
* `married_vs_loan_status.png`
* `property_area_vs_loan_status.png`

### Documentation

* `group.docx` – Group report and documentation

### Other

* `.gitignore` – Files and directories to be excluded from Git tracking
* `assets/` – (Directory for storing visualization and dashboard-related assets)

## Usage

1. **Install Dependencies**:

   ```bash
   pip install pandas numpy sklearn  matplotlib seaborn joblib 
   ```

2. **Run the Dashboard**:

   ```bash
   python run dashboard.py
   ```

3. **Model Inference**:
   Load and use the trained model:

   ```python
   from joblib import load
   model = load('tuned_logistic_regression.pkl')
   ```

4. **View Data**:
   Inspect any of the CSV files mentioned in the Datasets section to review the processing pipeline.

## Requirements

* Python 3.8+
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

## Team Members

* **Refuoe**
* **Sello**
* **Ramone**
* **Mosololi**
* **Heqoa**

