# Week4-Activitylog-Internship
Machine learning model development for Telco customer churn prediction training and comparing Logistic Regression, Decision Tree, Random Forest, SVM, and KNN classifiers.
# Telco Customer Churn — Machine Learning Model Development (Week 4, Phase II)

Machine learning model development for predicting customer churn, performed as Phase II of a Week 4 internship task. This phase uses the final feature-engineered dataset from Phase I to train, evaluate, and compare multiple classification models.

## Objective

To build and evaluate machine learning classification models capable of predicting customer churn, and to select the best-performing model based on standard classification metrics.

## Repository Contents

| File | Description |
|---|---|
| `Model_Development_Notebook.ipynb` | Main Jupyter/Colab notebook containing the full model development and evaluation workflow |
| `Model_Evaluation_Report.pdf` | Written report summarizing model performance and evaluation results |

## Workflow

The notebook follows these steps:

1. **Separate Features and Target** — split the dataset into input features (`X`) and target variable (`y`)
2. **Train-Test Split** — 80% training / 20% testing, with `random_state=42` and stratification for reproducibility
3. **Machine Learning Models** — define five classification algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest (200 estimators)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN, k=5)
4. **Train and Evaluate All Models** — fit each model and compute performance metrics
5. **Initial Result** — first-pass comparison across models
6. **Confusion Matrices** — visualize prediction outcomes for each model
7. **Classification Reports** — precision, recall, and F1 score per class
8. **K-Fold Cross-Validation** — 5-fold cross-validation to verify robustness of results
9. **Visualize Model Comparison** — charts comparing all models across metrics
10. **Conclusion** — final model selection and reasoning

## Results

| Metric | Logistic Regression (Best Model) |
|---|---|
| Accuracy | 80.55% |
| F1 Score | 0.6040 |
| ROC-AUC | 0.8420 |
| Precision | 0.6572 |
| Recall | 0.6040 |
| 5-Fold CV Mean Accuracy | 80.55% |
| 5-Fold CV Mean ROC-AUC | 0.8449 |

Five classification algorithms — Logistic Regression, Decision Tree, Random Forest, SVM, and KNN — were trained and evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Classification Report. **Logistic Regression** achieved the best overall performance and was selected as the model for this phase, offering a strong balance between predictive performance, consistency, and computational efficiency.

## Key Libraries Used

- `pandas`
- `numpy`
- `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `train_test_split`, cross-validation, metrics)
- `matplotlib`
- `seaborn`

## How to Run

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open `Model_Development_Notebook.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run the cells in order.

> **Note:** This notebook expects the final feature-engineered dataset (`X`, `y`) from the Week 3 Phase I feature engineering step as input.

## Author

Amna Ali
