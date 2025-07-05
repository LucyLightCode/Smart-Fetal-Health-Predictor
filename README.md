# Smart-Fetal-Health-Predictor

![pregnancy-immature-premature-labor-shown-fetus-placenta-top-uterus-umbilical-cord-uterus-bulging-60404409](https://github.com/user-attachments/assets/5de0ed1c-f6c4-487c-a908-8bc13db8a268)


Study Background

This project aims to develop an embedded AI-powered predictive model for fetal monitoring systems. The problem it addresses is that current fetal monitoring devices generate data requiring expert interpretation, leading to delays and inconsistent diagnoses, especially in underserved areas. The proposed solution is to integrate an AI model that analyzes real-time data to provide instant alerts for fetal health risks. The expected impact includes accelerating clinical response, empowering frontline staff, enabling remote monitoring, reducing fetal mortality, supporting timely medical decisions, and optimizing hospital resource allocation.


Problem Statement

"Current fetal monitoring devices generate raw cardiotocography (CTG) data that requires expert interpretation by trained OB/GYN specialists. In many clinical settings — especially in underserved or remote regions — this dependency leads to delays in diagnosis, inconsistent interpretation, and missed early warnings of fetal distress."


The aims and objectives

Aims:

* Develop and integrate an embedded AI-powered predictive model into fetal monitoring systems.

* Analyze real-time CTG data automatically to detect abnormal patterns and generate instant clinical alerts for fetal health risks.

Objectives (Expected Impact):

* Accelerate clinical response during emergencies.

* Empower frontline staff in areas with limited access to OB/GYN specialists.

* Optimize hospital resource allocation by triaging cases based on real-time risk detection.

Materials and Methods

Materials:

 * Dataset: The project uses the fetal_health.csv dataset.

 * Libraries: The project utilizes several Python libraries:
  - pandas for data manipulation and analysis.
  - seaborn and matplotlib.pyplot for data visualization.
  - numpy for numerical operations.
  - sklearn (Scikit-learn) for data preprocessing, model selection, and evaluation, including:
  * StandardScaler for feature scaling.
  * train_test_split for splitting data.
  * RandomForestClassifier,     LogisticRegression, SVC, XGBClassifier, DecisionTreeClassifier for modeling.
  * Classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, label_binarize for evaluation metrics.
  * GridSearchCV, cross_val_score, StratifiedKFold, Pipeline for model selection and evaluation.
* Joblib for saving and loading the trained model.
* Streamlit and Flask for building the web application interface.

Methods:

1. Data Loading: The fetal_health.csv dataset is loaded into a pandas DataFrame.

2. Data Preprocessing and EDA:

* The shape, info, null values, and duplicate values of the data are checked.
* Duplicate rows are removed.
* Descriptive statistics are generated.
* Distributions of features and the target variable are visualized using histograms, density plots, and bar plots.
* Boxplots are used to visualize the distribution of features across different fetal health categories.
* A correlation heatmap is generated to show relationships between features.
3. Feature Importance: A Random Forest model is fitted to the data to determine the importance of each feature. The top 10 most important features are identified.
4. Data Splitting: The data is split into training and testing sets using train_test_split, with stratification based on the target variable.
5. The fetal health categories into 3:- Normal (1), Suspect (2), Pathological (3)
6. Model Training and Evaluation: A function train_and_evaluate is defined to:
* Perform hyperparameter tuning using GridSearchCV with StratifiedKFold cross-validation.
* Train the model with the best parameters.
* Calculate and print training and testing accuracy.
* Generate and print a classification report.*Display a confusion matrix.
* Calculate and plot the ROC curve (One-vs-Rest) and report the macro average ROC AUC.
* This function is applied to the following models using only the top 10 features:

 * Logistic Regression
 * XGBoost (target labels are adjusted for XGBoost)
 * Random Forest
 * Support Vector Machine (features are scaled before training)
 * Decision Tree

7. Model Comparison: The training accuracy, test accuracy, and ROC AUC scores for all trained models are summarized in a table.
8. Model Validation and Selection: Based on the evaluation metrics, the Random Forest model is selected as the most balanced and robust model.
9. Model Saving: The selected Random Forest model is saved to a file named random_forest_model.pkl using joblib.
10. Deployment Interface (Streamlit): A Streamlit application is created to provide a user interface for predicting fetal health status using the saved Random Forest model and the top 10 features as input.

what are the Results : MSc students are required to have implemented 100% of their projects using online data
Based on the results presented in the notebook:

Key Findings and Model Performance:

* The project successfully trained and evaluated several classification models to predict fetal health status using the provided dataset.
* The models were evaluated using metrics such as training accuracy, test accuracy, and Macro Average ROC AUC (One-vs-Rest).
* A summary table shows the performance of each model:
 * XGBoost and Random Forest achieved the highest test accuracies (around 0.94) and Macro Average ROC AUC scores (around 0.98).
 * Support Vector Machine and Decision Tree also showed good performance, but slightly lower than XGBoost and Random Forest.
 * Logistic Regression had the lowest performance among the evaluated models.
The Random Forest model was selected as the most balanced and robust model, showing good generalization to the test set with a test accuracy of 0.943 and a Macro Average ROC AUC of 0.983. It performed well across all fetal health categories, including the less frequent "Pathological" class.
 * The project identified the top 10 most important features for prediction using a Random Forest model.

While the analysis and modeling steps themselves are independent of the data source being local or online, the data acquisition method would need to be adjusted to fulfill that specific requirement.


DOCUMENTRY
✅ A dual-mode AI app (Flask + Streamlit)

The picture below shows the result of the Streamlit app

![Screenshot (28)](https://github.com/user-attachments/assets/6db3c6fd-f47e-4c21-9237-6c091f23685b)


The picture below shows the result of the Flask app

![Screenshot (29)](https://github.com/user-attachments/assets/0148d2c5-bd91-49f4-b2a3-a23930bc07fb)



✅ Running securely inside Docker

✅ With a 1-click smart launcher

✅ That auto-kills ports & auto-starts Docker

✅ And it’s 🔥 production-ready!



##Conclution

This project successfully developed and evaluated several machine learning models for predicting fetal health status based on CTG data. The Random Forest model demonstrated the best overall performance, achieving high accuracy and strong generalization on unseen data, indicating its potential as a robust tool for real-time fetal health prediction. The identified key features and the deployed Streamlit interface highlight the feasibility of integrating such an AI-powered predictor into fetal monitoring systems to potentially improve clinical outcomes and resource allocation, especially in underserved regions.



