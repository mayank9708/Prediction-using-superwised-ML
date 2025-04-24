📊 Student Score Prediction using Supervised Machine Learning
This project demonstrates how supervised machine learning techniques can be applied to predict students' scores based on their study hours using regression and classification methods. It also evaluates student performance using the K-Nearest Neighbors (KNN) algorithm.

🔍 Dataset
We use a publicly available dataset:

Source: Student Scores Dataset

Attributes:

Hours: Number of study hours

Scores: Marks scored out of 100

🧠 Techniques Used
1. Exploratory Data Analysis (EDA)
Checked for missing/null values

Data types validation

Descriptive statistics

Outlier detection using Z-score

Correlation analysis between Hours and Scores

Data visualization using matplotlib and seaborn

2. Regression Analysis
Manually calculated linear regression line (using B0 and B1)

Evaluated prediction error using RMSE

Used statsmodels to generate regression line for comparison

3. Classification using KNN
Created a binary classification target (Result) based on a cut-off score (40)

Trained a KNN classifier to classify students as pass (True) or fail (False)

Accuracy evaluated on both train and test sets

📈 Libraries Used

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as smf


🧪 Model Evaluation
Regression RMSE: ~0.104 (Normalized Error)

KNN Accuracy:

Train Accuracy: ~94.4%

Test Accuracy: ~85.7%

 >> Predictions

knn.predict([[9.25]])   # Output: True
knn.predict([[14]])     # Output: True
knn.predict([[3]])      # Output: False


📊 Visualizations
Distribution plots for scores

Regression line plotted against actual values

Countplot showing classification results
