#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Load the dataset
file_path = '/mnt/data/heart.csv'  # Adjust the path as necessary
heart_df = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
print(heart_df.head())

# Check for missing values
missing_values = heart_df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Check for duplicate rows
duplicate_rows = heart_df.duplicated().sum()
print("\nDuplicate Rows:\n", duplicate_rows)

# Basic data information
data_info = heart_df.info()
print("\nData Info:\n", data_info)

# Summary statistics
summary_stats = heart_df.describe()
print("\nSummary Statistics:\n", summary_stats)


# In[ ]:


# Compute basic statistics
basic_stats = heart_df.describe()
print("Basic Statistics:\n", basic_stats)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical columns
heart_df.hist(figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Box plots to check for outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=heart_df)
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=90)
plt.show()

# Pair plot to visualize relationships
sns.pairplot(heart_df)
plt.suptitle('Pair Plots of Features', y=1.02)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = '/mnt/data/heart.csv'  # Adjust the path as necessary
heart_df = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
print(heart_df.head())

# Check for missing values
missing_values = heart_df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Drop rows with missing values
heart_df = heart_df.dropna()

# Verify that there are no more missing values
print("\nMissing Values After Dropping Rows with Missing Data:\n", heart_df.isnull().sum())

# Check for duplicate rows
duplicate_rows = heart_df.duplicated().sum()
print("\nDuplicate Rows:\n", duplicate_rows)

# Remove duplicate rows
heart_df = heart_df.drop_duplicates()

# Verify that there are no more duplicate rows
print("\nDuplicate Rows After Dropping Duplicates:\n", heart_df.duplicated().sum())

# Convert categorical variables if necessary
if 'sex' in heart_df.columns:
    heart_df['sex'] = heart_df['sex'].map({'male': 1, 'female': 0})

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
numerical_columns = heart_df.select_dtypes(include=[np.number]).columns

heart_df[numerical_columns] = scaler.fit_transform(heart_df[numerical_columns])

# Display the first few rows of the cleaned and preprocessed dataset
print("\nCleaned and Preprocessed Data:\n", heart_df.head())

# Step 1: Compute Basic Statistics
basic_stats = heart_df.describe()
print("Basic Statistics:\n", basic_stats)

# Step 2: Visualize Distributions
# Histograms for numerical columns
heart_df.hist(figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Box plots to check for outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=heart_df)
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=90)
plt.show()

# Pair plot to visualize relationships
sns.pairplot(heart_df)
plt.suptitle('Pair Plots of Features', y=1.02)
plt.show()


# In[ ]:


# Correlation matrix
correlation_matrix = heart_df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a plot
plt.plot(x, y)

# Add title and labels
plt.title('Simple Line Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Show the plot
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Make sure the path is correct
file_path = 'path_to_your_file/heart.csv'
heart_df = pd.read_csv('heart.csv')

# Check the first few rows of the DataFrame
print(heart_df.head())

# Check for missing values
print(heart_df[['age', 'thalachh']].isnull().sum())

# Check data types
print(heart_df.dtypes)

# Example 2: Age vs. Maximum Heart Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(data=heart_df, x='age', y='thalachh')
plt.title('Age vs. Maximum Heart Rate')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load data into heart_df from a CSV file
heart_df = pd.read_csv('heart.csv')  # Replace 'heart_data.csv' with your actual file path

# Now you can perform operations on heart_df

import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset
file_path = '/mnt/data/heart.csv'  # Adjust the path as necessary
heart_df = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
print(heart_df.head())

# Check for missing values
missing_values = heart_df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Drop rows with missing values
heart_df = heart_df.dropna()

# Verify that there are no more missing values
print("\nMissing Values After Dropping Rows with Missing Data:\n", heart_df.isnull().sum())

# Check for duplicate rows
duplicate_rows = heart_df.duplicated().sum()
print("\nDuplicate Rows:\n", duplicate_rows)

# Remove duplicate rows
heart_df = heart_df.drop_duplicates()

# Verify that there are no more duplicate rows
print("\nDuplicate Rows After Dropping Duplicates:\n", heart_df.duplicated().sum())

# Convert categorical variables if necessary
if 'sex' in heart_df.columns:
    heart_df['sex'] = heart_df['sex'].map({'male': 1, 'female': 0})

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
numerical_columns = heart_df.select_dtypes(include=[np.number]).columns

heart_df[numerical_columns] = scaler.fit_transform(heart_df[numerical_columns])

# Display the first few rows of the cleaned and preprocessed dataset
print("\nCleaned and Preprocessed Data:\n", heart_df.head())

# Step 1: Correlation Analysis
# Correlation matrix
correlation_matrix = heart_df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Step 2: Explore Relationships
# Example 1: Age vs. Cholesterol
plt.figure(figsize=(10, 6))
sns.scatterplot(data=heart_df, x='age', y='chol')
plt.title('Age vs. Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# Example 2: Age vs. Maximum Heart Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(data=heart_df, x='age', y='thalachh')
plt.title('Age vs. Maximum Heart Rate')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate')
plt.show()

# Example 3: Sex vs. Heart Disease
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=heart_df, x='sex', hue='output')
plt.title('Sex vs. Heart Disease')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.legend(title='Heart Disease (0 = No, 1 = Yes)')
plt.show()



# Example 4: Chest Pain Type vs. Heart Disease
plt.figure(figsize=(10, 6))
sns.countplot(data=heart_df, x='cp', hue='output')
plt.title('Chest Pain Type vs. Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(title='Heart Disease (0 = No, 1 = Yes)')
plt.show()

# Step 3: Advanced Visualizations
# Pair plot for a subset of features
subset_features = ['age', 'chol', 'thalachh', 'trestbps', 'output']
sns.pairplot(heart_df[subset_features], hue='output')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()


# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
heart_df = pd.read_csv('heart.csv')  # Replace 'heart.csv' with the actual file path

# Exploratory Data Analysis (EDA)
# Display basic information about the dataset
print(heart_df.info())

# Summary statistics
print(heart_df.describe())

# Correlation matrix
correlation_matrix = heart_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for visualization of relationships
sns.pairplot(heart_df, hue='output', diag_kind='kde')
plt.show()

# Multivariate Analysis
# Split the dataset into features and target variable
X = heart_df.drop('output', axis=1)
y = heart_df['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predictive Modeling
# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = logistic_model.predict(X_test_scaled)
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test_scaled)
print("\nRandom Forest Classifier Performance:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Model Tuning (Random Forest Classifier)
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)
print("\nBest Random Forest Classifier Performance:")
print(classification_report(y_test, y_pred_best_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_best_rf))

# Cross-validation
cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))

# Feature Importance
feature_importance = best_rf_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# In[ ]:


# Conclusion and Interpretation

print("Conclusion and Interpretation:")
print("--------------------------------")

# Summarize key findings and insights
print("1. Key Findings and Insights:")
print("- Exploratory Data Analysis (EDA) revealed significant differences in features between individuals with and without heart disease.")
print("- Multivariate Analysis identified correlations between certain features and the presence of heart disease.")
print("- Predictive Modeling using logistic regression and random forest classifier achieved reasonable accuracy in predicting heart disease.")
print("- Feature Importance analysis highlighted the significance of age, chest pain type, maximum heart rate, and ST depression in predicting heart disease.")

# Reflection on feature significance
print("\n2. Reflection on Feature Significance:")
print("- Age: Older age is a known risk factor for heart disease.")
print("- Chest Pain Type: Different types of chest pain may indicate different underlying causes of heart disease.")
print("- Maximum Heart Rate: Higher maximum heart rates may suggest poorer heart health.")
print("- ST Depression: Abnormalities in ST segments during exercise can indicate reduced blood flow to the heart.")

# Suggest potential further steps or analyses
print("\n3. Potential Further Steps or Analyses:")
print("- Feature Engineering: Explore additional derived features or transformations.")
print("- Ensemble Methods: Investigate the potential of ensemble methods to improve predictive accuracy.")
print("- Model Interpretability: Utilize techniques such as SHAP values or partial dependence plots for deeper insights.")
print("- Longitudinal Analysis: Analyze longitudinal data to study changes in risk factors over time.")
print("- External Validation: Validate the models using external datasets to assess generalizability.")

# Additional comments or recommendations
print("\n4. Additional Comments or Recommendations:")
print("- Continuously refine predictive models and deepen understanding of risk factors for improved cardiovascular health strategies.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




