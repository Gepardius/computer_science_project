import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


"""
PROJECT: COMPUTER SCIENCE PROJECT
DLMCSPCSP

Customer Churn Prediction Using Machine Learning
Developed by: Gal Bordelius
"""

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Load Telco Customer Churn dataset
df_imp = df
df = df.drop(['customerID', 'TotalCharges'], axis=1)  # Drop unnecessary columns

df['gender'] = pd.get_dummies(df['gender'], drop_first=True)    # Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn'], axis=1),
                                                    df['Churn'], test_size=0.15, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection using SelectKBest
selector = SelectKBest(f_classif, k=20)
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# Evaluation metrics
print('\nLogistic Regression')
print('Accuracy:', accuracy_score(y_test, lr_pred))
print('F1 Score:', f1_score(y_test, lr_pred, pos_label='Yes'))
print('Precision:', precision_score(y_test, lr_pred, pos_label='Yes'))
print('Recall:', recall_score(y_test, lr_pred, pos_label='Yes'))

print('\nDecision Tree')
print('Accuracy:', accuracy_score(y_test, dt_pred))
print('F1 Score:', f1_score(y_test, dt_pred, pos_label='Yes'))
print('Precision:', precision_score(y_test, dt_pred, pos_label='Yes'))
print('Recall:', recall_score(y_test, dt_pred, pos_label='Yes'))

print('\nRandom Forest')
print('Accuracy:', accuracy_score(y_test, rf_pred))
print('F1 Score:', f1_score(y_test, rf_pred, pos_label='Yes'))
print('Precision:', precision_score(y_test, rf_pred, pos_label='Yes'))
print('Recall:', recall_score(y_test, rf_pred, pos_label='Yes'))

print('\nGradient Boosting')
print('Accuracy:', accuracy_score(y_test, gb_pred))
print('F1 Score:', f1_score(y_test, gb_pred, pos_label='Yes'))
print('Precision:', precision_score(y_test, gb_pred, pos_label='Yes'))
print('Recall:', recall_score(y_test, gb_pred, pos_label='Yes'))

# graphical presentation
barWidth = 0.2
lr_scores = [accuracy_score(y_test, lr_pred), f1_score(y_test, lr_pred, pos_label='Yes'),
             precision_score(y_test, lr_pred, pos_label='Yes'), recall_score(y_test, lr_pred, pos_label='Yes')]
dt_scores = [accuracy_score(y_test, dt_pred), f1_score(y_test, dt_pred, pos_label='Yes'),
             precision_score(y_test, dt_pred, pos_label='Yes'), recall_score(y_test, dt_pred, pos_label='Yes')]
rf_scores = [accuracy_score(y_test, rf_pred), f1_score(y_test, rf_pred, pos_label='Yes'),
             precision_score(y_test, rf_pred, pos_label='Yes'), recall_score(y_test, rf_pred, pos_label='Yes')]
gb_scores = [accuracy_score(y_test, gb_pred), f1_score(y_test, gb_pred, pos_label='Yes'),
             precision_score(y_test, gb_pred, pos_label='Yes'), recall_score(y_test, gb_pred, pos_label='Yes')]

r1 = np.arange(len(lr_scores))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Plot the bar chart
plt.bar(r1, lr_scores, color='#7f6d5f', width=barWidth, edgecolor='white', label='Logistic Regression')
plt.bar(r2, dt_scores, color='#557f2d', width=barWidth, edgecolor='white', label='Decision Tree')
plt.bar(r3, rf_scores, color='#2d7f5e', width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r4, gb_scores, color='#8a2be2', width=barWidth, edgecolor='white', label='Gradient Boosting')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(lr_scores))], ['Accuracy', 'F1 Score', 'Precision', 'Recall'])

plt.legend()    # show the legend
plt.show()  # show the plot

# Convert non-numeric columns to numeric
for col in df_imp.columns:
    if df_imp[col].dtype == 'object':
        le = LabelEncoder()
        df_imp[col] = le.fit_transform(df_imp[col])

# Split into features and target
X = df_imp.drop(['Churn'], axis=1)
y = df_imp['Churn']

# Train decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

# Train random forest model
rf = RandomForestClassifier(random_state=1)
rf.fit(X, y)

# Train gradient boosting model
gb = GradientBoostingClassifier(random_state=1)
gb.fit(X, y)

# Train logistic regression model
lr = LogisticRegression(random_state=1, max_iter=1000)
lr.fit(X, y)

# Decision tree feature importances
dt_importances = dt.feature_importances_
dt_feature_names = X.columns
dt_importance_df_imp = pd.DataFrame({'Feature': dt_feature_names, 'Importance': dt_importances})
dt_importance_df_imp = dt_importance_df_imp.sort_values(by='Importance', ascending=False)
print('\nDecision Tree Feature Importances')
print(dt_importance_df_imp.head(10))

# Random forest feature importances
rf_importances = rf.feature_importances_
rf_feature_names = X.columns
rf_importance_df_imp = pd.DataFrame({'Feature': rf_feature_names, 'Importance': rf_importances})
rf_importance_df_imp = rf_importance_df_imp.sort_values(by='Importance', ascending=False)
print('\nRandom Forest Feature Importances')
print(rf_importance_df_imp.head(10))

# Gradient boosting feature importances
gb_importances = gb.feature_importances_
gb_feature_names = X.columns
gb_importance_df_imp = pd.DataFrame({'Feature': gb_feature_names, 'Importance': gb_importances})
gb_importance_df_imp = gb_importance_df_imp.sort_values(by='Importance', ascending=False)
print('\nGradient Boosting Feature Importances')
print(gb_importance_df_imp.head(10))

# Logistic regression coefficients
lr_coefficients = np.abs(lr.coef_)[0]
lr_feature_names = X.columns
lr_coefficient_df_imp = pd.DataFrame({'Feature': lr_feature_names, 'Coefficient': lr_coefficients})
lr_coefficient_df_imp = lr_coefficient_df_imp.sort_values(by='Coefficient', ascending=False)
print('\nLogistic Regression Coefficients')
print(lr_coefficient_df_imp.head(10))

# Plot the graphs for the importances
# Plot top 10 decision tree feature importances
plt.figure(figsize=(10, 5))
plt.title('Decision Tree Feature Importances')
plt.bar(dt_importance_df_imp['Feature'][:10], dt_importance_df_imp['Importance'][:10])
plt.xticks(rotation=90)
plt.show()

# Plot top 10 random forest feature importances
plt.figure(figsize=(10, 5))
plt.title('Random Forest Feature Importances')
plt.bar(rf_importance_df_imp['Feature'][:10], rf_importance_df_imp['Importance'][:10])
plt.xticks(rotation=90)
plt.show()

# Plot top 10 gradient boosting feature importances
plt.figure(figsize=(10, 5))
plt.title('Gradient Boosting Feature Importances')
plt.bar(gb_importance_df_imp['Feature'][:10], gb_importance_df_imp['Importance'][:10])
plt.xticks(rotation=90)
plt.show()

# Plot top 10 Logistic Regression Feature Coefficients
plt.figure(figsize=(10, 5))
plt.title('Logistic Regression Feature Coefficients')
plt.bar(lr_coefficient_df_imp['Feature'][:10], lr_coefficient_df_imp['Coefficient'][:10])
plt.xticks(rotation=90)
plt.show()
