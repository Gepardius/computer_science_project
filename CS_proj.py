import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
