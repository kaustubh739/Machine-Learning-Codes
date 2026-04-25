import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt

# Loading datasets
fake = pd.read_csv('fake.csv')
true = pd.read_csv('True.csv')

# Adding labels
fake['label'] = 0
true['label'] = 1

# Combine
df= pd.concat([fake,true],axis=0)

df = df[['text','label']]

# Drop missing values
df.dropna(inplace=True)

print(df.head())
print(df.columns)
print(df.shape)

tfidf = TfidfVectorizer(stop_words='english',max_features=5000)
x = tfidf.fit_transform(df['text'])
y = df['label']

print(df.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Individual models
log_clf = LogisticRegression(max_iter=1000)
dt_clf = DecisionTreeClassifier(max_depth=20, random_state=42)

# Voting Classifier
voting_hard = VotingClassifier(estimators=[('lr',log_clf),('dt',dt_clf)],voting='hard')
voting_soft = VotingClassifier(estimators=[('lr',log_clf),('dt',dt_clf)],voting='soft')

# Logistic Regression
log_clf.fit(x_train,y_train)
y_pred_lr = log_clf.predict(x_test)
print("Logistic Regression Accuracy : ",accuracy_score(y_test,y_pred_lr)*100)

# Decision Tree
dt_clf.fit(x_train,y_train)
y_pred_dt = dt_clf.predict(x_test)
print("DecisionTree Accuracy is : ",accuracy_score(y_test,y_pred_dt)*100)

#Hard Voting
voting_hard.fit(x_train,y_train)
y_pred_hard = voting_hard.predict(x_test)
print("hard voting accuracy is : ",accuracy_score(y_test,y_pred_hard)*100)

#Soft Voting
voting_soft.fit(x_train,y_train)
y_pred_soft = voting_soft.predict(x_test)
print("Soft voting accuracy is : ",accuracy_score(y_test,y_pred_soft)*100)

#Confusion Matrix
print("Confusion Matrix (soft voting): ",confusion_matrix(y_test,y_pred_soft))

# ROC-AUC (using probabilities for soft voting)
y_prob_soft = voting_soft.predict_proba(x_test)[:,1]
print("soft voting ROC-AUS is : ",roc_auc_score(y_test,y_prob_soft))

importance = pd.Series(dt_clf.feature_importances_,index = x.columns)
importance = importance.sort_values(ascending=False)

importance.plot(kind='bar',figsize=(10,6), title='Feature importance for Decision Tree Algorithm')
plt.show()

models = ['Logistic Regression','Decision Tree','Hard Voting', 'Soft Voting']

accuracies = [
    accuracy_score(y_test,y_pred_lr),
    accuracy_score(y_test,y_pred_dt),
    accuracy_score(y_test,y_pred_hard),
    accuracy_score(y_test,y_pred_soft)
]

plt.bar(models,accuracies,color=['blue','green','orange','red'])
plt.xlabel("Model Names")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.show()