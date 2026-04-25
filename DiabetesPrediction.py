import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

def DiabetesPrediction(Datapath):
    df = pd.read_csv(Datapath)

    print(df.head())
    print(df.columns)
    df.dropna(inplace=True)
    print("Describe : ",df.describe())
    print(df.shape)
    df.dropna(inplace= True)
    print("Missing values in each column : ", df.isnull().sum())

    print(df.shape)
    line = '=' * 50
    x = df.drop(columns=['Outcome'])
    y = df['Outcome']

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    model = LogisticRegression()
    x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size=0.2,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("accuracy of model is : ",accuracy * 100)
    print(line)
    print("Training Accuracy")
    print(model.score(x_train,y_train))
    print(line)
    print("Testing Accuracy")
    print(model.score(x_test,y_test))
    print(line)
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix is : ",cm)
    print(line)
    precision = precision_score(y_test,y_pred)
    print("Precision score is : ",precision)
    print(line)
    recall = recall_score(y_test,y_pred)
    print("Recall score is : ",recall)
    print(line)
    F1 = f1_score(y_test,y_pred)
    print("F1 score is : ",F1)
    print(line)
    print(line)

    model = DecisionTreeClassifier()
    x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size=0.2,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("accuracy of model is : ",accuracy * 100)
    print(line)
    print("Training Accuracy")
    print(model.score(x_train,y_train))
    print(line)
    print("Testing Accuracy")
    print(model.score(x_test,y_test))
    print(line)
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix is : ",cm)
    print(line)
    precision = precision_score(y_test,y_pred)
    print("Precision score is : ",precision)
    print(line)
    recall = recall_score(y_test,y_pred)
    print("Recall score is : ",recall)
    print(line)
    F1 = f1_score(y_test,y_pred)
    print("F1 score is : ",F1)

    figure()
    df['Outcome'].plot.hist().set_title("Survival report")
    plt.show()

    figure()
    df['Age'].plot.hist().set_title("Age report")
    plt.show()

    sns.countplot(x='Outcome', data=df)
    plt.title("Distribution of Diabetes Outcome")
    plt.show()

    sns.boxplot(data=df[['Glucose','BloodPressure','BMI']])
    plt.title("Boxplot of key Features")
    plt.show()

def main():
    DiabetesPrediction('diabetes.csv')

if __name__ == "__main__":
    main()