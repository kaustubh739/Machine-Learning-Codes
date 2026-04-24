import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt

line= '-' * 50
print(line)
def Run_DecisionTreeClassifier(X_train,X_test,Y_train,Y_test,X):
        model1 = DecisionTreeClassifier(max_depth=8)

        model1.fit(X_train,Y_train)
        y_pred = model1.predict(X_test)

        accuracy = accuracy_score(y_pred,Y_test)
        print("accuracy is : ",accuracy*100)

        Conf = confusion_matrix(y_pred,Y_test)
        print("Confusion matrix is : ",Conf)

        y_prob = model1.predict_proba(X_test)[:,1]
        ROC = roc_auc_score(Y_test,y_prob)
        print("ROC curve values are :", ROC)

        importance = pd.Series(model1.feature_importances_,index=X.columns)
        importance = importance.sort_values(ascending=False)

        importance.plot(kind='bar',figsize=(10,6),title="Feture Importance of DecisionTree")
        plt.show()

        print(line)

        print("Moving to RandomForest Algoritm")
        
        print(line)

def Run_RandomForestClassifier(X_train,X_test,Y_train,Y_test,X):
    model2 = RandomForestClassifier(n_estimators=150,max_depth=7, random_state=42)

    model2.fit(X_train,Y_train)
    y_pred = model2.predict(X_test)

    accuracy = accuracy_score(y_pred,Y_test)
    print("accuracy is : ",accuracy*100)

    Conf = confusion_matrix(y_pred,Y_test)
    print("Confusion matrix is : ",Conf)

    y_prob = model2.predict_proba(X_test)[:,1]
    ROC = roc_auc_score(Y_test,y_prob)
    print("ROC curve values are :", ROC)

    importance = pd.Series(model2.feature_importances_,index=X.columns)
    importance = importance.sort_values(ascending=False)

    importance.plot(kind='bar',figsize=(10,6),title="Feture Importance of RandomForest")
    plt.show()

def BreastCancerPrediction(Datapath):
    df = pd.read_csv(Datapath)

    print ("Dataset Columns name : ",df.columns)

    df.replace('?', np.nan, inplace=True)
    
    print("First 5 records : ",df.head())

    print("Shape of the dataset : ", df.shape)

    df.dropna(inplace=True)

    df.drop(columns=['CodeNumber'], inplace=True)

    print(df.shape)

    X = df.drop(columns=['CancerType'])
    Y = df['CancerType']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    Run_DecisionTreeClassifier(X_train,X_test,Y_train,Y_test,X)
    Run_RandomForestClassifier(X_train,X_test,Y_train,Y_test,X)

def main():
    BreastCancerPrediction('breast-cancer-wisconsin.csv')

if __name__ == "__main__":
    main ()