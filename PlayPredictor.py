import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
def playpredictor(Datapath):
    df = pd.read_csv(Datapath)

    print("Dataset loaded sucessfully")

    print(df.head())

    print("dimension of dataset are : ",df.shape)

    print(df.columns)

    df.drop(columns=['Unnamed: 0'],inplace=True)

    print("Dimension of dataset is :",df.shape)

    print(df.head())

    BallFeatures = [[1,1],[1,1],[2,1],[3,2],[3,3],[3,3],[2,3],[1,2],[1,3],[3,2]]
    BallNames = [0,0,1,1,1,0,1,0,1,1]
    
    k = 10
    obj = KNeighborsClassifier(k)

    obj = obj.fit(BallFeatures,BallNames)

    print(obj.predict([[3,1],[1,2],[2,2],[2,3]]))

    df.dropna(inplace= True)

    encoder = LabelEncoder()

    df['Whether'] = encoder.fit_transform(df['Whether'])
    df['Temperature'] = encoder.fit_transform(df['Temperature'])
    df['Play']=encoder.fit_transform(df['Play'])

    print("after encoding")
    print(df.head())

    x = df.drop(columns = ['Play'])
    y = df['Play']

    print("total records in dataset : ",x.shape)
    print("total records in dataset : ",y.shape)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=42)

    model = KNeighborsClassifier(k)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    
    cm = confusion_matrix(y_test,y_pred)
    print(cm)

    print(df.describe())

def main():
    playpredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()