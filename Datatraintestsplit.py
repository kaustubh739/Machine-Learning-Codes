import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("iris.csv")
    print("Dataset loaded successfully")
    

    print(df["variety"])
    

    df["variety"] = df["variety"].map({'Setosa' : 0, 'Versicolor' : 1, 'Virginica' : 2})
    

    X = df.drop("variety", axis = "columns")
    Y = df["variety"]

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

    print("Dimension of X_train : ",X_train.shape)
    print("Dimension of X_test : ",X_test.shape)
    print("Dimension of Y_train : ",Y_train.shape)
    print("Dimension of Y_test : ",Y_test.shape)

if __name__ == "__main__":
    main()
    