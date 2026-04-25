import pandas as pd

def main():
    df = pd.read_csv("iris.csv")
    print("Dataset loaded successfully")
    print("Dimension of dataset is : ",df.shape)

    print(df["variety"])
    #print(df["sepal.length"].head())

    df["variety"] = df["variety"].map({'Setosa' : 0, 'Versicolor' : 1, 'Virginica' : 2})
    print(df["variety"])

    x = df.drop("variety", axis = "columns")
    y = df["variety"]

    print("Independent variable dimension : ",x.shape)
    print("Dependent variable dimension : ",y.shape)

if __name__ == "__main__":
    main()