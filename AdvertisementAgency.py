import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def Advertise(Datapath):
    df = pd.read_csv(Datapath)

    print("Dataset sample is : ")
    print(df.head())

    print("Columns name : ")
    print(df.columns)

    df = df.drop(columns=['Unnamed: 0'])

    print(df.head())

    print("Missing values in each column : ", df.isnull().sum())

    print("Statistical Summary : ")
    print(df.describe())

    print("Correlation matrix : ")
    print(df.corr())

    plt.figure(figsize = (10,5))
    sns.heatmap(df.corr(),annot = True, cmap = 'coolwarm')
    plt.title("Marvellous correlation heatmap")
    plt.show()

    x = df[['TV','radio','newspaper']]
    y = df['sales']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = LinearRegression()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    MSE = metrics.mean_squared_error(y_test,y_pred)
    RMSE = np.sqrt(MSE)
    r2 = metrics.r2_score(y_test,y_pred)

    print("means squared error is : ",MSE)
    print("Root mean squared error is : ",RMSE)
    print("R square is : ",r2)

    print("Model Coefficient are : ")
    for col,coef in zip(x.columns,model.coef_):
        print(f"{col} : {coef}")

    print("Y intercept is : ",model.intercept_)

    plt.figure(figsize=(8,5))
    plt.scatter(y_test,y_pred,color ='blue')
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Advertisement Data")
    plt.grid(True)
    plt.show()

def main():
    Advertise('Advertising.csv')

if __name__ == "__main__":
    main()