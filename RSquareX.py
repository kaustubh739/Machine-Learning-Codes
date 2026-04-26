from sklearn.metrics import r2_score

def main():
    y_actual = [100,200,300,400,500]

    y_pred = [150,150,350,350,550]
# model degrade
    r2 = r2_score(y_actual,y_pred)

    print("Actual values : ",y_actual)
    print("Predicted Values : ",y_pred)
    print("R2 value is : ",r2)
    
if __name__ == "__main__":
    main()