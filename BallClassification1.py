from sklearn import tree

def main():
    BallFeatures = [[35,"rough"],[47,"rough"],[90,"Smooth"],[48,"rough"],[90,"Smooth"],[35,"rough"],[92,"Smooth"],[35,"rough"],[35,"rough"],[35,"rough"]]
    BallNames = ["tennis","tennis","cricket","tennis","cricket","tennis","cricket","tennis","tennis","tennis"]

    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(BallFeatures,BallNames)

    print(obj.predict([93,"smooth"]))

if __name__ == "__main__":
    main()