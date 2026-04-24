from sklearn import tree

# rough 0
# smooth 1
def main():
    BallFeatures = [[35,0],[47,0],[90,1],[48,0],[90,1],[35,0],[92,1],[35,0],[35,0],[35,0]] # feature encoding
    BallNames = ["tennis","tennis","cricket","tennis","cricket","tennis","cricket","tennis","tennis","tennis"] # label encoding

    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(BallFeatures,BallNames)

    print(obj.predict([93,1]))

if __name__ == "__main__":
    main()