import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_auc_score,roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def BankMarketting(Datapath):
    df = pd.read_csv(Datapath, sep=';')

    print(df.head())
    print(df.columns)
    print(df.shape)

    df.replace('unknown',np.nan, inplace=True)
    df.dropna(inplace=True)
    print(df.shape)

    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])
    #df = pd.get_dummies(df, drop_first=True)
    
    print("after encoding : ")
    print(df.head())

    x = df.drop(columns=["y"])
    #y = df['y'].map({'no':0, 'yes':1})
    y = df["y"]

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train,x_test,y_train,y_test = train_test_split(x_scale,y,test_size=0.2,random_state=42)

    models  = {

        "LogisticRegression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    Results = []
    for name,model in models.items():
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)
        report = classification_report(y_test,y_pred)
        roc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])

        Results.append([name,accuracy,roc])

        print(f"\n{name}")
        print("Accuracy", accuracy)
        print("Confusion matrix", cm)
        print("ROC-AUC :", roc)
        print("Classification Report :\n", report)

        print(df.describe())

        sns.heatmap(cm,annot=True, fmt='d',cmap='Blues')
        plt.title(f"Confusion Matrix")
        plt.show()

    fpr,tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    plt.plot(fpr,tpr, label=name)

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    #plt.legend(loc="lower right")
    plt.show()

    # Comparison Table

    results_df = pd.DataFrame(Results, columns=["Model","Accuracy","ROC-AUC"])
    print("\nModel Comparison: \n",results_df)
def main():
    BankMarketting('bank.csv')

if __name__ == "__main__":
    main()