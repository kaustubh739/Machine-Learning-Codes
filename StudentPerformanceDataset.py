import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def main():
    df = pd.read_csv('student-por.csv')

    print("First 5 records are : ",df.head())
    print("columns name : ", df.columns)
    print("Dataset shape : ",df.shape)

    # Select relevant features
    features = ['G1','G2','G3','studytime','failures','absences']
    x = df[features]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    WCSS = []

    for k in range(1,11):
        model = KMeans(n_clusters = k, init = 'k-means++',n_init=50,random_state=42)
        model.fit(x_scaled)
        WCSS.append(model.inertia_)

    plt.plot(range(1,11),WCSS,marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow method for optimal k")
    plt.show()

    model = KMeans(n_clusters=3,init='k-means++',n_init=50,random_state=42)
    y_kmeans = model.fit_predict(x_scaled)

    df['cluster'] = y_kmeans
    #print(y_kmeans)

    print(df.groupby('cluster')[features].mean())

    cluster_map = {0:'Top performers',1:'Average Students',2:'Struggling Students'}
    df['performanceGroup'] = df['cluster'].map(cluster_map)

    print(df[['G1','G2','G3','studytime','failures','absences','performanceGroup']].head())
    
if __name__ == "__main__":
    main()