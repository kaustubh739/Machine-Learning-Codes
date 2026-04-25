import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x': [12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,61,64,69,72],
    'y': [39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7,24]
})
Border = "-" * 50
print("Step 1 : Initialisation - K initial means (centroids) are geerated at random")

print(Border)
print("Dataset for training")
print(Border)
print(df)
print(Border)
np.random.seed(200)
k = 3
# centroids[i] = [x,y]

centroids = {
    i+1 : [np.random.randint(0,80),np.random.randint(0,80)]
    for i in range(k)
}
print(Border)
print("Random centroid generated")
print(centroids)
print(Border)

fig = plt.figure(figsize = (5,5))
plt.scatter(df['x'],df['y'],color ='k')

colmap = {1:'r',2:'g',3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color = colmap[i])

plt.title("Marvellous : Dataset with random centroid")

plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

# -----------------------------------------------------

# Assignment - K clusters are ctreated by associating each observation with the nearest cluster
def assignment(df,centroids):
    for i in centroids.keys():
        # sqrt((x1-x2))^2 - (y1-y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:,centroid_distance_cols].idxmin(axis = 1)
    df['closest'] = df['closest'].map(lambda x : int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

print("step 2 : Assignment - K clusters are created by associating each observation with the nearest centroid")

print("Before assignment dataset")
print(df)
df= assignment(df,centroids)

print("First centroid : Red")
print("Second centroid : Green")
print("Third centroid : Blue")

print("After assignment dataset")
print(df)

fig = plt.figure(figsize = (5,5))
plt.scatter(df['x'],df['y'],color = df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Marvellous : Dataset with clustering and random centroid")
plt.show()

# -------------------

old_centroids = copy.deepcopy(centroids)
print("Step 3 : Update - The centroid of the clusters becomes the new mean assignment and Update are repeated iteratively until convergence")

def update(k):
    print("Old values of centroids")
    print(k)

    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest']==i]['y'])

    print("New values of centroids")
    print(k)
    return k

centroids = update(centroids)

fig = plt.figure(figsize = (5,5))
ax = plt.axes()
plt.scatter(df['x'],df['y'],color = df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x,old_y,dx,dy,head_width=2,head_length =3,fc=colmap[i],ec=colmap[i])

plt.title("Marvellous : Dataset with clustering and updated centroids")
plt.show()

#--------------------

# Repeat Assignment Stage
print("Before assignment dataset")
print(df)
df = assignment(df,centroids)
print("After assignment dataset")
print(df)

# plot results
fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Marvelloius : Dataset with clustering and updated centroids")
plt.show()

# Continue until all assigned categories don't change any more 

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    print("Before assignment dataset")
    print(df)
    df = assignment(df,centroids)
    print("After assignment dataset")
    print(df)
    if closest_centroids.equals(df['closest']):
        break
print("Final values of centroids")
print(centroids)

fig = plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Marvellous : Final dataset with set centroids")
plt.show()
