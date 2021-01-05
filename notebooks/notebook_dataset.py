"""Copy of notebook cells into a py file"""

import pandas as pd

COLS = ["id", "lat", "long", "text"]
df_train = pd.read_csv("../data/training.txt", names=COLS)
df_val = pd.read_csv("../data/validation.txt", names=COLS)

df_train.head()

print(len([len(t) for t in df_train.text if len(t) <= 600]))

bounding_box = (
    df_train.long.min(),
    df_train.long.max(),
    df_train.lat.min(),
    df_train.lat.max(),
)
print(f"train: {bounding_box}")
print(
    f"validation: {(df_val.long.min(), df_val.long.max(), df_val.lat.min(), df_val.lat.max())}"
)

# plot the train and val tweets on the map of Germany

import matplotlib.pyplot as plt

gmap = plt.imread("../data/map.png")
fix, ax = plt.subplots(figsize=(15, 8))
ax.scatter(df_train.long, df_train.lat, zorder=1, c="r")
ax.scatter(df_val.long, df_val.lat, zorder=0.5, c="b")

ax.set_title("Tweets on German map")
ax.set_xlim(bounding_box[0], bounding_box[1])
ax.set_ylim(bounding_box[2], bounding_box[3])
ax.imshow(gmap, zorder=0, extent=bounding_box)

# KMeans classification test

from sklearn.cluster import KMeans
import numpy as np


def kmeans(points, num_clusters):
    km = KMeans(n_clusters=num_clusters, random_state=0).fit(points)
    return km


def get_clusters(kmeans, points, num_clusters):
    clusters = [[] for _ in range(num_clusters)]
    for point, label in zip(points, kmeans.labels_):
        clusters[label].append(point)
    return clusters


def predict_error(kmeans, coords):
    predicted = np.array(
        [kmeans.cluster_centers_[label] for label in kmeans.predict(coords)]
    )
    mae = np.abs(coords - predicted).mean(axis=0)
    assert mae.shape == (2,)
    return mae[0] + mae[1]


import math
from tqdm import tqdm


def dist(point1, point2):
    """Distance in km from lat, long coordinates"""
    earth_radius = 6373
    lat1, long1 = point1
    lat2, long2 = point2
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)

    delta_lat = lat2 - lat1
    delta_long = long2 - long1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_long / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def max_dist(clusters):
    # max dist between two points in the same cluster
    dmax = 0
    for cluster in tqdm(clusters):
        for first in cluster:
            for second in cluster:
                dmax = max(dmax, dist(first, second))
    return dmax


def cluster_stats(NUM_CLUSTERS):
    # statistics for KMeans clustering
    print(f"Trying for {NUM_CLUSTERS} clusters")
    coords = np.array(
        [
            [longitude, latitude]
            for longitude, latitude in zip(df_train.long, df_train.lat)
        ]
    )

    km = kmeans(coords, NUM_CLUSTERS)
    clusters = get_clusters(km, coords, NUM_CLUSTERS)

    print(f"MAXDIST: {max_dist(clusters)}")

    # print the coordinates on the map, different colors for different classes
    map = plt.imread("../data/map.png")
    fix, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(gmap, zorder=0, extent=bounding_box)
    ax.set_title(f"{NUM_CLUSTERS}-Clustered tweets on German map")
    ax.set_xlim(bounding_box[0], bounding_box[1])
    ax.set_ylim(bounding_box[2], bounding_box[3])

    cmap = plt.cm.get_cmap("hsv", NUM_CLUSTERS)
    for i, cluster in enumerate(clusters):
        ax.scatter(
            [first for first, _ in cluster],
            [second for _, second in cluster],
            zorder=1,
            color=cmap(i),
        )

    print(f"Cluster sizes: {[len(cluster) for cluster in clusters]}")
    print(f"Min cluster: {min([len(cluster) for cluster in clusters])}")
    mae = predict_error(
        km,
        np.array(
            [
                [longitude, latitude]
                for longitude, latitude in zip(df_val.long, df_val.lat)
            ]
        ),
    )

    print(f"Prediction error for validation: {mae}")


for i in [10, 15, 20, 25, 30]:
    cluster_stats(i)

# German cities data

cities_df = pd.read_csv("../data/de.csv")
cities_df.dropna(subset=["population"], inplace=True)
cities_df.tail()

# classification on cities tests

from sklearn.neighbors import KDTree
from collections import defaultdict
import numpy as np


def city_classes(df, df_test, cities_df, thresh=0):
    cities_df = cities_df[cities_df.population >= thresh]
    cities = np.array(cities_df.city)
    city_coords = np.array(
        [[lat, lng] for lat, lng in zip(cities_df.lat, cities_df.lng)]
    )

    kdtree = KDTree(city_coords)

    classes = defaultdict(list)
    for _, row in df.iterrows():
        _, ind = kdtree.query(np.array([[row.lat, row.long]]), k=1)
        classes[ind[0][0]].append([row.lat, row.long])

    city_coords = np.array([city_coords[i] for i in classes.keys()])
    kdtree = KDTree(city_coords)
    err = 0
    for _, row in df_test.iterrows():
        _, ind = kdtree.query(np.array([[row.lat, row.long]]), k=1)
        err += abs(city_coords[ind[0][0]][0] - row.lat) + abs(
            city_coords[ind[0][0]][1] - row.long
        )

    print(err / (2 * len(df_test)))
    return classes


classes = city_classes(df_train, df_val, cities_df, thresh=500000)
print(len(classes))

# plot cities classes on the map

map = plt.imread("../data/map.png")
fix, ax = plt.subplots(figsize=(15, 8))
ax.imshow(gmap, zorder=0, extent=bounding_box)
ax.set_title(f"{len(classes)} city classes on German map")
ax.set_xlim(bounding_box[0], bounding_box[1])
ax.set_ylim(bounding_box[2], bounding_box[3])

cmap = plt.cm.get_cmap("hsv", len(classes))
for i, cluster in enumerate(classes.values()):
    ax.scatter(
        [first for _, first in cluster],
        [second for second, _ in cluster],
        zorder=1,
        color=cmap(i),
    )
