from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


class ClassificationOnCities(object):
    def __init__(self, df_train, threshold=0):
        super().__init__()
        cities_df = pd.read_csv("data/de.csv")
        cities_df.dropna(subset=["population"], inplace=True)

        cities_df = cities_df[cities_df.population >= threshold]
        cities = np.array(cities_df.city)
        city_coords = np.array(
            [[lat, lng] for lat, lng in zip(cities_df.lat, cities_df.lng)]
        )

        kdtree = KDTree(city_coords)

        classes = set({})
        for _, row in df_train.iterrows():
            _, ind = kdtree.query(np.array([[row.lat, row.long]]), k=1)
            classes.add(ind[0][0])
        classes = list(classes)

        self.cities = [cities[i] for i in classes]
        self.city_coords = [city_coords[i] for i in classes]
        self.kdtree = KDTree(np.array(self.city_coords))

        print(f"Using {len(self.cities)} classes")

    def set_true_class(self, df):
        true_classes = []
        for _, row in df.iterrows():
            _, ind = self.kdtree.query(np.array([[row.lat, row.long]]), k=1)
            true_classes.append(ind[0][0])
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        df["predict_lat"] = [self.city_coords[i][0] for i in df["predict_class"]]
        df["predict_long"] = [self.city_coords[i][1] for i in df["predict_class"]]


class ClassificationOnRegions(object):
    def __init__(self, df_train):
        super().__init__()
        cities_df = pd.read_csv("data/de.csv")
        cities_df.dropna(subset=["population"], inplace=True)

        regions = np.array(cities_df.admin_name)
        city_coords = np.array(
            [[lat, lng] for lat, lng in zip(cities_df.lat, cities_df.lng)]
        )

        kdtree = KDTree(city_coords)

        classes = defaultdict(list)
        for _, row in df_train.iterrows():
            _, ind = kdtree.query(np.array([[row.lat, row.long]]), k=1)
            classes[regions[ind[0][0]]].append([row.lat, row.long])

        self.regions = list(classes.keys())
        self.regions_mean_coords = np.array(
            [
                [np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])]
                for coords in classes.values()
            ]
        )
        self.region_to_idx = {r: i for i, r in enumerate(classes.keys())}
        self.city_regions = regions
        self.kdtree = kdtree

        print(f"Using {len(self.regions)} classes")

    def set_true_class(self, df):
        true_classes = []
        for _, row in df.iterrows():
            _, ind = self.kdtree.query(np.array([[row.lat, row.long]]), k=3)
            i = 0
            while self.city_regions[ind[0][i]] not in self.regions:
                i += 1
            true_classes.append(self.region_to_idx[self.city_regions[ind[0][i]]])
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        df["predict_lat"] = [
            self.regions_mean_coords[i][0] for i in df["predict_class"]
        ]
        df["predict_long"] = [
            self.regions_mean_coords[i][1] for i in df["predict_class"]
        ]


class ClassificationOnKMeans(object):
    def __init__(self, df_train, num_clusters):
        super().__init__()

        coords = np.array([[lat, lng] for lat, lng in zip(df_train.lat, df_train.long)])
        self.km = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)

    def set_true_class(self, df):
        true_classes = []
        for _, row in df.iterrows():
            label = self.km.predict(np.array([[row.lat, row.long]]))[0]
            true_classes.append(label)
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        df["predict_lat"] = [
            self.km.cluster_centers_[i][0] for i in df["predict_class"]
        ]
        df["predict_long"] = [
            self.km.cluster_centers_[i][1] for i in df["predict_class"]
        ]
