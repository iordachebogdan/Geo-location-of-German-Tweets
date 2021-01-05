from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


class ClassificationOnCities(object):
    """Logic to bundle tweets into classes based on the closest city"""

    def __init__(self, df_train, threshold=0):
        """Build the classes, ignoring cities with a population lower than threshold"""
        super().__init__()
        # load cities data
        cities_df = pd.read_csv("data/de.csv")
        cities_df.dropna(subset=["population"], inplace=True)

        # filter cities
        cities_df = cities_df[cities_df.population >= threshold]
        cities = np.array(cities_df.city)
        city_coords = np.array(
            [[lat, lng] for lat, lng in zip(cities_df.lat, cities_df.lng)]
        )

        # build a KDTree based on city coordinates
        kdtree = KDTree(city_coords)

        # find for each training sample the closest city
        classes = set({})
        for _, row in df_train.iterrows():
            _, ind = kdtree.query(np.array([[row.lat, row.long]]), k=1)
            classes.add(ind[0][0])
        classes = list(classes)

        self.cities = [cities[i] for i in classes]
        self.city_coords = [city_coords[i] for i in classes]
        # store selected cities coordinates inside a kdtree
        self.kdtree = KDTree(np.array(self.city_coords))

        print(f"Using {len(self.cities)} classes")

    def set_true_class(self, df):
        # use the generated city kdtree in order to assign cities to data samples
        # based on coordinates
        true_classes = []
        for _, row in df.iterrows():
            _, ind = self.kdtree.query(np.array([[row.lat, row.long]]), k=1)
            true_classes.append(ind[0][0])
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        # given the predicted class, write the coordinates of the corresponding city
        df["predict_lat"] = [self.city_coords[i][0] for i in df["predict_class"]]
        df["predict_long"] = [self.city_coords[i][1] for i in df["predict_class"]]


class ClassificationOnRegions(object):
    """Logic to bundle tweets into classes based on the containing administrative region"""

    def __init__(self, df_train):
        super().__init__()
        cities_df = pd.read_csv("data/de.csv")
        cities_df.dropna(subset=["population"], inplace=True)

        regions = np.array(cities_df.admin_name)
        city_coords = np.array(
            [[lat, lng] for lat, lng in zip(cities_df.lat, cities_df.lng)]
        )

        # find for each training tweet the closest city and assign its region to it
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
        # for each tweet, find the closest city and then set the golden label as
        # that city's region
        # try for multiple cities if the first ones have a region that is not
        # part of the generated classes
        true_classes = []
        for _, row in df.iterrows():
            _, ind = self.kdtree.query(np.array([[row.lat, row.long]]), k=3)
            i = 0
            while self.city_regions[ind[0][i]] not in self.regions:
                i += 1
            true_classes.append(self.region_to_idx[self.city_regions[ind[0][i]]])
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        # set predicted coords based on the predicted class
        df["predict_lat"] = [
            self.regions_mean_coords[i][0] for i in df["predict_class"]
        ]
        df["predict_long"] = [
            self.regions_mean_coords[i][1] for i in df["predict_class"]
        ]


class ClassificationOnKMeans(object):
    """Build classes based on KMeans clusters obtained on coordinates"""

    def __init__(self, df_train, num_clusters):
        super().__init__()

        # build KMeans
        coords = np.array([[lat, lng] for lat, lng in zip(df_train.lat, df_train.long)])
        self.km = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)

    def set_true_class(self, df):
        # use predict on the KMeans clusters in order to set golden labels
        true_classes = []
        for _, row in df.iterrows():
            label = self.km.predict(np.array([[row.lat, row.long]]))[0]
            true_classes.append(label)
        df["true_class"] = true_classes

    def set_predicted_coords(self, df):
        # set predicted coords as the centoroid coordinates of the predicted class
        df["predict_lat"] = [
            self.km.cluster_centers_[i][0] for i in df["predict_class"]
        ]
        df["predict_long"] = [
            self.km.cluster_centers_[i][1] for i in df["predict_class"]
        ]


def class_labels_to_onehot(labels, num_classes):
    # convert list of class labels into one-hot representation
    one_hot = np.eye(num_classes, dtype="int64")
    onehot_labels = []
    for label in labels:
        label = int(label)
        onehot_labels.append(one_hot[label])
    return np.asarray(onehot_labels)
