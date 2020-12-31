from sklearn.ensemble import RandomForestRegressor


class RandomForest(object):
    def __init__(self, n_estimators=100, criterion="mae"):
        self.n_estimators = n_estimators
        self.criterion = criterion
        super().__init__()

    def get_regressor(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, criterion=self.criterion
        )

    def get_config(self):
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
        }
