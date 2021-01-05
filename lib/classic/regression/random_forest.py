from sklearn.ensemble import RandomForestRegressor


class RandomForest(object):
    """RandomForestRegressor wrapper that stores regularization parameters"""

    def __init__(self, n_estimators=100, criterion="mae"):
        self.n_estimators = n_estimators
        self.criterion = criterion
        super().__init__()

    def get_regressor(self):
        """Build the regressor"""
        return RandomForestRegressor(
            n_estimators=self.n_estimators, criterion=self.criterion
        )

    def get_config(self):
        """Return the parameters dict"""
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
        }
