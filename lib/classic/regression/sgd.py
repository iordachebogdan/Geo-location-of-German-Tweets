from sklearn.linear_model import SGDRegressor


class SGD(object):
    """SGDRegressor wrapper that stores regularization parameters"""

    def __init__(self):
        super().__init__()

    def get_regressor(self):
        """Build the regressor"""
        return SGDRegressor()

    def get_config(self):
        """Return the parameters dict"""
        return {
            "param": 0,
        }
