from sklearn.linear_model import SGDRegressor


class SGD(object):
    def __init__(self):
        super().__init__()

    def get_regressor(self):
        return SGDRegressor()

    def get_config(self):
        return {
            "param": 0,
        }
