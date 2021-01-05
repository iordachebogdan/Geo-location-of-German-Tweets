from sklearn import svm


class LinearSVR(object):
    """LinearSVR wrapper that stores regularization parameters"""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_regressor(self):
        """Build the regressor"""
        return svm.LinearSVR(C=self.c)

    def get_config(self):
        """Return the parameters dict"""
        return {
            "c": self.c,
        }
