from sklearn import svm


class LinearSVR(object):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_regressor(self):
        return svm.LinearSVR(C=self.c)

    def get_config(self):
        return {
            "c": self.c,
        }
