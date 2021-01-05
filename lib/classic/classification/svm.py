from sklearn import svm


class SVM(object):
    """SVM wrapper that stores regularization parameters"""

    def __init__(self, c, kernel):
        super().__init__()
        self.c = c
        self.kernel = kernel

    def get_classifier(self):
        """Build the classifier"""
        return svm.SVC(C=self.c, kernel=self.kernel, verbose=False)

    def get_config(self):
        """Return the parameters dict"""
        return {
            "c": self.c,
            "kernel": self.kernel,
        }
