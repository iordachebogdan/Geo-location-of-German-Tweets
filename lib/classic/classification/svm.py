from sklearn import svm


class SVM(object):
    def __init__(self, c, kernel):
        super().__init__()
        self.c = c
        self.kernel = kernel

    def get_classifier(self):
        return svm.SVC(C=self.c, kernel=self.kernel, verbose=False)

    def get_config(self):
        return {
            "c": self.c,
            "kernel": self.kernel,
        }
