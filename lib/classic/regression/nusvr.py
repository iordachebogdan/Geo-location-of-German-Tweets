from sklearn import svm


class NuSVR(object):
    def __init__(self, c, nu, kernel):
        super().__init__()
        self.c = c
        self.nu = nu
        self.kernel = kernel

    def get_regressor(self):
        return svm.NuSVR(nu=self.nu, C=self.c, kernel=self.kernel)

    def get_config(self):
        return {
            "c": self.c,
            "nu": self.nu,
            "kernel": self.kernel,
        }
