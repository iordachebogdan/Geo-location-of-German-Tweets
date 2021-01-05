from sklearn import svm


class NuSVR(object):
    """NuSVR wrapper that stores regularization parameters"""

    def __init__(self, c, nu, kernel):
        super().__init__()
        self.c = c
        self.nu = nu
        self.kernel = kernel

    def get_regressor(self):
        """Build the regressor"""
        return svm.NuSVR(nu=self.nu, C=self.c, kernel=self.kernel, verbose=True)

    def get_config(self):
        """Return the parameters dict"""
        return {
            "c": self.c,
            "nu": self.nu,
            "kernel": self.kernel,
        }
