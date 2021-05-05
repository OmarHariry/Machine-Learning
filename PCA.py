class PCA:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Z_train = None
        self.Z_test = None
        self.value = None
        self.vector = None

    def fit(self):
        mu_train = np.mean(self.X_train, axis=0)
        mu_test = np.mean(self.X_test, axis=0)

        self.Z_train = self.X_train - mu_train.T
        self.Z_test = self.X_test - mu_test.T

        cov = np.cov(self.Z_train.T, bias=True)
        # print(cov.shape)
        self.value, self.vector = np.linalg.eigh(cov)
        idx = self.value.argsort()[::-1]
        self.value = self.value[idx]
        self.vector = self.vector[:, idx]
        return self.value, self.vector

    def findNumberOfComponents(self, alpha):
        # vector represents eigen vector
        # value represents eigen values
        # Alpha represents the desired explained variance / information
        # the function returns number of components/axis to achieve the given alpha
        variance_explained = np.array(((self.value / np.sum(self.value))))
        n_components = 0
        for i in range(len(variance_explained)):
            if alpha <= 0:
                break
            else:
                alpha -= variance_explained[i]
                n_components = n_components + 1

        return n_components

    def transform(self, n_components):
        W = self.vector[:, 0:n_components]
        pca_train = np.dot(W.transpose(), self.Z_train.transpose()).transpose()
        pca_test = np.dot(W.transpose(), self.Z_test.transpose()).transpose()

        return pca_train, pca_test

