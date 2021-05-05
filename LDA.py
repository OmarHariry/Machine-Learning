class LDA:
    def __init__(self, X_train, X_test, y_train, y):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y = y
        self.value = None
        self.vector = None

    def fit(self):
        classes = np.unique(y).reshape(2, )
        self.y_train = self.y_train.reshape(y_train.shape[0], )
        n_features = self.X_train.shape[1]
        B = np.zeros((n_features, n_features))
        S = np.zeros((n_features, n_features))
        X_class0 = self.X_train[0 == self.y_train]
        mu_class0 = np.mean(X_class0, axis=0).reshape(n_features, 1)
        X_class1 = self.X_train[1 == self.y_train]
        mu_class1 = np.mean(X_class1, axis=0).reshape(n_features, 1)
        B = np.dot((mu_class0 - mu_class1), (mu_class0 - mu_class1).T)
        mu_class0 = mu_class0.reshape(n_features, )
        mu_class1 = mu_class1.reshape(n_features, )
        S_class0 = (X_class0 - mu_class0.T).T.dot(X_class0 - mu_class0.T)
        S_class1 = (X_class1 - mu_class1.T).T.dot(X_class1 - mu_class1.T)
        S = S_class0 + S_class1
        Sinv = np.linalg.inv(S)
        SinvB = np.dot(Sinv, B)
        self.value, self.vector = np.linalg.eigh(SinvB)
        idx = self.value.argsort()[::-1]
        self.value = self.value[idx]
        self.vector = self.vector[:, idx]
        return self.value, self.vector

    def transform(self, n_components):
        W = self.vector[:, :n_components]
        lda_train = np.dot(W.T, self.X_train.T).T
        lda_test = np.dot(W.T, self.X_test.T).T

        return lda_train, lda_test