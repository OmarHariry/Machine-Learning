class LDA:
  def __init__(self,X_train,X_test,y_train,y):
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y = y
    self.value = None
    self.vector = None

  def fit(self):
    classes = np.unique(y).reshape(40,)
    self.y_train = self.y_train.reshape(y_train.shape[0],)
    mu_sample = np.mean(self.X_train,axis=0)
    
    n_features = self.X_train.shape[1]
    B = np.zeros((n_features,n_features))
    S = np.zeros((n_features,n_features))

    for c in classes:
      X_class = self.X_train[c==self.y_train]
      mu_class = np.mean(X_class,axis = 0)
      n = X_class.shape[0]
      B += n*np.dot((mu_class - mu_sample),(mu_class - mu_sample).T)
      S += np.cov((X_class - mu_class).T,bias = False) 
      
    Sinv = np.linalg.inv(S)
    SinvB = np.dot(Sinv,B)

    self.value, self.vector = np.linalg.eigh(SinvB)
    idx = self.value.argsort()[::-1]   
    self.value = self.value[idx]
    self.vector = self.vector[:,idx]
    
    return self.value,self.vector

  def transform(self,n_components):
    W = self.vector[:,:n_components]
    lda_train = np.dot(W.T,self.X_train.T).T
    lda_test = np.dot(W.T,self.X_test.T).T
   
    return lda_train,lda_test