class KNearstNeighbour:
  def __init__(self,k):
    self.k = k

  def train(self,X_train,y_train):
    self.X_train = X_train
    self.y_train = y_train
 
  def predict(self,X_test):
    distances = self.euclidean_distance(X_test)
    return self.predict_labels(distances)
  
  def euclidean_distance(self,X_test):
    #We want to compute distance 
    #for each test points we want to compute the distance with all other train points
    n_test = X_test.shape[0]
    n_train = self.X_train.shape[0]
    distances = np.zeros((n_test,n_train))

    for i in range(n_test):
      for j in range(n_train):
        distances[i,j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :]) ** 2) )
         
    return distances
  
  def predict_labels(self,distances):
      n_test = distances.shape[0]
      y_pred = np.zeros(n_test)
      for i in range(n_test):
        idx = np.argsort(distances[i,:])
        kn = self.y_train[idx[:self.k]]
        kn = kn.flatten()
        y_pred[i] = np.argmax(np.bincount(kn)).astype(int)
      return y_pred.reshape(len(y_test),1)


