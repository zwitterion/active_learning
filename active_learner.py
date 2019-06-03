
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ActiveLearner():
    def __init__(self, X, y, model=None):
        # reference to data pool
        self.X = X
        self.y = y
        
        if (model == None):
            self.model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                      intercept_scaling=1, max_iter=100, multi_class='warn',
                      n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
                      tol=0.0001, verbose=0, warm_start=False)
        else:    
            self.model = model
            
        self.seed_size_in_rows = 10
        self.model_trained = False
        self.score = 0

    def entropy(self, values):
        return -np.sum([p * np.log(p) for p in values])
    
    def train(self, test_size=0.33, random_state=None):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,  test_size=test_size, random_state=random_state)
        
        if (len(self.y_train.unique()) >= 2):
            # at least two classes are required 
            self.model.fit(self.X_train, self.y_train)
            self.score = self.model.score(self.X_test, self.y_test)
            self.model_trained = True
            return True
        else:
            return False
    
    
    def query_samples_to_label(self, number_of_samples, strategy="entropy"):
        """
        Query best samples to be labeled using entropy strategy
        """
        if (not self.model_trained):
            samples_idx = np.random.choice(self.X.shape[0], number_of_samples, replace=False)
        else:
            if (strategy == 'entropy'):
                proba = self.model.predict_proba(self.X)
                entropies = [ self.entropy(r) for r in proba ]
                entropies_sort = np.argsort(entropies)

                samples_idx = []
                i=1
                while(len(samples_idx) < number_of_samples and i<len(self.X)):
                      candidate = entropies_sort[-i]
                      if (not self.labeled_idx[candidate]):
                            samples_idx.append(candidate)
                      i += 1
                samples_idx = np.array(samples_idx)
            else:
                samples_idx = np.random.choice(self.X.shape[0], self.seed_size_in_rows, replace=False)
           
        return samples_idx

    
   