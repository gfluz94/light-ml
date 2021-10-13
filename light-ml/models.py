import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class BalancedRandomForest(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_estimators=100, max_depth=5, max_feats="sqrt", random_state=99):
        """ Custom RandomForest algorithm which attempts to overcome imbalance by bootstrapping balanced 
        data for each Decision Tree to be fitted on.
	
        Attributes:
            n_estimators (int) representing the number of trees
            max_depth (int) representing the maximum depth of each tree
            max_feats (str) representing the number of features to be considered for each split (sqrt/max)
            random_state (int) representing the seed to make the experiment reproducible
                
        """
        assert max_feats in ["sqrt", "max"], \
            "Allowed max_feats are `sqrt`, `max`"
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__max_feats = max_feats
        self.__random_state = random_state
        self.__estimators = [
            DecisionTreeClassifier(max_depth=self.__max_depth, random_state=self.__random_state) \
                for _ in range(self.__n_estimators)
        ]
        self.__features = []
        
    @property
    def n_estimators(self):
        return self.__n_estimators
    
    @property
    def max_depth(self):
        return self.__max_depth
    
    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def max_feats(self):
        return self.__max_feats
    
    def __generate_balanced_bootstrapping(self, X, y):
        min_class = y.value_counts().sort_values().index[0]
        min_class_count = y.value_counts().sort_values().iloc[0]
        y_min = y[y==min_class].sample(frac=1.0, replace=True)
        y_maj = y[y!=min_class].sample(n=2*min_class_count, replace=True)
        new_y = pd.concat([y_min, y_maj], axis=0).sample(frac=1.0, replace=False)
        new_y_indices = new_y.index
        new_X = X.loc[new_y_indices, :]
        return new_X, new_y
    
    def __select_features(self, X):
        X_ = X.copy()
        features = np.array(X.columns)
        n_features = len(features)
        if self.__max_feats == "sqrt":
            new_n_features = int(np.sqrt(n_features))
        elif self.__max_feats == "max":
            new_n_features = n_features
        np.random.shuffle(features)
        return features[:new_n_features]
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        assert isinstance(y, pd.Series), "`y` should be a pandas series"
    
        X_ = X.copy()
        y_ = y.copy()
        for i in range(self.__n_estimators):
            X_boot, y_boot = self.__generate_balanced_bootstrapping(X_, y_)
            feats = self.__select_features(X_boot)
            self.__estimators[i].fit(X_boot.loc[:, feats], y_boot)
            self.__features.append(feats)
        return self
    
    def predict_proba(self, X):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        predicted_proba = np.zeros((self.__n_estimators, len(X), 2))
        for i in range(self.__n_estimators):
            predicted_proba[i, :, :] = self.__estimators[i].predict_proba(X.loc[:, self.__features[i]])
        return predicted_proba.mean(axis=0)
    
    def predict(self, X):
        predicted_proba = self.predict_proba(X)
        return predicted_proba.argmax(axis=1)