import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

sns.set_theme(style="darkgrid")


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols: list):
        """ Custom sklearn transformer to select a set of columns.
	
        Attributes:
            cols (list of str) representing the columns to be selected 
            in a pandas DataFrame.
                
        """
        self.__cols = cols
        
    @property
    def cols(self):
        return self.__cols
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        return X.loc[:, self.__cols]
    
    
class NumericImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "mean", fill_value=None):
        """ Custom sklearn transformer to impute numeric data when it is missing.
	
        Attributes:
            method (str) representing the method (mean/median/constant)
            fill_value (int/float) representing the constant value to be imputed 
                
        """
        assert method in ["mean", "median", "constant"], \
               "Allowed methods are `mean`, `median`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        
    @property
    def method(self):
        return self.__method
    
    @property
    def fill_value(self):
        return self.__fill_value
    
    def __define_func(self):
        if self.__method == "mean":
            return np.mean
        elif self.__method == "median":
            return np.median
        
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method in ["mean", "median"]:
            func = self.__define_func()
            for column in X_.columns:
                self.__learned_values[column] = func(X_.loc[:, column])
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self
    
    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "mean", fill_value=None):
        """ Custom sklearn transformer to impute categorical data when it is missing.
	
        Attributes:
            method (str) representing the method (mean/median/constant)
            fill_value (int/float) representing the constant value to be imputed 
                
        """
        assert method in ["most_frequent", "constant"], \
               "Allowed methods are `most_frequent`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        
    @property
    def method(self):
        return self.__method
    
    @property
    def fill_value(self):
        return self.__fill_value
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method == "most_frequent":
            for column in X_.columns:
                self.__learned_values[column] = X_.loc[:, column].value_counts(ascending=False).index[0]
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self
    
    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_
    

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        """ Custom sklearn OneHotEncoder that returns pandas DataFrame with the 
        new column names.
	
        Attributes:
            None
                
        """
        self.__oh = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype=np.int32)
        self.__cols = []
        
    def get_feature_names(self):
        return self.__cols
        
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        self.__oh.fit(X, y)
        self.__cols = self.__oh.get_feature_names(input_features=X.columns)
        return self
    
    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        output = pd.DataFrame(self.__oh.transform(X), columns=self.__cols, index=X.index)
        return output


class FeatureSelectorKDD(BaseEstimator, TransformerMixin):
    
    def __init__(self, val_size: float, float_cols: list, strategy: str = "better_than_average"):
        """ Custom sklearn transformer to perform feature selection according to KDD-2009 method.
	
        Attributes:
            val_size (float) representing the validation set size
            float_cols (list of str) representing the columns which contain floats
            strategy (str) representing the strategy to select features (better_than_average/better_than_random)
                
        """
        assert strategy in ["better_than_average", "better_than_random"], \
            "Allowed strategies are `better_than_average`, `better_than_random`"
        self.__strategy = strategy
        self.__val_size = val_size
        self.__features_results = {}
        self.__selected_features = []
        self.__float_cols = float_cols
        
    @property
    def strategy(self):
        return self.__strategy
    
    @property
    def val_size(self):
        return self.__val_size
    
    @property
    def features_results(self):
        return self.__features_results
    
    @property
    def selected_features(self):
        return self.__selected_features
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        assert isinstance(y, pd.Series), "`y` should be a pandas series"
        
        features = X.columns
        df = pd.concat([X, y], axis=1)
        target_name = y.name
        
        idx = int(len(df)*self.__val_size)
        df = df.sample(frac=1.0, replace=False, random_state=99)
        df_train = df.iloc[idx:, :]
        df_val = df.iloc[:idx, :]
        
        for feature in features:
            aux_df_train = df_train.loc[:, [feature, target_name]]
            aux_df_val = df_val.loc[:, [feature, target_name]]
            if feature in self.__float_cols:
                kbin = KBinsDiscretizer(n_bins=3, encode="ordinal")
                aux_df_train.loc[:, feature] = kbin.fit_transform(aux_df_train.loc[:, feature].values.reshape(-1, 1))
                aux_df_val.loc[:, feature] = kbin.transform(aux_df_val.loc[:, feature].values.reshape(-1, 1))
            unique_values = pd.concat([aux_df_train, aux_df_val]).loc[:, feature].unique()
            mapping = aux_df_train.groupby(feature).mean().to_dict()
            for k, v in mapping.items():
                for u_value in unique_values:
                    if u_value not in v.keys():
                        mapping[k][u_value] = 0.0
            for k, v in mapping.items():
                aux_df_val.loc[:, f"pred_{k}"] = aux_df_val.loc[:, feature].map(mapping[k])
                roc_auc = roc_auc_score(aux_df_val.loc[:, target_name], aux_df_val.loc[:, f"pred_{target_name}"])
                self.__features_results[feature] = roc_auc
            
        cut_value = 0.5
        if self.__strategy == "better_than_average":
            cut_value = np.mean(list(self.__features_results.values()))
        self.__selected_features = [f for f, metric in self.__features_results.items() if metric > cut_value]
        return self
    
    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "`X` should be a pandas dataframe"
        if len(self.__selected_features)==0:
            raise ValueError("Transformer not fitted yet!")
        return X.loc[:, self.__selected_features]


class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, trials: int, percentile: float = 0.005,
                 keep_only_tail: bool = False, random_state: int = 99):
        """ Custom sklearn transformer to perform feature selection according to Boruta method.
	
        Attributes:
            trials (int) representing the number of trials to be performed
            percentile (float) representing the percentile to exclude irrelevant features
            keep_only_tail (bool) representing whether or not to discard intermediate features
            random_state (int) representing the seed
                
        """
        self.__PROBABILITY_BINOMIAL = 0.5
        self.__trials = trials
        self.__percentile = percentile
        self.__keep_only_tail = keep_only_tail
        self.__random_state = random_state
        self.__feature_importance_results = dict()
        self.__dt = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        self.__selected_features = None
        
    @property
    def trials(self):
        return self.__trials
    
    @property
    def percentile(self):
        return self.__percentile
    
    @property
    def keep_only_tail(self):
        return self.__keep_only_tail
    
    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def selected_features(self):
        return self.__selected_features
    
    @property
    def feature_importance_results(self):
        return self.__feature_importance_results
    
    def __get_shadow_features(self, X):
        X_boruta = X.apply(np.random.permutation)
        X_boruta.columns = [f"{c}_shadow" for c in X.columns]
        X_concat = pd.concat([X, X_boruta], axis=1)
        return X_concat
    
    def __get_binomial_distribution(self, only_percentiles: bool = False):
        x = np.arange(self.__trials + 1)
        pmf = [sp.stats.binom.pmf(x, self.__trials, self.__PROBABILITY_BINOMIAL) for x in range(self.__trials + 1)]
        lower_x = int(sp.stats.binom.ppf(self.__percentile, self.__trials, self.__PROBABILITY_BINOMIAL))
        upper_x = int(sp.stats.binom.ppf(1-self.__percentile, self.__trials, self.__PROBABILITY_BINOMIAL))
        if only_percentiles:
            return (lower_x, upper_x)
        return (x, pmf), (lower_x, upper_x)
    
    def summary(self, ):
        assert self.__selected_features is not None, "Transformer not fitted yet!"
        
        features_to_drop = []
        features_to_tentatively_keep = []
        features_to_keep = []
        
        lower_x, upper_x = self.__get_binomial_distribution(only_percentiles=True)
        
        for col, hits in self.__feature_importance_results.items():
            to_print = f"[hits: {hits}]"
            if hits <= lower_x:
                features_to_drop.append(f"{col:20} {to_print}")
            elif hits < upper_x:
                features_to_tentatively_keep.append(f"{col:20} {to_print}")
            else:
                features_to_keep.append(f"{col:20} {to_print}")
        
        length = 50
        print("*"*length)
        title = "SUMMARY"
        length_blank = (length-len(title)-2)//2
        print("*"+" "*length_blank+title+" "*(length_blank+(length-len(title)-2)%2)+"*")
        print("*"*length, end="\n\n")
        
        print(f">> Features to drop (<= {lower_x}):", end="\n\t")
        text = ""
        if len(features_to_drop) > 0:
            text = "* "+"\n\t* ".join(features_to_drop)
        print(text, end="\n\n")
        
        print(f">> Features to tentatively keep ({lower_x} < hits < {upper_x}):", end="\n\t")
        text = ""
        if len(features_to_tentatively_keep) > 0:
            text = "* "+"\n\t* ".join(features_to_tentatively_keep)
        print(text, end="\n\n")
        
        print(f">> Features to drop (>= {upper_x}):", end="\n\t")
        text = ""
        if len(features_to_keep) > 0:
            text = "* "+"\n\t* ".join(features_to_keep)
        print(text, end="\n\n")
        
    
    def show_decision_regions(self, show_features: bool = False):
        assert self.__selected_features is not None, "Transformer not fitted yet!"
        
        colors = ["red", "blue", "green"]
        alpha = 0.25
        (x, pmf), (lower_x, upper_x) = self.__get_binomial_distribution()
        
        plt.figure(figsize=(12,6))
        
        plt.fill_between(x[:lower_x+1], pmf[:lower_x+1], alpha=alpha, color=colors[0])
        sns.scatterplot(y=pmf[:lower_x+1], x=x[:lower_x+1], color=colors[0], label="drop")

        plt.fill_between(x[lower_x:upper_x+1], pmf[lower_x:upper_x+1], alpha=alpha, color=colors[1])
        sns.scatterplot(y=pmf[lower_x:upper_x+1], x=x[lower_x:upper_x+1], color=colors[1], label="tentatively keep")

        plt.fill_between(x[upper_x:], pmf[upper_x:], alpha=alpha, color=colors[2])
        sns.scatterplot(y=pmf[upper_x:], x=x[upper_x:], color=colors[2], label="keep")
        
        if show_features:
            for col, hits in self.__feature_importance_results.items():
                plt.annotate(col, (hits, sp.stats.binom.pmf(hits, self.__trials, self.__PROBABILITY_BINOMIAL)), 
                             arrowprops=dict(arrowstyle="->"), textcoords="offset points",
                             xytext=(np.random.randint(-50, 50), np.random.randint(50, 200)))

        plt.title("Boruta | Decision Zones")
        plt.ylabel(f"Binomial Distribution pmf | n = {self.__trials}, p = {self.__PROBABILITY_BINOMIAL}")
        plt.xlabel(f"Number of hits for {self.__trials} trials")

        plt.legend()
        plt.show()
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "A pandas dataframe should be inserted"
        X_ = X.copy()
        original_features = list(X_.columns)
        hits = np.zeros(len(original_features), dtype=np.int32)
        for _ in range(self.__trials):
            X_concat = self.__get_shadow_features(X_)
            self.__dt.fit(X_concat, y)
            feature_importance = self.__dt.feature_importances_
            feature_importance_original = feature_importance[:X_.shape[1]]
            max_importance_shadow = np.max(feature_importance[X_.shape[1]:])
            hits += (feature_importance_original > max_importance_shadow)
        
        lower_x, upper_x = self.__get_binomial_distribution(only_percentiles=True)
        self.__feature_importance_results = {c: hit for c, hit in zip(original_features, hits)}
        filter_value = lower_x
        if self.__keep_only_tail:
            filter_value = upper_x
        self.__selected_features = [col for col, imp in self.__feature_importance_results.items() \
                                            if imp > filter_value]
        return self
        
    def transform(self, X):
        assert self.__selected_features is not None, "Transformer not fitted yet!"
        assert isinstance(X, pd.DataFrame), "A pandas dataframe should be inserted"
        return X.loc[:, self.__selected_features]
