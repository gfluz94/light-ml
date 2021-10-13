from typing import List
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_numerical_analysis(df: pd.DataFrame, cols: List[str], univariate: bool = True, hue_col_name: str = None):
    """Function to display histogram and boxplot for each numerical feature. Additionally, an additional column can be included 
    in order to consider bivariate analysis.
		
		Args: 
            df (pandas.DataFrame): dataset with all predictors
            cols (list of str): columns to be considered in this analysis
            univariate (bool): whether it is an univariate or a bivariate statistical analysis
            hue_col_name (str): if univariate is `False`, then a column by which data should be groupped must be informed here
		
		Returns: 
			None
	
	""" 
    _, ax = plt.subplots(len(cols), 2, figsize=(15, 5*len(cols)))
    if ax.ndim == 1:
        ax = ax.reshape(1, -1)
    for i, col in enumerate(cols):
        ax[i][0].set_title(f"{col} - Boxplot")
        ax[i][1].set_title(f"{col} - Distribution")
        if univariate:
            sns.boxplot(y=col, data=df, ax=ax[i][0])
            df.loc[:, col].hist(ax=ax[i][1])
        else:
            assert hue_col_name is not None, "Please inform the name of the hue column"
            sns.boxplot(y=col, x=hue_col_name, data=df, ax=ax[i][0])
            df.loc[df[hue_col_name] == 0, col].hist(ax=ax[i][1])
            df.loc[df[hue_col_name] == 1, col].hist(ax=ax[i][1])
    plt.tight_layout()
    plt.show()


def plot_categorical_analysis(df: pd.DataFrame, cols: List[str], univariate: bool = True, hue_col_name: str = None):
    """Function to display countplot for every single categorical predictor.
		
		Args: 
            df (pandas.DataFrame): dataset with all predictors
            cols (list of str): columns to be considered in this analysis
            univariate (bool): whether it is an univariate or a bivariate statistical analysis
            hue_col_name (str): if univariate is `False`, then a column by which data should be groupped must be informed here
		
		Returns: 
			None
	
	""" 
    _, ax = plt.subplots(len(cols), 1, figsize=(15, 5*len(cols)))
    df_ = df.copy()
    for i, col in enumerate(cols):
        if df_[col].nunique() > 30:
            count = df_[col].value_counts()
            vals_to_keep = list(count[count/len(df_)>0.01].index)
            df_[col] = df_[col].apply(lambda x: x if x in vals_to_keep else "< 1%")
        ax[i].set_title(f"{col} - Count")
        if univariate:
            sns.countplot(y=col, data=df_, ax=ax[i])
        else:
            assert hue_col_name is not None, "Please inform the name of the hue column"
            sns.countplot(y=col, hue=hue_col_name, data=df_, ax=ax[i])
    plt.tight_layout()
    plt.show()