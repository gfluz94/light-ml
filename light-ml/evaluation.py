import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                             precision_recall_curve, r2_score, mean_squared_error)

sns.set_theme(style="darkgrid")


def __rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_regression_analysis(model, X, y_true, name=None):
    """Function to display four visualizations to evaluate model's performance: True x Predicted values histogram, 
    Residuals Plot, QQ-Plot and True x Predicted values scatter-plot.
		
		Args: 
			model (sklearn.Pipeline): complete preprocessing amd model building pipeline
            X (pandas.DataFrame): dataset with all predictors
            y_true (pandas.Series): true labels
            name (str): model's name to be displayed in the visualizations
		
		Returns: 
			None
	
	""" 
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14,3))
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    if name is not None:
        txt_to_show = f"{name} | RMSE: {__rmse(y_true, y_pred):.3f} | R2: {r2_score(y_true, y_pred):.3f}"
        print(len(txt_to_show)*'-')
        print(txt_to_show)
        print(len(txt_to_show)*'-')
    ax1.set_title("Histogram", fontsize=13)
    ax1.hist(y_true, color="lightblue", edgecolor="navy", alpha=1, label="True")
    ax1.hist(y_pred, color="red", edgecolor="red", alpha=0.6, label="Predicted")
    ax1.legend()

    ax2.set_title("Residuals Plot", fontsize=13)
    sns.regplot(x=y_pred, y=residuals, ci=0, 
                scatter_kws={"color":"lightblue", "linewidth":1, "edgecolors":"navy"}, 
                line_kws={"color": "red"}, ax=ax2)
    ax2.plot([min(y_pred), max(y_pred)], [0, 0], color="black", linestyle="--")
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")

    standardized_residuals = (np.sort(residuals)-np.mean(residuals))/np.std(residuals, ddof=1)
    theoretical_quantiles = [scipy.stats.norm.ppf(p) for p in np.linspace(0.01, 0.99, len(standardized_residuals))]
    ax3.set_title("QQ-Plot", fontsize=13)
    sns.regplot(x=theoretical_quantiles, y=standardized_residuals, fit_reg=False,
                scatter_kws={"color":"lightblue", "linewidth":1, "edgecolors":"navy"}, 
                line_kws={"color": "red"}, ax=ax3)
    ax3.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
             [min(theoretical_quantiles), max(theoretical_quantiles)], color="red", linestyle="--")
    ax3.set_xlim((min(theoretical_quantiles), max(theoretical_quantiles)))
    ax3.set_ylim((min(theoretical_quantiles), max(theoretical_quantiles)))
    ax3.set_ylabel("Standardized Residuals")
    ax3.set_xlabel("Theoretical Quantiles")

    ax4.set_title("Predicted x True", fontsize=13)
    min_val = min([y_pred.min(), y_true.min()])
    max_val = max([y_pred.max(), y_true.max()])
    sns.scatterplot(y_pred, y_true, color="lightblue", edgecolor="navy", ax=ax4)
    ax4.plot([min_val, max_val], [min_val, max_val], color="red", ls="dashed")
    ax4.set_xlim([min_val, max_val])
    ax4.set_ylim([min_val, max_val])
    ax4.set_xlabel("Predicted Values")
    ax4.set_ylabel("True Values")

    plt.tight_layout()
    plt.show()


def binary_classification_evaluation_plot(model, X, y_true):
    """Function to display four visualizations to evaluate model's performance: ROC-AUC, CAP, 
    Precision-Recall and KS.
		
		Args: 
			model (sklearn.Pipeline): complete preprocessing amd model building pipeline
            X (pandas.DataFrame): dataset with all predictors
            y_true (pandas.Series): true labels
		
		Returns: 
			None
	
	""" 
    try:
        y_proba = model.predict_proba(X)[:,1]
    except:
        raise ValueError("Model does not return probabilities!")

    if len(np.unique(y_true))!=2:
        raise ValueError("Multiclass Problem!")

    _, ax = plt.subplots(2,2,figsize=(12,8))
    __plot_roc(y_true, y_proba, ax[0][0])
    __plot_pr(y_true, y_proba, ax[0][1])
    __plot_cap(y_true, y_proba, ax[1][0])
    __plot_ks(y_true, y_proba, ax[1][1])
    plt.tight_layout()
    plt.show()

def __plot_cap(y_test, y_proba, ax):
    cap_df = pd.DataFrame(data=y_test, index=y_test.index)
    cap_df["Probability"] = y_proba

    total = cap_df.iloc[:, 0].sum()
    perfect_model = (cap_df.iloc[:, 0].sort_values(ascending=False).cumsum()/total).values
    current_model = (cap_df.sort_values(by="Probability", ascending=False).iloc[:, 0].cumsum()/total).values

    max_area = 0
    covered_area = 0
    h = 1/len(perfect_model)
    random = np.linspace(0, 1, len(perfect_model))
    for i, (am, ap) in enumerate(zip(current_model, perfect_model)):
        try:
            max_area += (ap-random[i]+perfect_model[i+1]-random[i+1])*h/2
            covered_area += (am-random[i]+current_model[i+1]-random[i+1])*h/2
        except:
            continue
    accuracy_ratio = covered_area/max_area

    ax.plot(np.linspace(0, 1, len(current_model)), current_model, 
                        color="green", label=f"AR = {accuracy_ratio:.3f}")
    ax.plot(np.linspace(0, 1, len(perfect_model)), perfect_model, color="red", label="Perfect Model")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("Individuals", fontsize=12)
    ax.set_ylabel("Target Individuals", fontsize=12)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.01))
    ax.legend(loc=4, fontsize=10)
    ax.set_title("CAP Analysis", fontsize=13)

def __plot_roc(y_test, y_proba, ax):
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    ax.plot(fpr, tpr, color="red", label=f"(AUC = {roc_auc_score(y_test, y_proba):.3f})")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("ROC Analysis", fontsize=13)

def __plot_pr(y_test, y_proba, ax):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    ax.plot(recall, precision, color="red", label=f"PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("Precision-Recall Analysis", fontsize=13)

def __plot_ks(y_test, y_proba, ax):
    prediction_labels = pd.DataFrame(y_test.values, columns=["True Label"])
    prediction_labels["Probabilities"] = y_proba
    prediction_labels["Thresholds"] = prediction_labels["Probabilities"].apply(lambda x: np.round(x, 2))
    df = prediction_labels.groupby("Thresholds").agg(["count", "sum"])[["True Label"]]
    ks_df = pd.DataFrame(df["True Label"]["sum"]).rename(columns={"sum":"Negative"})
    ks_df["Positive"] = df["True Label"]["count"]-df["True Label"]["sum"]
    ks_df["Negative"] = ks_df["Negative"].cumsum()/ks_df["Negative"].sum()
    ks_df["Positive"] = ks_df["Positive"].cumsum()/ks_df["Positive"].sum()
    ks_df["KS"] = ks_df["Positive"]-ks_df["Negative"]
    ks_df.loc[0.0, :] = [0.0, 0.0, 0.0]
    ks_df = ks_df.sort_index()
    max_ks_thresh = ks_df.KS.idxmax()

    ks_df.drop("KS", axis=1).plot(color=["red", "navy"], ax=ax)
    ax.set_title("KS Analysis", fontsize=13)
    ax.plot([max_ks_thresh, max_ks_thresh], 
            [ks_df.loc[max_ks_thresh,"Negative"], ks_df.loc[max_ks_thresh,"Positive"]],
            color="green", label="Max KS")
    ax.text(max_ks_thresh-0.16, 0.5, f"KS={ks_df.loc[max_ks_thresh,'KS']:.3f}", fontsize=12, color="green")
    ax.legend()
    
def print_classification_report(model, X_test, y_test, threshold=0.5):
    """Function to display classification's report (recall, precisionm f1-score for each class) according to 
    a given threshold.
		
		Args: 
			model (sklearn.Pipeline): complete preprocessing amd model building pipeline
            X_test (pandas.DataFrame): dataset with all predictors
            y_test (pandas.Series): true labels
            threshold (float): threshold to be considered in order to apply the metrics
		
		Returns: 
			None
	
	""" 
    y_pred = (model.predict_proba(X_test)[:,1]>threshold)*1
    print(classification_report(y_test, y_pred))