# Databricks notebook source
# -*- coding: utf-8 -*-

from typing import Union, NoReturn
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA
import random
from statsmodels.stats.power import TTestIndPower
from numpy.random import binomial, normal
from statistics import harmonic_mean
from scipy.stats import zscore
from random import shuffle
from time import time




# COMMAND ----------

def _validate_context_type(contexts) -> NoReturn:
    """
        Validates that context data is 2D
        """
    if isinstance(contexts, np.ndarray):
        check_true(contexts.ndim == 2,
                   TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))
    elif isinstance(contexts, list):
        check_true(np.array(contexts).ndim == 2,
                   TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))
    else:
        check_true(isinstance(contexts, (pd.Series, pd.DataFrame)),
                   TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))


def _convert_matrix(matrix_like, row=False) -> Union[None, np.ndarray]:
    """
    Convert contexts to numpy array for efficiency.
    For fit and partial fit, decisions must be provided.
    The numpy array need to be in C row-major order for efficiency.
    If the data is a series for a single row, set the row flag to True.
    """
    if matrix_like is None:
        return None
    elif isinstance(matrix_like, np.ndarray):
        if matrix_like.flags['C_CONTIGUOUS']:
            return matrix_like
        else:
            return np.asarray(matrix_like, order="C")
    elif isinstance(matrix_like, list):
        return np.asarray(matrix_like, order="C")
    elif isinstance(matrix_like, pd.DataFrame):
        if matrix_like.values.flags['C_CONTIGUOUS']:
            return matrix_like.values
        else:
            return np.asarray(matrix_like.values, order="C")
    elif isinstance(matrix_like, pd.Series):
        if row:
            return np.asarray(matrix_like.values, order="C").reshape(1, -1)
        else:
            return np.asarray(matrix_like.values, order="C").reshape(-1, 1)
    else:
        raise NotImplementedError("Unsupported contexts data type")


def check_true(expression: bool, exception: Exception):
    """
    Raises the given exception if the given boolean expression does not evaluate to true

    :param expression: the given expression to be evaluated
    :param exception: the exception to raise if the expression is not true
    """
    if not expression:
        raise exception
        
def _validate_report_args(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> NoReturn:
    check_true(len(y_true) == len(y_pred), ValueError('Dimensions of y_true and y_pred must be the same.'))
    if weights is not None:
        check_true(len(y_true) == len(weights), ValueError('Dimensions of y_true and weights must be the same.'))
        check_true(np.min(weights) >= 0, ValueError("Weight values must be positive"))

        
class CapFloor(TransformerMixin):
    """
    Cap and floor outlier values in contexts based on lower and upper percentiles for each feature.
    """

    def __init__(self, p_low: float = 0.001, p_high: float = 0.999):
        """
        Initialize the given parameters.

        :param p_low: Lower percentile used to compute floor. Float.
        :param p_high: Upper percentile used to compute cap. Float.
        """
        self.p_low = p_low
        self.p_high = p_high

        # Initialize
        self.floor = None
        self.cap = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Compute cap and floor values based on input data.

        :param X: Input contexts. Array-like.
        """

        contexts = _convert_matrix(X)

        self.floor = np.quantile(X, q=self.p_low, axis=0).astype(contexts.dtype)
        self.cap = np.quantile(X, q=self.p_high, axis=0).astype(contexts.dtype)

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Cap and floor contexts based on fitted cap and floor values.

        :param X: Input contexts. Array-like.
        :return: X: Transformed contexts array.
        """

        contexts = _convert_matrix(X)

        return np.clip(contexts, self.floor, self.cap)

    def fit_transform(self, X, y=None, **fit_params):
        """
        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        self.fit(X)
        return self.transform(X)
      



# COMMAND ----------

def get_iv_class(y: np.ndarray, x: np.ndarray, x_name: str = None, uniq=10, bins=5):
    """

    :param y:
    :param x:
    :param x_name:
    :param uniq:
    :param bins:
    :return:
    """
    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Convert to data frame
    in_df = pd.DataFrame(arr, columns=['y', x_name])

    # checking the datatype of the input variable
    if in_df[x_name].dtype == 'O':
        try:
            in_df[x_name] = in_df[x_name].apply(np.float32)
        except:
            pass

    # impute missing values with median else do nothing
    if in_df[x_name].isnull().sum() > 0:
        med = in_df[x_name].median()
        in_df[x_name] = in_df[x_name].apply(lambda x: med if pd.isnull(x) == True else x)

    n_uniq = in_df[x_name].nunique()
    # binning X variable separately for indicator and continuous values
    if n_uniq <= uniq:
        in_df['bins'] = in_df[x_name]
    else:
        in_df['bins'] = pd.qcut(in_df[x_name], bins, labels=False, duplicates='drop')

    # if variable has only 1 bin then return 0 IV
    if in_df['bins'].nunique() <= 1:
        return [x_name, 0.0]

    else:
        in_df.sort_values(by=['bins'], inplace=True)
        wwoe = []
        for i, val in enumerate(in_df['y'].unique()):

            # calculating total events and non events
            resp = float(in_df[in_df['y'] == val].shape[0])
            non_resp = float(in_df[in_df['y'] != val].shape[0])

            # calculating bin level distribution of events and non events
            resp_bin = in_df[in_df['y'] == val].groupby(['bins'])['y'].apply(lambda x: x.count() / resp)
            non_resp_bin = in_df[in_df['y'] != val].groupby(['bins'])['y'].apply(lambda x: x.count() / non_resp)

            bin_stats = pd.merge(non_resp_bin.rename("non_resp_bin"), 
                           resp_bin.rename("resp_bin"), 
                           left_index=True, right_index=True, how='outer').fillna(0)

            resp_bin = bin_stats['resp_bin']
            non_resp_bin = bin_stats['non_resp_bin']

            # calculating differnce in bin level distribution of events and non events
            if n_uniq <= uniq:
                distribution_diff = np.array(non_resp_bin - resp_bin).reshape(n_uniq, )
            else:
                distribution_diff = np.array(non_resp_bin - resp_bin).reshape(in_df['bins'].nunique(), )

            # calculating WOE and IV
            if n_uniq <= uniq:
                woe_bin = np.array(np.log(non_resp_bin / resp_bin)).reshape(n_uniq, )
            else:
                woe_bin = np.array(np.log(non_resp_bin / resp_bin)).reshape(in_df['bins'].nunique(), )

            wwoe.append(distribution_diff * np.nan_to_num(woe_bin, neginf=0, posinf=0))

        return [x_name, round(np.array(wwoe).max(axis=0).sum(), 3)]


def get_iv_reg(y: np.ndarray, x: np.ndarray, x_name: str = None, uniq=10, bins=5):
    """

    :param y:
    :param x:
    :param x_name:
    :param uniq:
    :param bins:
    :return:
    """
    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Convert to data frame
    in_df = pd.DataFrame(arr, columns=['y', x_name])

    # checking the datatype of the input variable
    if in_df[x_name].dtype == 'O':
        in_df[x_name] = in_df[x_name].apply(np.float32)

    # impute missing values with median else do nothing
    if in_df[x_name].isnull().sum() > 0:
        med = in_df[x_name].median()
        in_df[x_name] = in_df[x_name].apply(lambda x: med if pd.isnull(x) == True else x)

    # calculating sum of dep var
    dep_sum = float(in_df['y'].values.sum())
    dep_count = float(in_df['y'].count())

    n_uniq = in_df[x_name].nunique()
    # binning X variable separately for indicator and continuous values
    if n_uniq <= uniq:
        in_df['bins'] = in_df[x_name]
    else:
        in_df['bins'] = pd.qcut(in_df[x_name], bins, labels=False, duplicates='drop')

    # if variable has only 1 bin then return 0 IV
    if in_df['bins'].nunique() <= 1:
        return [x_name, 0.0]

    else:
        # calculating bin level distribution of bin sum and total records
        resp_bin = in_df.groupby(['bins'])['y'].apply(lambda x: x.sum() / dep_sum)
        pop_bin = in_df.groupby(['bins'])['y'].apply(lambda x: (x.count()) / dep_count)

        bin_stats = pd.merge(non_resp_bin.rename("non_resp_bin"), 
                           resp_bin.rename("resp_bin"), 
                           left_index=True, right_index=True, how='outer').fillna(0)

        resp_bin = bin_stats['resp_bin']
        non_resp_bin = bin_stats['non_resp_bin']

        # calculating differnce in bin level distribution of events and non events
        if n_uniq <= uniq:
            distribution_diff = np.array(pop_bin - resp_bin).reshape(1, n_uniq)
        else:
            distribution_diff = np.array(pop_bin - resp_bin).reshape(1, in_df['bins'].nunique())

        # calculating WOE and IV
        if n_uniq <= uniq:
            woe_bin = np.array(np.log(pop_bin / resp_bin)).reshape(n_uniq, 1)
        else:
            woe_bin = np.array(np.log(pop_bin / resp_bin)).reshape(in_df['bins'].nunique(), 1)
        iv = np.dot(distribution_diff, np.nan_to_num(woe_bin, neginf=0, posinf=0))[0, 0]

        return [x_name, round(iv, 3)]


def iv_group(iv):
    """

    :param iv:
    :return:
    """
    if iv >= 0.5 and np.isinf(iv) == False:
        return 'suspecious'
    elif 0.5 > iv >= 0.3:
        return 'strong'
    elif 0.3 > iv >= 0.1:
        return 'medium'
    elif 0.1 > iv >= 0.02:
        return 'weak'
    else:
        return 'useless'

# COMMAND ----------

class BalancingWeights:
    """
    Class to calculate weights for balancing biased sample to a population using context features
    """

    def __init__(self, prob_scale: bool = True, prob_score: str = 'log', n_bins: int = 200, n_components: int = None,
                 random_state: int = None):
        """
        Initializes the given parameters.

        :param prob_scale: Scale population total probability to 1 if True else no scaling.
        :param prob_score: 'log' or 'actual' probabilities.
        :param n_bins: Number of equally sized groups for weight calculation.
        :param n_components: Number of principal components to use. All components used if None.
        :param random_state: The random seed to initialize the internal random number generator. Integer.
        """

        self.prob_scale = prob_scale
        self.prob_score = prob_score
        self.n_bins = n_bins
        self.n_components = n_components
        self.random_state = random_state

        # Initialize
        self.imputer = SimpleImputer(strategy="median")
        self.cap_floor = CapFloor(p_low=0.005, p_high=0.995)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.mvn = None

    def fit(self, df: pd.DataFrame):
        """
        Fit a multivariate normal distribution on contexts from population to generate weights for sample.

        :param df: DataFrame with pre-processed contexts from population
        """

        self.n_components = df.shape[1] if self.n_components is None else self.n_components

        # pre-process contexts and fit PCA
        df = self.imputer.fit_transform(df)
        df = self.cap_floor.fit_transform(df)
        df = self.scaler.fit_transform(df)
        df = pd.DataFrame(self.pca.fit_transform(df))

        # Multivariate Normal Distribution
        mu = df.mean(axis=0).values
        sigma = df.cov().values
        self.mvn = multivariate_normal(cov=sigma, mean=mu, allow_singular=True, seed=self.random_state)

        # Population bins
        self._bin_population(df)

    def get_weights(self, df: pd.DataFrame):
        """
        Score the learned population distribution from fit on the sample contexts and calculate weight for each
        observation in sample.

        :param df: DataFrame with pre-processed contexts from sample
        :return: weights array
        """

        # Scale contexts and fit PCA
        df = self.imputer.transform(df)
        df = self.cap_floor.transform(df)
        df = self.scaler.transform(df)
        df = pd.DataFrame(self.pca.transform(df))

        # Bin sample
        bins = self._bin_sample(df)

        # Sample, population totals for calculating weights
        sample_total = len(df)
        population_total = sum(self.bin_to_population_count.values())

        # Calculate weight for each bin
        bin_to_weight = {}
        for b in range(self.n_bins):
            population_proportion = self.bin_to_population_count[b] / population_total
            sample_proportion = self.bin_to_sample_count[b] / sample_total
            bin_to_weight[b] = population_proportion / sample_proportion

        # Get weight for each observation based on bin
        weights = np.zeros(sample_total)
        for index, b in enumerate(bins):
            weights[index] = bin_to_weight[b]

        return weights

    def _sample_mvn(self, df):
        if self.prob_score == 'actual':
            scores = self.mvn.pdf(df)
        elif self.prob_score == 'log':
            scores = self.mvn.logpdf(df)
        else:
            raise NotImplementedError
        return scores

    def _bin_population(self, df):

        scores = self._sample_mvn(df)
        if self.prob_scale:
            self.population_weight = np.sum(scores)
            scores = scores / self.population_weight

        bins, bin_edges = pd.qcut(scores, q=self.n_bins, labels=False, retbins=True, duplicates='drop')

        self.bin_edges = bin_edges
        self.bin_to_population_count = dict(pd.value_counts(bins))

    def _bin_sample(self, df):
        scores = self._sample_mvn(df)
        if self.prob_scale:
            scores = scores / self.population_weight

        if np.min(scores) < self.bin_edges[0]:
            self.bin_edges[0] = np.min(scores)
        if np.max(scores) > self.bin_edges[-1]:
            self.bin_edges[-1] = np.max(scores)

        bins = pd.cut(scores, bins=self.bin_edges, labels=False, include_lowest=True)
        self.bin_to_sample_count = dict(pd.value_counts(bins))

        if len(self.bin_to_sample_count) != self.n_bins:
            raise ValueError('Some bins contain zero samples. Choose a lower number of bins')

        return bins


def balancing_report(population: pd.DataFrame, sample: pd.DataFrame, weights: np.ndarray):
    """
    Calculate mean for each context in population and sample data (weighted and unweighted).

    :param population: Target population. Data Frame.
    :param sample: Sample population for which weights were computed. Data Frame.
    :param weights: Computed weight for each observation in sample. Array.

    :return: report: Report Data Frame.
    """
    mean_list = []
    for col in population.columns:
        d = {'feature': col,
             'population_mean': np.average(population[col]),
             'sample_mean': np.average(sample[col]),
             'sample_mean_w': np.average(sample[col], weights=weights)}
        mean_list.append(d)
    report = pd.DataFrame(mean_list)
    report.sort_values(['feature'], inplace=True)
    cols = ['feature', 'sample_mean', 'sample_mean_w', 'population_mean']
    return report[cols]


def grouped_t_test(df: pd.DataFrame, group_var: str, varlist: list) -> pd.DataFrame:
    """
    :param df:
    :param group_var:
    :param varlist:
    :return:
    """
    means_df = df.groupby([group_var])[varlist].mean().T
    means_df.columns = [f'{group_var} = 0', f'{group_var} = 1']

    t_test_results = {}

    index = df[group_var] == 1.0

    for column in varlist:
        t_test_results[column] = ttest_ind(df[index][column].values,
                                                       df[~index][column].values)
    t_test_df = pd.DataFrame.from_dict(t_test_results, orient='Index')
    t_test_df.columns = ['t-test', 'p-value']
    t_test_df['significance'] = t_test_df['p-value'].apply(lambda x: "***" if x <= 0.01 else
    ("**" if x <= 0.05 else ("*" if x <= 0.1 else "")))

    results = np.around(pd.concat([means_df, t_test_df], axis=1), decimals=2)

    return results

# COMMAND ----------

def lift_report(y_true: np.ndarray, y_pred: np.ndarray, bin_cutoffs: np.ndarray = None,
                weights: np.ndarray = None, segment: np.ndarray = None, n: int = 20) -> pd.DataFrame:
    """
    Calculates cumulative precision, recall and lift by bins.

    Samples are grouped into bins of equal size by ranking the predicted values in descending order. Used
    for classification problems.

    :param y_true: Ground truth (correct) labels. Array-like of shape = (n_samples).
    :param y_pred: Predicted probabilities. Array-like of shape = (n_samples).
    :param bin_cutoffs: apply pre-defined bin cutoffs on y_pred
    :param weights: Weight given to each observation.
    :param segment: indicator variable for a segment.
    :param n: Number of bins. Integer. Default value is 10.

    :return: df: Report Data Frame.
    """

    _validate_report_args(y_true, y_pred, weights)
    check_true(np.all(np.unique(y_true) == [0, 1]), ValueError("y_true must be a binary response with 0/1 values."))

    # Create unit weights/segment if None
    if segment is None:
        segment = np.ones(len(y_true))
    if weights is None:
        weights = np.ones(len(y_true))

    # Stack arrays with responses and predictions
    arr = np.vstack((y_true, y_pred, weights, segment)).T

    # Convert to data frame
    df = pd.DataFrame(arr, columns=['y_true', 'y_pred', 'weight', 'segment'])

    # Weighted actual variable
    df['y_true_w'] = df['y_true'] * df['weight']

    if bin_cutoffs is None:

        # Create bins based on weighted ranked predictions
        quantiles = np.linspace(0, 1, n + 1)
        df.sort_values('y_pred', inplace=True, ascending=False)
        cum_weight = df['weight'].cumsum().values
        df['bin'] = pd.cut(cum_weight / cum_weight[-1], bins=quantiles, labels=False)

    else:

        # Create bins based on given bin cutoffs

        n = len(bin_cutoffs)
        choices = []
        for i in range(n):
            choices.append(i)

        conditions = []
        for cutoff in bin_cutoffs:
            conditions.append((df['y_pred'] >= cutoff))

        df['bin'] = np.select(conditions, choices, default=n - 1)

    check_true(df['bin'].nunique() == n, ValueError('Some bins have no samples. Choose a lower number of bins.'))

    # Filter for segment
    df = df[df.segment == 1].reset_index(drop=True)

    # Group samples by bin
    grp = df.groupby('bin')

    # Calculate number of samples and weight for each bin
    N = grp.size().rename('N')
    weight = grp['weight'].sum()

    # Calculate the predicted mean, actual mean and number
    # of positive responses by bin
    pred = grp['y_pred'].agg(['min', 'mean', 'max'])
    actual = grp['y_true'].agg(['mean', 'sum'])
    actual_w = grp['y_true_w'].agg(['mean', 'sum'])

    # Convert to data frame
    df = pd.concat((N, weight, pred, actual, actual_w), axis=1)
    df.columns = ['N', 'weight', 'pred_min', 'pred_mean', 'pred_max',
                  'precision', 'responders', 'precision_w', 'responders_w']

    # total_obs and Event rates
    n_obs = np.sum(df['N'].values)
    n_obs_w = np.sum(df['weight'].values)
    n_resp = np.sum(y_true)
    n_resp_w = np.sum(weights*y_true)
    n_non_resp = n_obs - n_resp
    n_non_resp_w = n_obs_w - n_resp_w
    rate = np.mean(y_true)
    rate_w = np.average(y_true, weights=weights)

    # Cumulative obs, responders and non responders
    df['cum_responders'] = df['responders'].cumsum()
    df['cum_responders_w'] = df['responders_w'].cumsum()

    df['non_responders'] = df['N'] - df['responders']
    df['non_responders_w'] = df['weight'] - df['responders_w']
    df['cum_non_responders'] = df['non_responders'].cumsum()
    df['cum_non_responders_w'] = df['non_responders_w'].cumsum()

    # Cumulative precision
    df['cum_precision'] = df['cum_responders'] / df['N'].cumsum()
    df['cum_precision_w'] = df['cum_responders_w'] / df['weight'].cumsum()

    # Cumulative recall
    df['cum_recall'] = df['cum_responders'] / df['responders'].sum()
    df['cum_recall_w'] = df['cum_responders_w'] / df['responders_w'].sum()

    # Lift (unweighted and weighted)
    df['lift'] = df['cum_precision'] / rate
    df['lift_w'] = df['cum_precision_w'] / rate_w

    # F1 score calculation
    df['f1_score'] = df.apply(lambda x: harmonic_mean(x[['cum_precision', 'cum_recall']]), axis=1)
    df['f1_score_w'] = df.apply(lambda x: harmonic_mean(x[['cum_precision_w', 'cum_recall_w']]), axis=1)

    # Accuracy calculation
    df['accuracy'] = df.apply(lambda x: (x['cum_responders'] + n_non_resp - x['cum_non_responders'])/n_obs, axis=1)
    df['accuracy_w'] = df.apply(lambda x: (x['cum_responders_w'] + n_non_resp_w - x['cum_non_responders_w'])/n_obs_w,
                                axis=1)

    if int(np.unique(weights)[0]) == 1:

        cols = ['N', 'pred_min', 'pred_mean', 'pred_max','cum_responders', 'cum_precision',
                'cum_recall', 'lift', 'f1_score', 'accuracy']

    else:

        cols = ['N', 'weight', 'pred_min', 'pred_mean', 'pred_max', 'cum_responders', 'cum_responders_w',
                'cum_precision', 'cum_precision_w', 'cum_recall', 'cum_recall_w',
                'lift', 'lift_w', 'f1_score', 'f1_score_w', 'accuracy', 'accuracy_w']

    return df[cols]


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def stats_report(y_true: np.ndarray, y_pred: np.ndarray, bin_cutoffs: np.ndarray = None,
                 weights: np.ndarray = None, segment: np.ndarray = None, n: int = 20) -> pd.DataFrame:
    """
    Calculates statistics (min, mean, max) of response and predicted values bins (n).

    Samples are grouped into bins of equal size by ranking the predicted values in descending order. The
    relevant statistics are then computed for each semi-decile. In general this is expected to be more useful to
    evaluate output of regression output with continuous responses.

    :param y_true: Ground truth (correct) target values. Array-like of shape = (n_samples).
    :param y_pred: Estimated target values. Array-like of shape = (n_samples).
    :param bin_cutoffs: apply pre-defined bin cutoffs on y_pred
    :param weights: Weight given to each observation.
    :param segment: indicator variable for a segment.
    :param n: Number of bins. Integer. Default value is 10.
    :return: df: Report Data Frame.
    """

    _validate_report_args(y_true, y_pred, weights)

    # Create unit weights/segment if None
    if weights is None:
        weights = np.ones(len(y_true))
    if segment is None:
        segment = np.ones(len(y_true))

    # Stack arrays with responses and predictions
    arr = np.vstack((y_true, y_pred, weights, segment)).T

    # Convert to data frame
    df = pd.DataFrame(arr, columns=['y_true', 'y_pred', 'weight', 'segment'])

    # Weighted actual variable
    df['y_true_w'] = df['y_true'] * df['weight']

    if bin_cutoffs is None:

        # Create bins based on weighted ranked predictions
        quantiles = np.linspace(0, 1, n + 1)
        df.sort_values('y_pred', inplace=True, ascending=False)
        cum_weight = df['weight'].cumsum().values
        df['bin'] = pd.cut(cum_weight / cum_weight[-1], bins=quantiles, labels=False)

    else:

        # Create bins based on given bin cutoffs

        n = len(bin_cutoffs)
        choices = []
        for i in range(n):
            choices.append(i)

        conditions = []
        for cutoff in bin_cutoffs:
            conditions.append((df['y_pred'] >= cutoff))

        df['bin'] = np.select(conditions, choices, default=n - 1)

    check_true(df['bin'].nunique() == n, ValueError('Some bins have no samples. Choose a lower number of bins.'))

    # Filter for segment
    df = df[df.segment == 1].reset_index(drop=True)

    # Group samples by bin
    grp = df.groupby('bin')

    # Number of samples and total weight by bin
    N = grp.size().rename('N')
    weight = grp['weight'].sum()

    # Statistics by bin (min, weighted average, max)
    pred = grp['y_pred'].agg(['min', 'mean', 'max'])

    actual_min = grp['y_true'].min()

    actual_percentile_1 = grp['y_true'].agg([percentile(1)])
    actual_percentile_10 = grp['y_true'].agg([percentile(10)])
    actual_percentile_25 = grp['y_true'].agg([percentile(25)])

    actual_mean_w = grp['y_true_w'].sum() / weight
    actual_median = grp['y_true'].agg([percentile(50)])

    actual_percentile_75 = grp['y_true'].agg([percentile(75)])
    actual_percentile_90 = grp['y_true'].agg([percentile(90)])
    actual_percentile_99 = grp['y_true'].agg([percentile(99)])

    actual_max = grp['y_true'].max()

    # Convert to data frame
    df = pd.concat((N, weight, pred,
                    actual_min,
                    actual_percentile_1, actual_percentile_10, actual_percentile_25,
                    actual_mean_w, actual_median,
                    actual_percentile_75, actual_percentile_90, actual_percentile_99,
                    actual_max), axis=1)

    # Update column names
    if int(np.unique(weights)[0]) == 1:

        df.columns = ['N', 'weight', 'pred_min', 'pred_mean', 'pred_max',
                      'actual_min',
                      'actual_percentile_1', 'actual_percentile_10', 'actual_percentile_25',
                      'actual_mean', 'actual_median',
                      'actual_percentile_75', 'actual_percentile_90', 'actual_percentile_99',
                      'actual_max']

    else:
        df.columns = ['N', 'weight', 'pred_min', 'pred_mean', 'pred_max',
                      'actual_min',
                      'actual_percentile_1', 'actual_percentile_10', 'actual_percentile_25',
                      'actual_mean_w', 'actual_median',
                      'actual_percentile_75', 'actual_percentile_90', 'actual_percentile_99',
                      'actual_max']

    return df


def create_bins(df_index: np.ndarray, y_pred: np.ndarray, bin_cutoffs: np.ndarray = None,
                weights: np.ndarray = None, segment: np.ndarray = None, n: int = 20) -> pd.DataFrame:
    """
    Calculates cumulative precision, recall and lift by semi-decile.

    Samples are grouped into semi-deciles (bins) of equal size by ranking the predicted values in descending order. Used
    for classification problems.

    :param df_index: index of the dataframe for merging back bins.
    :param y_pred: Predicted probabilities. Array-like of shape = (n_samples).
    :param bin_cutoffs: apply pre-defined bin cutoffs on y_pred
    :param weights: Weight given to each observation.
    :param segment: indicator variable for a segment.
    :param n: Number of bins. Integer. Default value is 10.

    :return: df: Report Data Frame.
    """

    _validate_report_args(df_index, y_pred, weights)

    # Create unit weights/segment if None
    if segment is None:
        segment = np.ones(len(y_pred))
    if weights is None:
        weights = np.ones(len(y_pred))

    # Stack arrays with df_indexes and predictions
    arr = np.vstack((df_index, y_pred, weights, segment)).T

    # Convert to data frame
    df = pd.DataFrame(arr, columns=['df_index', 'y_pred', 'weight', 'segment'])
    index_df = df[['df_index', 'segment']]

    if bin_cutoffs is None:

        # Create bins based on weighted ranked predictions
        quantiles = np.linspace(0, 1, n + 1)
        df.sort_values('y_pred', inplace=True, ascending=False)
        cum_weight = df['weight'].cumsum().values
        df['bin'] = pd.cut(cum_weight / cum_weight[-1], bins=quantiles, labels=False)

    else:

        # Create bins based on given bin cutoffs
        n = len(bin_cutoffs)
        choices = []
        for i in range(n):
            choices.append(i)

        conditions = []
        for cutoff in bin_cutoffs:
            conditions.append((df['y_pred'] >= cutoff))

        df['bin'] = np.select(conditions, choices, default=n - 1)

    # Filter for segment
    df = df[df.segment == 1].reset_index(drop=True)

    # merge back to retain original index
    df = index_df.merge(df, on='df_index', how='left')

    return df['bin']

# COMMAND ----------

def precision_recall(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    check_true(np.all(np.unique(y_pred) == [0, 1]), ValueError("y_pred must be a binary variable with 0/1 values."))
    check_true(np.all(np.unique(y_true) == [0, 1]), ValueError("y_true must be a binary response with 0/1 values."))

    arr = np.vstack((y_pred, y_true)).T
    df = pd.DataFrame(arr, columns=['y_pred', 'y_true'])

    grp = df.groupby('y_pred')
    N = grp.size().rename('N').reset_index()
    met = grp['y_true'].agg(['sum', 'mean']).reset_index()

    df = pd.concat((N, met[['sum', 'mean']]), axis=1)
    df.columns = ['y_pred', 'N', 'responders', 'precision']
    df['recall'] = df['responders'] / df['responders'].sum()
    df['f1_Score'] = 2*((df['precision'] * df['recall']) / (df['precision'] + df['recall']))
    df = df.sort_values(by=['f1_Score'], ascending=False)

    return df


def variable_profiling(df: pd.DataFrame, y_pred_var: str, varlist: list, bin_cutoffs: np.ndarray = None,
                       weight: str = None, n: int = 10) -> pd.DataFrame:
    """

    :param df: pandas dataframe with x variables (non scaled), weight variable (if any) and model scores
    :param y_pred_var: name of predicted score variable
    :param varlist: list of variables for creating profile
    :param bin_cutoffs: apply pre-defined bin cutoffs on y_pred_var
    :param weight: obs weight variable name (optional)
    :param n: number of equal sized bins (optional, default=10)
    :return:
    """

    df_var_profile = pd.DataFrame()
    bins = np.linspace(0, 1, n + 1)

    if bool(weight):
        pass
    else:
        df['weight'] = np.ones(df.shape[0])
        weight = 'weight'

    if bin_cutoffs is None:

        df.sort_values(y_pred_var, inplace=True, ascending=False)
        cum_weight = df[weight].cumsum().values
        df['bin'] = pd.cut(cum_weight / cum_weight[-1], bins=bins, labels=False)

    else:

        # Create bins based on given bin cutoffs
        n = len(bin_cutoffs)
        choices = []
        for i in range(n):
            choices.append(i)

        conditions = []
        for cutoff in bin_cutoffs:
            conditions.append((df[y_pred_var] >= cutoff))

        df['bin'] = np.select(conditions, choices, default=n - 1)

    for i, var in enumerate(varlist):

        a = pd.DataFrame(df.groupby(['bin'])[var].agg(['mean']))
        a.reset_index(drop=False, inplace=True)
        a.columns = ['bin', var]
        if i == 0:
            df_var_profile = a
        else:
            df_var_profile = pd.merge(df_var_profile, a, on='bin')

    bin_min_score = df.groupby(['bin'])[y_pred_var].agg(['min', 'count'])
    bin_min_score.rename(columns={'min': 'pred_min'}, inplace=True)
    df_var_profile = pd.concat([bin_min_score, df_var_profile], axis=1)

    return df_var_profile


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False) -> pd.DataFrame:
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.unique(np.interp(quantiles, weighted_quantiles, values))

# weighted_quantile(df[varlist_t[0]], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], df['weight'])


def explained_variance(data: pd.DataFrame, varlist: list, sel_varlist: list) -> dict:
    """
        Attributes
        ----------
        data:
            input dataframe
        varlist:
            variable list passed as an input to the algorithm
        sel_varlist:
            selected variables from the algorithm

        Returns
        -------
        A dictionary
    """
    all_vars = list(set(varlist + sel_varlist))

    X = data[all_vars].values
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    corr_mat_df = pd.DataFrame(np.dot(X.T, X) / X.shape[0], index=all_vars, columns=all_vars)

    keep_index = np.where(np.nan_to_num(np.diagonal(corr_mat_df)) != 0)[0].tolist()
    corr_mat_df = corr_mat_df.iloc[keep_index, keep_index]
    to_drop = list(set(varlist) - set(corr_mat_df.columns.tolist()))
    corr_mat_abs = corr_mat_df.abs()
    upper = corr_mat_abs.where(np.triu(np.ones(corr_mat_abs.shape), k=1).astype(np.bool))
    to_drop = [var for var in upper.columns if any(upper[var] > 1)] + to_drop
    to_keep = [var for var in corr_mat_df.columns if var not in to_drop]
    sel_varlist = [var for var in sel_varlist if var in to_keep]

    inv = np.linalg.pinv(corr_mat_df.loc[sel_varlist, sel_varlist])
    var_exp_n = np.trace(np.dot(np.dot(corr_mat_df.loc[:, sel_varlist], inv), corr_mat_df.loc[sel_varlist, :]))
    var_exp_d = float(corr_mat_df.shape[0])
    var_exp = var_exp_n / var_exp_d

    return {'exp_exp': np.around(var_exp_n, decimals=3),
            'total_var': np.around(var_exp_d, decimals=3),
            'ratio': np.around(var_exp, decimals=3),
            'vars_dropped': to_drop}


def variance_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: input dataframe with selected variables
    :return: VIF report as pandas dataframe
    """
    start = time()
    sel_varlist = df.columns.tolist()
    print('No. of features = ' + str(len(sel_varlist)))
    x = df.values
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    corr_mat = pd.DataFrame(np.dot(x.T, x) / x.shape[0], index=sel_varlist, columns=sel_varlist)
    vif_list = []
    for i, var in enumerate(sel_varlist):
        x_list = list(set(sel_varlist) - set([var]))
        x_list_inv = np.linalg.pinv(corr_mat.loc[x_list, x_list])
        r2 = np.trace(np.dot(np.dot(corr_mat.loc[[var], x_list], x_list_inv), corr_mat.loc[x_list, [var]]))
        if r2 == 1.00:
            vif = round((1 / (1 - 0.999)), 2)
        else:
            vif = round((1 / (1 - r2)), 2)
        vif_list.append([var, vif])
    print("Runtime is %s minutes " % round((time() - start) / 60.0, 2))
    return pd.DataFrame(vif_list, columns=['feature', 'vif']).sort_values(by='vif', ascending=False,
                                                                          inplace=False)


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None):
    """

    :param y_true:
    :param y_pred:
    :param weight:
    :return:
    """

    if weight is None:
        weight = np.ones(y_true.shape[0])
    else:
        pass

    rmse = np.sqrt(np.average(np.square(y_pred - y_true), axis=-1, weights=weight))

    y_true_mean = np.average(y_true, weights=weight)
    erss = (weight * np.square(y_true - y_pred)).sum(axis=0)
    tss = (weight * np.square(y_true - y_true_mean)).sum(axis=0)
    r2 = 1 - (erss / tss)

    return rmse, r2


def sample_size_estimator_proportions(
        model_precision: float,
        model_obs: int,
        no_model_precision: float,
        no_model_obs: int,
        treatment_control_obs_ratio: float,
        se_effect: float = 1.0,
        power: float = 0.99,
        alpha: float = 0.01) -> NoReturn:
    """
    :param model_precision: Event rate above model decided cutoff
    :param model_obs: # obs above model decided cutoff
    :param no_model_precision: Event rate in case of no model (BAU)
    :param no_model_obs: # obs in case of no model (BAU)
    :param treatment_control_obs_ratio: Required ratio of treatment vs control obs when in use
    :param se_effect: Required standard error effect size (based on standard normal distn of Model - No_Model)
    :param power: Required power of the test (1-Beta)
    :param alpha: Required alpha of the test - Significance level
    :return: No Return
    """

    # calculate p and se
    p = (model_precision - no_model_precision)
    se = np.sqrt((model_precision * (1 - no_model_precision)) * ((1 / model_obs) + (1 / no_model_obs)))

    dist_df = pd.DataFrame(zip(binomial(model_obs, model_precision, size=(100000,)),
                               binomial(no_model_obs, no_model_precision, size=(100000,)),
                               normal(loc=p, scale=se, size=(100000,))),
                           columns=['Model', 'No_Model', 'Model - No_Model'])

    dist_df[['Model', 'No_Model']].plot(kind='density')
    dist_df[['Model - No_Model']].plot(kind='density')

    # perform power analysis
    analysis = TTestIndPower()
    treatment_obs = np.ceil(analysis.solve_power(effect_size=se_effect,
                                                 power=power, nobs1=None,
                                                 ratio=treatment_control_obs_ratio,
                                                 alpha=alpha))

    control_obs = np.ceil(treatment_obs * treatment_control_obs_ratio)
    total_obs = treatment_obs + control_obs

    print("Sample Size for Treatment: %.0f" % treatment_obs,
          "\nSample Size for Control: %.0f" % control_obs,
          "\nTotal Sample Size: %.0f" % total_obs)

# COMMAND ----------

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))