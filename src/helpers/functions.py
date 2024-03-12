# Databricks notebook source

import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime
import tecton
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
import pickle

SCOPE = "tecton"
SNOWFLAKE_DATABASE = "TIDE"
SNOWFLAKE_USER = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_USER")
SNOWFLAKE_PASSWORD = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_PASSWORD")
SNOWFLAKE_ACCOUNT = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_ACCOUNT")
SNOWFLAKE_ROLE = "DATABRICKS_ROLE"
SNOWFLAKE_WAREHOUSE = "DATABRICKS_WH"
CONNECTION_OPTIONS = dict(sfUrl=SNOWFLAKE_ACCOUNT,#"https://tv61388.eu-west-2.aws.snowflakecomputing.com/",
                           sfUser=SNOWFLAKE_USER,
                          sfPassword=SNOWFLAKE_PASSWORD,
                          sfDatabase=SNOWFLAKE_DATABASE,
                          sfWarehouse=SNOWFLAKE_WAREHOUSE,
                          sfRole=SNOWFLAKE_ROLE)

def get_spark():
    return SparkSession.builder.appName('strawberries').getOrCreate()

def spark_connector(query_string: str)-> DataFrame:
  return spark.read.format("snowflake").options(**CONNECTION_OPTIONS).option("query", query_string).load().cache()

def get_dataset(query_string):
  output = spark_connector(query_string)
  output = output.rename(columns={col: col.lower() for col in list(output.columns)})
  spine =  get_spark().createDataFrame(output)
  return spine

# COMMAND ----------

import sys
import tempfile
import warnings
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pycountry
warnings.filterwarnings("ignore")
from typing import Tuple
from sklearn.calibration import CalibratedClassifierCV
from tqdm.notebook import tqdm
tqdm.pandas()
from optbinning import OptimalBinning
from sklearn import metrics
from fuzzywuzzy import fuzz
import json
from sklearn.model_selection import (
  KFold,
  StratifiedKFold,
  TimeSeriesSplit,
)
from sklearn.metrics import recall_score
from sklearn.metrics import (
  roc_curve,
  auc,
  precision_recall_curve,
  PrecisionRecallDisplay
)
from sklearn.preprocessing import MinMaxScaler
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import category_encoders.wrapper as cew
import copy

def calculate_weight_decay(input_data_df_experiment: pd.DataFrame, date_col_name: str):
    """
    Calculates the decay for training the model with a temporal property
    decay_stretching = 0.001 came from testing, it makes data from 2 years ago to be
    worth about half of current data
    """
    
    date_col = pd.to_datetime(input_data_df_experiment[date_col_name])

    max_date = date_col.max()
    days_diff = date_col.apply(
        lambda x: (x - max_date).days)
    max_days = abs(days_diff).max()
    
    exp_days_diff = np.exp(days_diff/max_days)
    factor = input_data_df_experiment.shape[0]/exp_days_diff.sum()
    decay_weight = exp_days_diff*factor
    
    return decay_weight
  
def plot_roc_auc(fpr: np.ndarray, tpr: np.ndarray, data_split_type: str, color: str= 'b', log: bool = False):
    with tempfile.NamedTemporaryFile(prefix="roc_curve", suffix=".png") as roc_curve:
      roc_curve_file_name = roc_curve.name
       
      fig = plt.figure(1, figsize=(8, 8))
      plt.title('Receiver Operating Characteristics - Area Under Curve')
      plt.plot(fpr, tpr, color, label=f'{data_split_type} AUC = %0.2f' % metrics.auc(fpr, tpr))
      plt.legend(loc='lower right')
      plt.plot([0, 1], [0, 1], 'r--')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.ylabel('True Positive Rate')
      plt.xlabel('False Positive Rate')   
      if log:
        plt.savefig(roc_curve_file_name, bbox_inches = 'tight')
      if log:
        mlflow.log_artifact(roc_curve_file_name, "ROC Plot") 

def plot_precision_recall(precision: np.ndarray, recall: np.ndarray, data_split_type: str, color: str = 'b', log: bool = False):
    with tempfile.NamedTemporaryFile(prefix="precision_recall_curve", suffix=".png") as pr_curve:
      pr_curve_file_name = pr_curve.name
       
      fig = plt.figure(1, figsize=(8, 8))
      plt.title('Precision Recall Curve')
      plt.plot(recall, precision, color, label=f'{data_split_type} AUC = %0.2f' % metrics.auc(recall, precision))
      plt.legend(loc='lower right')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.ylabel('Precision')
      plt.xlabel('Recall')   
      if log:
        plt.savefig(pr_curve_file_name, bbox_inches = 'tight')
      if log:
        mlflow.log_artifact(pr_curve_file_name, "Precision recall Plot")


# COMMAND ----------

from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, SparkTrials
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb
from typing import Tuple

class HPOpt(object):

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, 
                 x_test: pd.DataFrame, y_test: pd.Series, 
                 w_train: pd.Series = None, w_test: pd.Series = None,
                 date_var_name: str = 'None'):
      
        lb = LabelBinarizer()
        self.x_train = x_train.drop(columns=[date_var_name]).values
        self.y_train  = lb.fit_transform(y_train)
        self.x_test = x_test.drop(columns=[date_var_name]).values
        self.y_test  = lb.transform(y_test)
        self.train_time_weights = calculate_weight_decay(x_train, date_var_name).values
        
        if isinstance(w_train, pd.Series) and isinstance(w_test, pd.Series):
          self.w_train = w_train.values
          self.w_test  = w_test.values
        else:
          print("warning: w_train/w_train not of type pd.Series")
          self.w_train = np.ones(self.y_train.shape[0])
          self.w_test  = np.ones(self.y_test.shape[0])
        
        self.model_name = None
        self.fn = None
          
    def process(self, fn_name, space, trials, algo, max_evals, random_state):
        
        self.model_name = fn_name
        self.fn = getattr(self, fn_name)

        try:
            result = fmin(fn=self.fn, space=space, algo=algo, max_evals=max_evals, trials=trials, rstate = random_state)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_model(self, para):
        reg = xgb.XGBClassifier(use_label_encoder=False, **para['param_grid'])
        return self.train_reg(reg, para)
    
    def lgb_model(self, para):
        reg = lgb.LGBMClassifier(**para['param_grid'])
        return self.train_reg(reg, para)
      
    def cb_model(self, para):
        reg = cb.CatBoostClassifier(**para['param_grid'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        
        train_weights = self.train_time_weights.copy() * self.w_train.copy()
        test_weights = self.w_train.copy()
        
        if self.model_name == 'xgb_model':
          reg.fit(self.x_train, self.y_train, sample_weight=train_weights,
                  eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                  sample_weight_eval_set = [self.w_train.copy(), self.w_test.copy()],
                  **para['fit_params'])
        
        elif self.model_name == 'lgb_model':
          reg.fit(self.x_train, self.y_train, sample_weight=train_weights,
                  eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                  eval_sample_weight = [self.w_train.copy(), self.w_test.copy()],
                  **para['fit_params'])

        elif self.model_name == 'ctb_model':
          train_eval_set = ctb.Pool(data=self.x_train, label=self.y_train, weight=self.w_train.copy())
          test_eval_set = ctb.Pool(data=self.x_test, label=self.y_test, weight=self.w_test.copy())
          reg.fit(self.x_train, self.y_train, sample_weight=train_weights,
                  eval_set=[train_eval_set, test_eval_set],
                  **para['fit_params'])

        else:
          raise Exception("Not Implemented")
        
        pred = reg.predict_proba(self.x_test)
        loss = para['loss_func'](self.y_test, pred, self.w_test.copy())
        return {'loss': loss, 'status': STATUS_OK, 'Trained_Model': reg}

# COMMAND ----------

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']

# COMMAND ----------

def calc_prec_recall_gt_threshold(risk_df, threshold):

  tp = risk_df[(risk_df.appf_rating_raw >= threshold) & (risk_df.is_app_fraud==1) & (risk_df.is_approved==1)]['weights'].sum()
  fp = risk_df[(risk_df.appf_rating_raw >= threshold) & (risk_df.is_app_fraud==0) & (risk_df.is_approved==1)]['weights'].sum()
  fn = risk_df[(risk_df.appf_rating_raw < threshold) & (risk_df.is_app_fraud==1) & (risk_df.is_approved==1)]['weights'].sum()
  tn = risk_df[(risk_df.appf_rating_raw < threshold) & (risk_df.is_app_fraud==0) & (risk_df.is_approved==1)]['weights'].sum()
  return (round(tp * 100.0 / (tp + fp + np.finfo(np.float).eps), 3), 
          round(tp * 100.0 / (tp + fn + np.finfo(np.float).eps), 3), 
          round(fp * 100.0 / (fp + tn + np.finfo(np.float).eps), 3), 
          round(risk_df[(risk_df.appf_rating_raw >= threshold)]['weights'].sum() * 100.0 / (risk_df['weights'].sum() + np.finfo(np.float).eps), 3))

def calc_prec_recall_lt_threshold(risk_df, threshold):
  
  tp = risk_df[(risk_df.appf_rating_raw < threshold) & (risk_df.is_app_fraud==1) & (risk_df.is_approved==1)]['weights'].sum()
  fp = risk_df[(risk_df.appf_rating_raw < threshold) & (risk_df.is_app_fraud==0) & (risk_df.is_approved==1)]['weights'].sum()
  fn = risk_df[(risk_df.appf_rating_raw >= threshold) & (risk_df.is_app_fraud==1) & (risk_df.is_approved==1)]['weights'].sum()
  tn = risk_df[(risk_df.appf_rating_raw >= threshold) & (risk_df.is_app_fraud==0) & (risk_df.is_approved==1)]['weights'].sum()
  
  return (round(tp * 100.0 / (tp + fp + np.finfo(np.float).eps), 3), 
          round(tp * 100.0 / (tp + fn + np.finfo(np.float).eps), 3),
          round(fp * 100.0 / (fp + tn + np.finfo(np.float).eps), 3),  
          round(risk_df[(risk_df.appf_rating_raw < threshold)]['weights'].sum() * 100.0 / (risk_df['weights'].sum() + np.finfo(np.float).eps), 3))

# COMMAND ----------

def cal_thresholds(model_decisions_df: pd.DataFrame, 
                   reject_min_precision: float,
                   reject_max_recall: float,
                   reject_max_group_size: float, 
                   approved_max_precision: float, 
                   approved_max_recall: float,
                   approved_max_group_size: float) -> dict:
  
  thresholds = {}
  
  # Calculate mkyc->reject threshold
  high_risk_threshold = model_decisions_df.appf_rating_raw.max()
  step_size = 1
  while True:
    precision, recall, fpr, group_size = calc_prec_recall_gt_threshold(model_decisions_df, high_risk_threshold)
    if precision >= reject_min_precision and recall <= reject_max_recall and group_size <= reject_max_group_size:
      high_risk_threshold -= step_size
      continue
    else:
      break
  thresholds["medium -> high"] = {"threshold": high_risk_threshold, "Precision": precision, "Recall": recall, "fpr": fpr, "group_size": group_size}

  # Calculate approved->mkyc risk threshold
  low_risk_threshold = model_decisions_df.appf_rating_raw.min()
  step_size = 1
  while True:
    precision, recall, fpr, group_size = calc_prec_recall_lt_threshold(model_decisions_df, low_risk_threshold)
    if precision <= approved_max_precision and recall <= approved_max_recall and group_size <= approved_max_group_size:
      low_risk_threshold += step_size
      continue
    else:
      break
  thresholds["low -> medium"] = {"threshold": low_risk_threshold, "Precision": precision, "Recall": recall, "fpr": fpr, "group_size": group_size}
  
  return thresholds
  