# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %pip install catboost hiplot

# COMMAND ----------

import sys
import os
import hiplot as hip
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import tempfile
import warnings
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")
from tqdm import tqdm, tqdm_notebook
from typing import Tuple
from sklearn.calibration import CalibratedClassifierCV
tqdm.pandas()
from sklearn import metrics
import json
from sklearn.model_selection import (
  KFold,
  StratifiedKFold,
  TimeSeriesSplit,
)
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import (
  roc_curve,
  auc,
  precision_recall_curve,
  PrecisionRecallDisplay
)
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import category_encoders.wrapper as cew
import copy

from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler


def predict_fraud_rate(df, rule_filter):
    filtered_data = df.copy()
    filtered_data['predicted_fraud'] = np.where(rule_filter, 1, 0)

    total_records = filtered_data.shape[0]
    total_samples = filtered_data[filtered_data['predicted_fraud'] == 1].shape[0]
    total_frauds = filtered_data[rule_filter]['is_app_fraud'].sum()

    if total_samples == 0:
        print(f"No samples meet the specified conditions.")
        return None
    
    fraud_rate = round(filtered_data[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1)].shape[0] * 100 / filtered_data[(filtered_data['predicted_fraud'] == 1)].shape[0], 2)

    true_negative_count, false_positive_count, false_negative_count, true_positive_count = (
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count()
    )

    true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
    )

    recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
    recall_amount = round(true_positive_amount * 100 / (true_positive_amount + false_negative_amount), 2)
    precision = round(precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100, 2)
    recall = round(recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100,2)
    f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    false_positive_rate = round(false_positive_count * 100 / (false_positive_count + true_negative_count), 2)
    false_positive_rate_amount = round(false_positive_amount * 100 / (false_positive_amount + true_negative_amount), 2)

    result = {
        'Total Records': total_records,
        'Total alerts': total_samples,
        'APPF' : total_frauds,
        'Alert Rate' :  round(total_samples*100/total_records,2),
        'Precision (True fraud alert rate)': precision,
        'Recall/True Positive rate (by member)': recall_rate,
        'Recall Rate (by value)': recall_amount,
        'False Positive Rate': false_positive_rate,
        'F1': f1,
            'TP' : true_positive_count,
            'FP' : false_positive_count,
            'TN' : true_negative_count,
            'FN' : false_negative_count,
            'Total' : true_positive_count + false_positive_count + true_negative_count + false_negative_count,
    }

    return pd.DataFrame(data=[result])


# COMMAND ----------

df = pd.read_csv("/dbfs/dbfs/Shared/Decisioning/Strawberries/App_fraud/app_fraud_feature_encoded_dataset.csv",
                 dtype={"company_id": "str", "membership_id": "str"})
df.set_index('company_id', inplace=True)
df['company_created_on'] = pd.to_datetime(df['company_created_on']).apply(lambda x: x.date())
from ast import literal_eval
df['company_sic'] = df['company_sic'].apply(lambda x: literal_eval(x))
df['applicant_nationality'] = df['applicant_nationality'].apply(lambda x: literal_eval(x))
print(df.shape)
df.head()

# COMMAND ----------

test = (pd.to_datetime('2022-12-01') <= pd.to_datetime(df.company_created_on)) & (pd.to_datetime(df.company_created_on) < pd.to_datetime('2023-02-01')) & (df.is_approved ==1)
oot = (pd.to_datetime('2023-02-01') <= pd.to_datetime(df.company_created_on)) & (df.is_approved ==1)

df_test = df[test]
df_oot = df[oot]

df_train = df[~test & ~oot & (df.is_approved ==1)]
df_train = df_train.fillna(0)
print(df_train.shape, df_test.shape, df_oot.shape)



# COMMAND ----------

features = ['age_at_completion',
       'applicant_id_type', 
       'rc_rule_in_fbr', 'fraud_fail', 'applicant_idcountry_issue',
       'applicant_nationality_0', 
       'is_restricted_keyword_present',
       'applicant_postcode', #'company_sic', 
       'company_icc', 'company_type',
       'applicant_device_type', 'applicant_email_domain',
       'applicant_email_numeric', 'applicant_name_mismatch_ind',
       'company_age_at_completion', 'individual_blocklist',
       'individual_identity_address', 'individual_sanctions_pep',
       'individual_id_scan', 'business_internal_checks',
       'count_failed_business_rules',
       'applicant_years_to_id_expiry', 'company_directors_count',
       'company_structurelevelwise', 'company_postcode',
       'company_postcode_risk', 'company_status',
       'directors_avg_age_at_completion']
numeric_features = df_train[features].select_dtypes(include=['int', 'float']).columns.to_list()
non_numeric_features = df_train[features].select_dtypes(include=['object']).columns.to_list()
print(numeric_features, non_numeric_features)
df_train[non_numeric_features].head()

# COMMAND ----------


original_data = df_train[features + ['is_app_fraud']]

# Estimate the average memory usage per row
avg_memory_per_row = original_data.memory_usage(index=True).sum()/(1024*1024*original_data.shape[0])
desired_chunk_size = 10
rows_per_chunk = int(desired_chunk_size / avg_memory_per_row)
rows_per_chunk = max(rows_per_chunk, 1)
num_chunks = int(len(original_data) / rows_per_chunk) + 1
print(num_chunks)

output_directory = "/Workspace/Shared/Decisioning/Strawberries/uk_app_fraud_model/app_fraud_engine_training_v2/artefacts/"
os.makedirs(output_directory, exist_ok=True)

for i in range(num_chunks):
    start_idx = i * rows_per_chunk
    end_idx = (i + 1) * rows_per_chunk
    chunk_data = original_data.iloc[start_idx:end_idx]
    output_filename = os.path.join(output_directory, f'chunk_{i + 1}.csv')
    chunk_data.to_csv(output_filename, index=False)

print(f'{num_chunks} chunks created successfully.')


# COMMAND ----------

data = [
       {'dropout':0.1, 
        'learning_rate': 0.001, 
        'optimizer': 'SGD', 
        'loss': 10.0
       },
       {'dropout':0.15, 
        'learning_rate': 0.01, 
        'optimizer': 'Adam', 
        'loss': 3.5
       },
       {'dropout':0.3, 
        'learning_rate': 0.1, 
        'optimizer': 'Adam', 
        'loss': 4.5
       }]
hip.Experiment.from_iterable(data).display(force_full_width=True)


# COMMAND ----------

