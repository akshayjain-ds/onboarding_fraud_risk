# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %pip install graphviz
# MAGIC %pip install catboost
# MAGIC %pip install ipywidgets

# COMMAND ----------

import sys
import pandas as pd
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



# COMMAND ----------

df = pd.read_csv("/dbfs/dbfs/Shared/Decisioning/Strawberries/App_fraud/app_fraud_feature_encoded_dataset.csv",
                 dtype={"company_id": "str", "membership_id": "str"})
df.set_index('company_id', inplace=True)
df['company_created_on'] = pd.to_datetime(df['company_created_on']).apply(lambda x: x.date())
from ast import literal_eval
df['company_sic'] = df['company_sic'].apply(lambda x: literal_eval(x))
df['applicant_nationality'] = df['applicant_nationality'].apply(lambda x: literal_eval(x))
print(df.shape)
nationality_count = df['applicant_nationality'].apply(lambda x: len(x)).max()
sic_count = df['company_sic'].apply(lambda x: len(x)).max()
nationality_count, sic_count

# COMMAND ----------

df.head()

# COMMAND ----------

test = (pd.to_datetime('2022-12-01') <= pd.to_datetime(df.company_created_on)) & (pd.to_datetime(df.company_created_on) < pd.to_datetime('2023-02-01'))
oot = pd.to_datetime('2023-02-01') <= pd.to_datetime(df.company_created_on)
df_test = df[test]
df_oot = df[oot]
df_train = df[~test & ~oot]
print(df_train.shape, df_test.shape, df_oot.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_dataset, test_dataset, oot_dataset= df_train.copy(), df_test.copy(), df_oot.copy()

train_labels = train_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)
test_labels = test_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)
oot_labels = oot_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)

w_train = train_dataset['is_approved']
w_test = test_dataset['is_approved']
w_oot = oot_dataset['is_approved']

wv_train = pd.Series(scaler.fit_transform(w_train.values.reshape(-1,1) + scaler.fit_transform(train_labels[['app_fraud_amount']])).flatten(), index=w_train.index)
wv_train = wv_train * (wv_train.shape[0]/wv_train.sum())

print(train_dataset.shape, train_labels.shape, w_train.shape, wv_train.shape, "\n",
      test_dataset.shape, test_labels.shape, w_test.shape, "\n",
      oot_dataset.shape, oot_labels.shape, w_oot.shape) 

# COMMAND ----------

# MAGIC %md
# MAGIC # Rule : Email contains numbers and domain is outlook

# COMMAND ----------

print(pd.concat([train_dataset[train_dataset.is_approved == 1][['applicant_email_numeric','applicant_email_domain']], train_labels], axis=1).shape)
print(train_dataset[train_dataset.is_approved == 1][['applicant_email_numeric','applicant_email_domain']].shape)


# COMMAND ----------

rule_data = train_dataset[train_dataset.is_approved == 1][['applicant_email_numeric', 'applicant_email_domain']].merge(
    train_labels, left_index=True, right_index=True
)
rule_data['predicted_fraud'] = (rule_data['applicant_email_numeric'] == 1) & (rule_data['applicant_email_domain'] == 'outlook.com')

cm = confusion_matrix(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
print(cm)

true_negative_count, false_positive_count, false_negative_count, true_positive_count = cm.ravel()
true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
    rule_data.loc[(rule_data['predicted_fraud'] == 0) & (rule_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 1) & (rule_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 0) & (rule_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 1) & (rule_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
)


amount_matrix = pd.DataFrame({
    'True Positive Amount': [true_positive_amount],
    'False Positive Amount': [false_positive_amount],
    'True Negative Amount': [true_negative_amount],
    'False Negative Amount': [false_negative_amount]
})

print("\nAmount Matrix:")
print(amount_matrix)
precision = precision_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
recall = recall_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
f1 = f1_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
print(f"\nRecall is {round(recall*100,2)}")
recall_rate = true_positive_count / (true_positive_count + false_negative_count)
recall_amount = true_positive_amount / (true_positive_amount + false_negative_amount)
print(f"\nRecall by member: {round(recall_rate * 100, 2)}%")
print(f"Recall by value: {round(recall_amount * 100, 2)}%")

false_positive_rate = false_positive_count / (false_positive_count + true_negative_count)
false_positive_rate_amount = false_positive_amount / (false_positive_amount + true_negative_amount)
print(f"\n False Positive Rate by member: {round(false_positive_rate * 100, 2)}%")
print(f"False Positive Rate by value: {round(false_positive_rate_amount * 100, 2)}%")


# COMMAND ----------

# MAGIC %md
# MAGIC # Rule : Age of member and age of company
# MAGIC

# COMMAND ----------

rule_data = merged_data = train_dataset[train_dataset.is_approved == 1][['age_at_completion', 'company_age_at_completion']].merge(
    train_labels, left_index=True, right_index=True
)
rule_data = rule_data.dropna()
rule_data['count'] = 1
print(rule_data.shape)

results = []

def predict_fraud_rate(df, threshold_age, threshold_company_age):
    filtered_data = df.copy()

    condition_age = filtered_data['age_at_completion'] <= threshold_age
    condition_company_age = filtered_data['company_age_at_completion'] <= threshold_company_age

    filtered_data['predicted_fraud'] = np.where((condition_age) & (condition_company_age), 1, 0)

    total_samples = filtered_data[filtered_data['predicted_fraud'] == 1].shape[0]
    total_frauds = filtered_data[(condition_age) & (condition_company_age)]['is_app_fraud'].sum()

    if total_samples == 0:
        print(f"\nNo samples meet the specified conditions for age <= {threshold_age} and company_age <= {threshold_company_age}.")
        return None
    
    fraud_rate = round(filtered_data[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1)].shape[0] * 100 / filtered_data[(filtered_data['predicted_fraud'] == 1)].shape[0], 2)

    #cm = confusion_matrix(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    #true_negative_count, false_positive_count, false_negative_count, true_positive_count = cm.ravel()

    true_negative_count, false_positive_count, false_negative_count, true_positive_count = (
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'count'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'count'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'count'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'count'].sum()
    )
    true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
        filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
    )

    recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
    recall_amount = round(true_positive_amount * 100 / (true_positive_amount + false_negative_amount), 2)
    precision = precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    recall = recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    false_positive_rate = round(false_positive_count * 100 / (false_positive_count + true_negative_count), 2)
    false_positive_rate_amount = round(false_positive_amount * 100 / (false_positive_amount + true_negative_amount), 2)

    result = {
        'Age Threshold': threshold_age,
        'Company Age Threshold': threshold_company_age,
        'Sample size': total_samples,
        'APPF' : total_frauds,
        'Fraud Rate': fraud_rate,
        'Recall': recall,
        'Recall Rate (by member)': recall_rate,
        'Recall Rate (by value)': recall_amount,
        'False Positive Rate': false_positive_rate,
        'Precision': precision,
        'F1': f1,
            'TP' : true_positive_count,
            'FP' : false_positive_count,
            'TN' : true_negative_count,
            'FN' : false_negative_count,
            'Total' : true_positive_count + false_positive_count + true_negative_count + false_negative_count,
    }

    return result


max_age = 30 #int(train_dataset['age_at_completion'].max())
max_company_age = 12 #int(train_dataset['company_age_at_completion'].max())


age_thresholds = list(range(20, max_age + 1, 1))
company_age_thresholds = list(range(1, max_company_age + 1, 1))

# age_thresholds = [25, 30, 35, 40, 45, 50, 55, 60, 65]
# company_age_thresholds = [6, 12, 24, 36, 48, 60]

for age_threshold in age_thresholds:
    for company_age_threshold in company_age_thresholds:
        result = predict_fraud_rate(rule_data, age_threshold, company_age_threshold)
        if result is not None:
            results.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by='Recall Rate (by value)')


# COMMAND ----------

# MAGIC %md
# MAGIC # UNIVARIATE FEATURE VIEW

# COMMAND ----------

candidates = ['company_structurelevelwise',
               'company_type', 'fraud_fail','individual_identity_address',
              'applicant_postcode_risk',
 'applicant_idcountry_issue_risk',
 'applicant_nationality_risk',
 'company_keywords_risk',
 'company_type_risk',
 'company_nob_risk',
 'company_postcode_risk',
  ]
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

rule_data = train_dataset[train_dataset.is_approved == 1][candidates].merge(
    train_labels, left_index=True, right_index=True
)
def predict_fraud_rate_univariate(df, candidates):
    results = []
    for categorical_column in candidates:
        for value in df[categorical_column].unique():
            filtered_data = df.copy()
            filtered_data['count'] = 1
            condition = filtered_data[categorical_column]==value
            filtered_data['predicted_fraud'] = np.where(condition, 1, 0)
            total_samples = filtered_data['predicted_fraud'].sum()
            total_frauds = filtered_data['is_app_fraud'].sum()
            
            if total_samples == 0:
                continue  

            fraud_rate = round(filtered_data[(filtered_data['predicted_fraud']==1) & (filtered_data['is_app_fraud']==1)].shape[0]* 100 / filtered_data[(filtered_data['predicted_fraud']==1)].shape[0], 2)

            precision = precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
            recall = recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
            f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
            cm = confusion_matrix(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
            true_negative_count, false_positive_count, false_negative_count, true_positive_count = cm.ravel()
            true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
            filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
            filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
            filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
            filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
        )
         
            recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
            recall_amount = round(true_positive_amount*100 / (true_positive_amount + false_negative_amount), 2)

            false_positive_rate = round(false_positive_count*100 / (false_positive_count + true_negative_count), 2)
            false_positive_rate_amount = round(false_positive_amount*100 / (false_positive_amount + true_negative_amount), 2)

            result = {
                'Feature' : categorical_column,
                'Category': value,
                'Sample Size': total_samples,
                'APPF' : total_frauds,
                'Fraud Rate': fraud_rate,
                'Recall' : recall,
                'Recall (member)': recall_rate,
                'Recall (value)' : recall_amount,
                'Precision':precision,
                'F1 score': f1,
                'False Positive Rate': false_positive_rate,
                # 'TP' : true_positive_count,
                # 'FP' : false_positive_count,
                # 'TN' : true_negative_count,
                # 'FN' : false_negative_count,
                # 'Total' : true_positive_count+false_positive_count+true_negative_count+false_negative_count,

            }
            results.append(result)

    return pd.DataFrame(results)

univariate_df = predict_fraud_rate_univariate(rule_data, candidates)
univariate_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # BIVARIATE FEATURE VIEW

# COMMAND ----------

from itertools import combinations

candidates = ['company_structurelevelwise', 'company_type', 'fraud_fail', 'individual_identity_address',
              'applicant_postcode_risk',
              'applicant_idcountry_issue_risk',
              'applicant_nationality_risk',
              'company_keywords_risk',
              'company_type_risk',
              'company_nob_risk',
              'company_postcode_risk']

combo_features = [(x, y) for x, y in combinations(candidates, 2) if x != y]


# COMMAND ----------


def predict_fraud_rate_categorical(df, candidates):
    results = []
    for features in candidates:
        categorical_column1, categorical_column2 = features
        for value1 in df[categorical_column1].unique():
            for value2 in df[categorical_column2].unique():
                filtered_data = df.copy()
                condition = (filtered_data[categorical_column1] == value1) & (filtered_data[categorical_column2] == value2)
                filtered_data['predicted_fraud'] = np.where(condition, 1, 0)
                total_samples = filtered_data['predicted_fraud'].sum()
                total_frauds = filtered_data[condition]['is_app_fraud'].sum()

                if total_samples == 0:
                    continue  

                fraud_rate = round(filtered_data[(filtered_data['predicted_fraud']==1) & (filtered_data['is_app_fraud']==1)].shape[0] * 100 / filtered_data[(filtered_data['predicted_fraud']==1)].shape[0], 2)

                cm = confusion_matrix(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
                true_negative_count, false_positive_count, false_negative_count, true_positive_count = cm.ravel()
                true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
                    filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
                    filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
                    filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
                    filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
                )
                recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
                recall_amount = round(true_positive_amount * 100 / (true_positive_amount + false_negative_amount), 2)

                precision = precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
                recall = recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
                f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])

                false_positive_rate = round(false_positive_count * 100 / (false_positive_count + true_negative_count), 2)
                false_positive_rate_amount = round(false_positive_amount * 100 / (false_positive_amount + true_negative_amount), 2)

                result = {
                    'Feature1': categorical_column1,
                    'Value1': value1,
                    'Feature2': categorical_column2,
                    'Value2': value2,
                    'Sample Size': total_samples,
                    'APPF': total_frauds,
                    'Fraud Rate': fraud_rate,
                    'Recall' : recall,
                'Recall (member)': recall_rate,
                'Recall (value)' : recall_amount,
                'Precision':precision,
                'F1 score': f1,
                'False Positive Rate': false_positive_rate,
                    # 'TP': true_positive_count,
                    # 'FP': false_positive_count,
                    # 'TN': true_negative_count,
                    # 'FN': false_negative_count,
                    # 'Total': true_positive_count + false_positive_count + true_negative_count + false_negative_count,
                }
                results.append(result)

    return pd.DataFrame(results)
bivariate_df = predict_fraud_rate_categorical(df = rule_data, candidates = combo_features)

# COMMAND ----------

bivariate_df.head()

# COMMAND ----------

location = "/Workspace/Shared/Decisioning/Strawberries/uk_app_fraud_model/app_fraud_engine_training_v2/artefacts/"
univariate_df.to_csv(location+"univariate_rule.csv", index = False)
bivariate_df.to_csv(location+"bivariate_rule.csv", index = False)

# COMMAND ----------

df_test = rule_data.copy()
condition1 = (rule_data['company_type']=="sole-trader") & (rule_data['applicant_idcountry_issue_risk']=="High")
condition2 = (rule_data['applicant_idcountry_issue_risk']=="High") & (rule_data['company_type_risk']=="High") 
df_test['condition1'] = np.where(condition1, 1,0)
df_test['condition2'] = np.where(condition2, 1, 0)
pd.crosstab(df_test['condition1'], df_test['condition2'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Narrow down list of countries ['IR', 'AF', 'GW', 'TJ', 'LA', 'MZ', 'ML', 'UG', 'KH', 'TZ', 'KE', 'LR', 'MM', 'NP', 'BF', 'PY', 'HT', 'VN', 'ZM', 'ST', 'NE', 'BJ', 'BO', 'LS', 'LK', 'SL', 'LB', 'VU', 'SD', 'PA', 'CV', 'MR', 'NG', 'GH', 'TT', 'ZW', 'YE', 'MH', 'GM', 'RW', 'AR', 'DO', 'TR', 'TH', 'NI', 'PK', 'JM', 'NA', 'AO', 'VE', 'CN', 'UA', 'CI', 'DZ', 'TL', 'KZ', 'MA', 'EC', 'TN', 'KG', 'ID', 'SN', 'GY', 'RU', 'PH', 'BR', 'GT', 'PG', 'MN', 'MY', 'UZ', 'AE', 'GD', 'BW', 'KW', 'SA', 'RO', 'LV'] in the rule : sole trader and high risk countries
# MAGIC

# COMMAND ----------


cols_to_keep = ['company_type', 'applicant_idcountry_issue_risk', 'applicant_idcountry_issue']
df_test = train_dataset[train_dataset.is_approved == 1][cols_to_keep].merge(
    train_labels, left_index=True, right_index=True
)
print(df_test.columns)
condition_ = (rule_data['company_type']=="sole-trader") & (rule_data['applicant_idcountry_issue_risk']=="High")
df_test = df_test[condition_]
risky_countries = predict_fraud_rate_univariate(df_test, candidates = ['applicant_idcountry_issue']).sort_values('F1 score', ascending=False)
risky_countries


# COMMAND ----------

train_dataset[['company_sic', 'company_icc', 'company_nob_risk']].sample(9)

# COMMAND ----------

cols_to_keep = ['company_icc', 'applicant_idcountry_issue']
df_test = train_dataset[(train_dataset.is_approved == 1) & (train_dataset.company_nob_risk=='High') & (train_dataset.applicant_idcountry_issue_risk=='High')][cols_to_keep].merge(
    train_labels, left_index=True, right_index=True
)
print(df_test.columns)
df_rule_test = predict_fraud_rate_categorical(df_test, candidates = [('applicant_idcountry_issue', 'company_icc')]).sort_values('F1 score', ascending=False)
df_rule_test.head()

# COMMAND ----------

df_rule_test.to_csv()

# COMMAND ----------

df_rule_test.to_csv(location+"rule_test_country_icc.csv", index = False)


# COMMAND ----------

nob_list = ['category.construction'
'category.domestic_cleaner',
'category.cleaner',
'category.household_cleaning_services',
'category.transport_and_storage',
'category.industrial_cleaning_services']

rule_data = train_dataset[train_dataset.is_approved == 1][['company_icc', 'applicant_idcountry_issue']].merge(
    train_labels, left_index=True, right_index=True
)
rule_data['predicted_fraud'] = (rule_data['company_icc'].isin(nob_list)) & (rule_data['applicant_idcountry_issue'] == 'RO')

cm = confusion_matrix(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
print(cm)

true_negative_count, false_positive_count, false_negative_count, true_positive_count = cm.ravel()
true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
    rule_data.loc[(rule_data['predicted_fraud'] == 0) & (rule_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 1) & (rule_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 0) & (rule_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
    rule_data.loc[(rule_data['predicted_fraud'] == 1) & (rule_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
)


amount_matrix = pd.DataFrame({
    'True Positive Amount': [true_positive_amount],
    'False Positive Amount': [false_positive_amount],
    'True Negative Amount': [true_negative_amount],
    'False Negative Amount': [false_negative_amount]
})

print("\nAmount Matrix:")
print(amount_matrix)
precision = precision_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
recall = recall_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
f1 = f1_score(rule_data['is_app_fraud'], rule_data['predicted_fraud'])
print(f"\nRecall is {round(recall*100,2)}")
recall_rate = true_positive_count / (true_positive_count + false_negative_count)
recall_amount = true_positive_amount / (true_positive_amount + false_negative_amount)
print(f"\nRecall by member: {round(recall_rate * 100, 2)}%")
print(f"Recall by value: {round(recall_amount * 100, 2)}%")

false_positive_rate = false_positive_count / (false_positive_count + true_negative_count)
false_positive_rate_amount = false_positive_amount / (false_positive_amount + true_negative_amount)
print(f"\n False Positive Rate by member: {round(false_positive_rate * 100, 2)}%")
print(f"False Positive Rate by value: {round(false_positive_rate_amount * 100, 2)}%")



# COMMAND ----------

