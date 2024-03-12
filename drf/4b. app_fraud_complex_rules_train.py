# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %pip install catboost

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

print(df_train.shape, df_test.shape, df_oot.shape)

train_dataset, test_dataset, oot_dataset= df_train.copy(), df_test.copy(), df_oot.copy()

train_labels = train_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)
test_labels = test_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)
oot_labels = oot_dataset[['is_app_fraud', 'app_fraud_amount']].apply(pd.to_numeric)


# COMMAND ----------

train_dataset.columns

# COMMAND ----------

features = ['age_at_completion',
       'applicant_id_type', 
       'rc_rule_in_fbr', 'fraud_fail', 'applicant_idcountry_issue',
       #'applicant_nationality', 
       'is_restricted_keyword_present',
       'applicant_postcode', #'company_sic', 
       'company_icc', 'company_type',
       'applicant_device_type', 'applicant_email_domain',
       'company_age_at_completion', 'individual_blocklist',
       'individual_identity_address', 'individual_sanctions_pep',
       'individual_id_scan', 'business_internal_checks',
       'count_failed_business_rules',
       'applicant_years_to_id_expiry', 'company_directors_count',
       'company_structurelevelwise', 'company_postcode',
       'company_postcode_risk', 'company_status',
       'directors_avg_age_at_completion']
train_dataset[features].info()

# COMMAND ----------

train_dataset[features].head()

# COMMAND ----------

X_train = train_dataset[features]
y_train = train_labels['is_app_fraud']
X_test = test_dataset[features]
y_test = test_labels['is_app_fraud']
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
clf = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, loss_function='Logloss', 
                         cat_features=train_dataset[features].select_dtypes(include=['object']).columns.tolist(),
                         class_weights=class_weights)
clf.fit(X_train, y_train, verbose=False)

# COMMAND ----------


y_probs = clf.predict_proba(X_test)[:, 1]
custom_threshold = 0.85
y_pred = (y_probs > custom_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# COMMAND ----------

# Get feature importances
feature_importances = clf.get_feature_importance()
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

# COMMAND ----------


def get_top_n_largest(df, column_name, metric ='mean', n=5):
    result_nlargest = df.groupby([column_name]).agg({'is_app_fraud': ['count', 'sum', 'mean']}).nlargest(n, ('is_app_fraud', metric))
    nlargest_list = result_nlargest.index.get_level_values(column_name).tolist()
    print(nlargest_list)
    return result_nlargest

# COMMAND ----------

for f in features:
  print("*"*10,f)
  print(get_top_n_largest(df_train, f,'sum', 10))

# COMMAND ----------


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

id_list = ['Other_ID', 'Provisional_Licence', 'Undefined']
country_list = ['GB', 'RO', 'IN', 'PT', 'SK']
icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']

rule_filter=(df_train.applicant_id_type.isin(id_list)) & (df_train.applicant_idcountry_issue.isin(country_list)) &  (df_train.company_icc.isin(icc_list))
predict_fraud_rate(df_train, rule_filter)

# COMMAND ----------

postcode_list = ['B', 'E', 'M', 'CV', 'IG', 'N', 'LS', 'SE', 'IP', 'NE']
rule_filter=(df_train.applicant_postcode.isin(postcode_list)) & (df_train.individual_identity_address.isin([1])) &  (df_train.company_icc.isin(icc_list))
predict_fraud_rate(df_train, rule_filter)

# COMMAND ----------

icc_list =['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services', 'category.online_retailer_/_online_shop_/_ecommerce', 'category.industrial_cleaning_services', 'category.land_freight_transport', 'category.domestic_cleaner', 'category.retail_of_textiles,_clothes_&_footwear']
rule_filter=(df_train.company_icc.isin(icc_list)) & (df_train.age_at_completion.lt(25)) & (df_train.applicant_idcountry_issue.isin(country_list))
predict_fraud_rate(df_train, rule_filter)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

