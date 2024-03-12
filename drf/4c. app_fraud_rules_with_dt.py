# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

import sys
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


# def predict_fraud_rate(df, rule_filter):
#     filtered_data = df.copy()
#     filtered_data['predicted_fraud'] = np.where(rule_filter, 1, 0)

#     total_records = filtered_data.shape[0]
#     total_samples = filtered_data[filtered_data['predicted_fraud'] == 1].shape[0]
#     total_frauds = filtered_data[rule_filter]['is_app_fraud'].sum()

#     if total_samples == 0:
#         print(f"No samples meet the specified conditions.")
#         return None
    
#     fraud_rate = round(filtered_data[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1)].shape[0] * 100 / filtered_data[(filtered_data['predicted_fraud'] == 1)].shape[0], 2)

#     true_negative_count, false_positive_count, false_negative_count, true_positive_count = (
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count()
#     )

#     true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
#         filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
#     )

#     recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
#     recall_amount = round(true_positive_amount * 100 / (true_positive_amount + false_negative_amount), 2)
#     precision = round(precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100, 2)
#     recall = round(recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100,2)
#     f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
#     false_positive_rate = round(false_positive_count * 100 / (false_positive_count + true_negative_count), 2)
#     false_positive_rate_amount = round(false_positive_amount * 100 / (false_positive_amount + true_negative_amount), 2)

#     result = {
#         'Total Records': total_records,
#         'Total alerts': total_samples,
#         'APPF' : total_frauds,
#         'Alert Rate' :  round(total_samples*100/total_records,2),
#         'Precision (True fraud alert rate)': precision,
#         'Recall/True Positive rate (by member)': recall_rate,
#         'Recall Rate (by value)': recall_amount,
#         'False Positive Rate': false_positive_rate,
#         'F1': f1,
#             'TP' : true_positive_count,
#             'FP' : false_positive_count,
#             'TN' : true_negative_count,
#             'FN' : false_negative_count,
#             'Total' : true_positive_count + false_positive_count + true_negative_count + false_negative_count,
#     }

#     return pd.DataFrame(data=[result])


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

test = (pd.to_datetime('2022-12-01') <= pd.to_datetime(df.company_created_on)) & (pd.to_datetime(df.company_created_on) < pd.to_datetime('2023-02-01')) #& (df.is_approved ==1)
oot = (pd.to_datetime('2023-02-01') <= pd.to_datetime(df.company_created_on)) #& (df.is_approved ==1)

df_test = df[test]
df_oot = df[oot]

df_train = df[~test & ~oot] #& (df.is_approved ==1)]

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

df_train = df_train.fillna(0)

# COMMAND ----------

X_train = df_train[df_train.is_approved==1][features]
y_train = df_train[df_train.is_approved==1]['is_app_fraud']
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
clf = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1, loss_function='Logloss', 
                         cat_features=df_train[features].select_dtypes(include=['object']).columns.tolist(),
                         class_weights=class_weights)
clf.fit(X_train, y_train, verbose=False)
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


col = 'company_icc'
target_column = 'is_app_fraud'
df_train[df_train.is_approved==1].groupby(col).agg(
                total=(target_column, 'count'),
                fraud=(target_column, 'sum'),
                non_fraud=(target_column, lambda x: (x == 0).sum())
            ).reset_index().assign(ratio=lambda x: x.fraud / x.non_fraud).assign(group_rank=lambda x: x['ratio'].rank(ascending=False, method='max')).sort_values(by='group_rank').head(10)

# COMMAND ----------

class CustomEncoder:
    def __init__(self, ratio_threshold=0.021):
        self.ratio_threshold = ratio_threshold
        self.encoding_maps = {}

    def fit_transform(self, df, columns, target_column):
        df_copy = df.copy()
        for col in columns:
            # Calculate the ratio for each category in the specified column
            ratio_df = df_copy.groupby(col).agg(
                total=(target_column, 'count'),
                fraud=(target_column, 'sum'),
                non_fraud=(target_column, lambda x: (x == 0).sum())
            ).reset_index().assign(ratio=lambda x: x.fraud / x.non_fraud).assign(group_rank=lambda x: x['ratio'].rank(ascending=False, method='max'))
            # Identify high-risk categories based on the threshold
            high_risk_categories = ratio_df[(ratio_df.ratio > self.ratio_threshold)][col].unique()
            # Create a mapping dictionary for encoding
            encoding_map = {category: 1 if category in high_risk_categories else 0 for category in df_copy[col].unique()}
            # Apply the custom encoding to the category column
            df_copy[col + '_risk_encoded'] = df_copy[col].map(encoding_map)
            df_copy[col + '_rank_encoded'] = df_copy[col].map(ratio_df.set_index(col)['group_rank'])
            # Save the encoding map for future reference
            self.encoding_maps[col] = encoding_map
        return df_copy


encoder = CustomEncoder(ratio_threshold=0.021)
df_train_transformed_ = encoder.fit_transform(df_train[df_train.is_approved==1], columns=non_numeric_features, target_column='is_app_fraud')
df_train_transformed = pd.concat([df_train_transformed_[[col for col in df_train_transformed_.columns if "_risk_encoded" in col] + numeric_features], df_train_transformed_['is_app_fraud']], axis = 1)
df_train_transformed = df_train_transformed.fillna(0)
df_train_transformed_.head()

# COMMAND ----------

rule_features = ['applicant_idcountry_issue_risk_encoded', 'applicant_id_type_risk_encoded','age_at_completion', 'company_type_risk_encoded']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

rule_features = ['applicant_idcountry_issue_risk_encoded', 'applicant_id_type_risk_encoded','age_at_completion']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=2, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

print(df_train_transformed_[df_train_transformed_['applicant_idcountry_issue_risk_encoded']==1]['applicant_idcountry_issue'].unique())
print(df_train_transformed_[df_train_transformed_['applicant_id_type_risk_encoded']==1]['applicant_id_type'].unique())


# COMMAND ----------


def predict_fraud_rate(df, rule_filter):
    filtered_data = df.copy()
    filtered_data['is_app_fraud'] = np.where(filtered_data.is_approved==0, 1, filtered_data.is_app_fraud)

    total_rm = filtered_data.shape[0]
    total_km = filtered_data[filtered_data.is_approved == 1].shape[0]
    filtered_data['predicted_fraud'] = np.where(rule_filter, 1, 0)
    total_alerts = filtered_data[filtered_data['predicted_fraud'] == 1].shape[0]
    total_frauds = filtered_data[rule_filter]['is_app_fraud'].sum()

    if total_alerts == 0:
        print(f"No samples meet the specified conditions.")
        return None
    
    # fraud_rate = round(filtered_data[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1)].shape[0] * 100 / filtered_data[(filtered_data['predicted_fraud'] == 1)].shape[0], 2)

    # true_negative_count, false_positive_count, false_negative_count, true_positive_count = (
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'is_app_fraud'].count(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'is_app_fraud'].count()
    # )

    # true_negative_amount, false_positive_amount, false_negative_amount, true_positive_amount = (
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 0), 'app_fraud_amount'].sum(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 0) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum(),
    #     filtered_data.loc[(filtered_data['predicted_fraud'] == 1) & (filtered_data['is_app_fraud'] == 1), 'app_fraud_amount'].sum()
    # )

    # recall_rate = round(true_positive_count * 100 / (true_positive_count + false_negative_count), 2)
    # recall_amount = round(true_positive_amount * 100 / (true_positive_amount + false_negative_amount), 2)
    # precision = round(precision_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100, 2)
    # recall = round(recall_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])*100,2)
    # f1 = f1_score(filtered_data['is_app_fraud'], filtered_data['predicted_fraud'])
    # false_positive_rate = round(false_positive_count * 100 / (false_positive_count + true_negative_count), 2)
    # false_positive_rate_amount = round(false_positive_amount * 100 / (false_positive_amount + true_negative_amount), 2)

    actual_frauds = filtered_data[(filtered_data.is_app_fraud ==1) & (filtered_data.is_approved==1)].shape[0]
    alerted_actual_frauds = filtered_data[(filtered_data.is_app_fraud == 1) & (filtered_data.is_approved==1) & (filtered_data.predicted_fraud == 1)].shape[0]
    alerted_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1)].shape[0]
    alerted_frauds_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 1)].shape[0]
    alerted_nonfrauds_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 0)].shape[0]

    result = {
        'RM': total_rm,
        'KM': total_km,
        'Total alerts': total_alerts,
        'APPF' : total_frauds,
        'Alert Rate' :  round(total_alerts*100/total_rm,2),
        'Precision (True Alert Rate)': round(alerted_actual_frauds*100/alerted_km,2),
        'Recall/True Positive rate (by member)': round(alerted_frauds_km*100/actual_frauds,2),
        #'Recall Rate (by value)': recall_amount,
        'False Alert Rate': round(alerted_nonfrauds_km*100/alerted_km,2)
    }

    return pd.DataFrame(data=[result])


# COMMAND ----------

country_list = ['PT', 'SK', 'HU', 'SE', 'BG', 'RO', 'ES', 'PK', 'IN', 'CZ', 'TW', 'DK', 'MT', 'CH', 'GM', 'JM', 'MD', 'JO', 'DZ', 'CM']
id_type = ['National_ID', 'Other_ID']             
rule_filter = (df_train['applicant_idcountry_issue'].isin(country_list)) & (df_train.age_at_completion<=30) & (df_train.applicant_id_type.isin(id_type))
predict_fraud_rate(df_train, rule_filter)

# COMMAND ----------

rule_features = ['age_at_completion', 'company_age_at_completion','business_internal_checks', 'company_icc_risk_encoded']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1) ]['company_icc'].unique())
print(icc_list)

# COMMAND ----------

#icc_list = ['category.wholesale_of_machines_&_equipment', 'category.bailiff', 'category.agricultural_contractors', 'category.boot_&_shoe_shop', 'category.bathroom_&_kitchen_design', 'category.demolition', 'category.watch_&_clock_shop_(including_repair)', 'category.carpet_fitter', 'category.ceiling_contractors', 'category.hygiene_and_cleansing_services', 'category.sunglasses_shop', 'category.knitted_garment_maker', 'category.cooking_&_ironing_services', 'category.appliance_repair_shop', 'category.wine_shop', 'category.appliances_shop', 'category.swimming_pool_installers', 'category.cookery_shop_/_cooking_ingredients', 'category.metalworker', 'category.tour_guide', 'category.timber_preservation', 'category.rendering', 'category.kitchen_storage', 'category.bakery_(factory)', 'category.amusement_arcade', 'category.insurance_agent_/_broker', 'category.washing_machine_repairs_&_servicing_shop', 'category.frozen_food_shop', 'category.exhibition_stand_erector', 'category.sewing_machine_shop_/_sewing_supplies', 'category.personal_training_shop', 'category.barrister', 'category.conservatory_installers', 'category.water_freight_transport', 'category.pedicurist_&_manicurist', 'category.shed_&_carport_erector', 'category.textile_shop', 'category.body_artist_shop', 'category.insurance_motor_vehicle_inspector', 'category.data_removal', 'category.decorators_supplies', 'category.web_cafe', 'category.milkman', 'category.furnisher', 'category.patent_agent', 'category.preacher', 'category.cladding_removal', 'category.fascia_/_guttering_installation', 'category.manufacturing,_processing_&_machining_of_ceramics', 'category.tobacco'] 
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['age_at_completion']<=25)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

icc_list = ['category.wholesale_of_machines_&_equipment', 'category.bailiff', 'category.agricultural_contractors', 'category.boot_&_shoe_shop', 'category.bathroom_&_kitchen_design', 'category.demolition', 'category.watch_&_clock_shop_(including_repair)', 'category.carpet_fitter', 'category.ceiling_contractors', 'category.hygiene_and_cleansing_services', 'category.sunglasses_shop', 'category.knitted_garment_maker', 'category.cooking_&_ironing_services', 'category.appliance_repair_shop', 'category.wine_shop', 'category.appliances_shop', 'category.swimming_pool_installers', 'category.cookery_shop_/_cooking_ingredients', 'category.metalworker', 'category.tour_guide', 'category.timber_preservation', 'category.rendering', 'category.kitchen_storage', 'category.bakery_(factory)', 'category.amusement_arcade', 'category.insurance_agent_/_broker', 'category.washing_machine_repairs_&_servicing_shop', 'category.frozen_food_shop', 'category.exhibition_stand_erector', 'category.sewing_machine_shop_/_sewing_supplies', 'category.personal_training_shop', 'category.barrister', 'category.conservatory_installers', 'category.water_freight_transport', 'category.pedicurist_&_manicurist', 'category.shed_&_carport_erector', 'category.textile_shop', 'category.body_artist_shop', 'category.insurance_motor_vehicle_inspector', 'category.data_removal', 'category.decorators_supplies', 'category.web_cafe', 'category.milkman', 'category.furnisher', 'category.patent_agent', 'category.preacher', 'category.cladding_removal', 'category.fascia_/_guttering_installation', 'category.manufacturing,_processing_&_machining_of_ceramics', 'category.tobacco'] 
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['age_at_completion']<=25)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_features = ['company_structurelevelwise_risk_encoded', 'individual_identity_address','business_internal_checks', 'company_icc_risk_encoded']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['individual_identity_address']==1) & (df_train_transformed_['business_internal_checks']==1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_features = ['company_structurelevelwise_risk_encoded', 'individual_identity_address','count_failed_business_rules', 'company_icc_risk_encoded']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['count_failed_business_rules']>=2) & (df_train_transformed_['individual_identity_address']==1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['individual_identity_address']==1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_features = ['company_status_risk_encoded', 'is_restricted_keyword_present', 'company_icc_risk_encoded', 'directors_avg_age_at_completion']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight=class_weights)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule_filter = (df_train['company_icc'].isin(icc_list)) & (df_train['directors_avg_age_at_completion']<=26) & (df_train['is_restricted_keyword_present']==1)
predict_fraud_rate(df_train, rule_filter)

# COMMAND ----------

icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule_filter = (df_train_transformed_['company_icc'].isin(icc_list)) & (df_train_transformed_['directors_avg_age_at_completion']<=26) & (df_train_transformed_['is_restricted_keyword_present']==1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_features = ['company_status_risk_encoded', 'individual_sanctions_pep', 'applicant_years_to_id_expiry', 'directors_avg_age_at_completion']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, 
                             max_depth=None, 
                             class_weight=class_weights, 
                             max_features=3,
                             max_leaf_nodes = 8)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

rule_filter = (df_train_transformed_['applicant_years_to_id_expiry']<=5) & (df_train_transformed_['directors_avg_age_at_completion']<=25) & (df_train_transformed_['individual_sanctions_pep']==1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_features = ['applicant_email_numeric', 'applicant_email_domain_risk_encoded', 'applicant_name_mismatch_ind', 'applicant_years_to_id_expiry', 'applicant_nationality_0_risk_encoded']
class_counts = df_train_transformed['is_app_fraud'].value_counts()
total_instances = len(df_train_transformed['is_app_fraud'])
class_weights = {class_label: total_instances / (class_counts[class_label] * len(class_counts)) 
                 for class_label in class_counts.index}
print(class_weights)
#class_weights = {0:1, 1:100}
clf = DecisionTreeClassifier(random_state=42, 
                             max_depth=None, 
                             class_weight=class_weights, 
                             max_features=3,
                             max_leaf_nodes = 8)
clf.fit(df_train_transformed[rule_features], df_train_transformed['is_app_fraud'])
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
tree.plot_tree(clf, feature_names=rule_features,
          class_names=[str(i) for i in clf.classes_], filled=True, fontsize=10)  # Adjust fontsize as needed
plt.show()

# COMMAND ----------

rule_filter = (df_train_transformed_['applicant_email_numeric']==1) & (df_train_transformed_['applicant_nationality_0_risk_encoded']==1) & (df_train_transformed_['applicant_years_to_id_expiry']>=5) 
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_filter = (df_train_transformed_['applicant_email_numeric']==1) & (df_train_transformed_['applicant_nationality_0_risk_encoded']==1) & (df_train_transformed_['applicant_years_to_id_expiry']<=5) 
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_filter = (df_train_transformed_['applicant_email_numeric'] == 1) & (df_train_transformed_['applicant_email_domain'] == 'outlook.com')
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_filter = (df_train_transformed_['age_at_completion'] <= 20) & (df_train_transformed_['company_age_at_completion'] <= 1)
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

rule_filter = (df_train_transformed_['company_type']=="sole-trader") & (df_train_transformed_['applicant_idcountry_issue_risk']=="High")
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

nob_list = ['category.construction'
'category.domestic_cleaner',
'category.cleaner',
'category.household_cleaning_services',
'category.transport_and_storage',
'category.industrial_cleaning_services']
rule_filter = (df_train_transformed_['company_icc'].isin(nob_list)) & (df_train_transformed_['applicant_idcountry_issue'] == 'RO')
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

id_list = ['Other_ID', 'Provisional_Licence', 'Undefined']
country_list = ['GB', 'RO', 'IN', 'PT', 'SK']
icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']
rule_filter=(df_train_transformed_.applicant_id_type.isin(id_list)) & (df_train_transformed_.applicant_idcountry_issue.isin(country_list)) &  (df_train_transformed_.company_icc.isin(icc_list))
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

postcode_list = ['B', 'E', 'M', 'CV', 'IG', 'N', 'LS', 'SE', 'IP', 'NE']
rule_filter=(df_train_transformed_.applicant_postcode.isin(postcode_list)) & (df_train_transformed_.individual_identity_address.isin([1])) &  (df_train_transformed_.company_icc.isin(icc_list))
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

icc_list =['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services', 'category.online_retailer_/_online_shop_/_ecommerce', 'category.industrial_cleaning_services', 'category.land_freight_transport', 'category.domestic_cleaner', 'category.retail_of_textiles,_clothes_&_footwear']
rule_filter=(df_train_transformed_.company_icc.isin(icc_list)) & (df_train_transformed_.age_at_completion.lt(25)) & (df_train_transformed_.applicant_idcountry_issue.isin(country_list))
predict_fraud_rate(df_train_transformed_, rule_filter)

# COMMAND ----------

