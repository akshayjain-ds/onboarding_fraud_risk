# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

#%pip install catboost

# COMMAND ----------

import sys
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

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

# import catboost
# from catboost import CatBoostClassifier
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

pd.to_datetime(df.company_created_on).describe()

# COMMAND ----------

test = (pd.to_datetime('2023-01-01') <= pd.to_datetime(df.company_created_on)) & (pd.to_datetime(df.company_created_on) < pd.to_datetime('2023-07-01')) #& (df.is_approved ==1)
#oot = (pd.to_datetime('2023-02-01') <= pd.to_datetime(df.company_created_on)) #& (df.is_approved ==1)

df_test = df[test]
#df_oot = df[oot]

df_train = df[~test]# & ~oot] #& (df.is_approved ==1)]

print(df_train.shape, df_test.shape)


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


def predict_fraud_rate(df, rule_filter, rule_name):
    filtered_data = df.copy()
    filtered_data['is_app_fraud'] = np.where(filtered_data['is_approved']==0, 1, filtered_data['is_app_fraud'])

    total_rm = filtered_data.shape[0]
    total_km = filtered_data[filtered_data.is_approved == 1].shape[0]
    filtered_data['predicted_fraud'] = np.where(rule_filter, 1, 0)
    total_alerts = filtered_data[filtered_data['predicted_fraud'] == 1].shape[0]

    if total_alerts == 0:
        print(f"No samples meet the specified conditions.")
        return None

    actual_frauds = filtered_data[(filtered_data.is_app_fraud ==1) & (filtered_data.is_approved==1)].shape[0]
    actual_fraudvalue =  filtered_data[(filtered_data.is_app_fraud ==1) & (filtered_data.is_approved==1)].app_fraud_amount.sum()
    alerted_actual_frauds = filtered_data[(filtered_data.is_app_fraud == 1) & (filtered_data.is_approved==1) & (filtered_data.predicted_fraud == 1)].shape[0]
    alerted_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1)].shape[0]
    alerted_rm = filtered_data[(filtered_data.predicted_fraud==1)].shape[0]

    alerted_frauds_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 1)].shape[0]
    alerted_frauds_rm = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_app_fraud == 1)].shape[0]
    alerted_fraudvalue_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 1)].app_fraud_amount.sum()

    alerted_nonfrauds_km = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 0)].shape[0]
    alerted_nonfrauds_rm = filtered_data[(filtered_data.predicted_fraud==1) & (filtered_data.is_app_fraud == 0)].shape[0]


    result = {
        'Rule' : rule_name,
        'RM': total_rm,
        'KM': total_km,
        'Alerted KM': alerted_km,
        'Alerted RM': alerted_rm,
        'Total alerts': total_alerts,
        'Alerted frauds(KM)' : alerted_frauds_km,
        'Alerted frauds(RM)' : alerted_frauds_rm,
        'Alert Rate' :  round(total_alerts*100/total_rm,2),
        'Precision (True Alert Rate)': round(alerted_actual_frauds*100/alerted_km,2),
        'Recall/True Positive rate (by member)': round(alerted_frauds_km*100/actual_frauds,2),
        'Recall/True Positive rate (by value)': round(alerted_fraudvalue_km*100/actual_fraudvalue,2),
        'False Alert Rate(KM)': round(alerted_nonfrauds_km*100/alerted_km,2),
        'False Alert Rate(RM)': round(alerted_nonfrauds_rm*100/alerted_rm,2),

        'Fraud per KM (overall)' : filtered_data[(filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 1)].shape[0]/filtered_data[(filtered_data.is_approved==1)].shape[0],
        'Fraud value per KM (overall)' : filtered_data[(filtered_data.is_approved==1) & (filtered_data.is_app_fraud == 1)].app_fraud_amount.sum()/filtered_data[(filtered_data.is_approved==1)].shape[0],

        'Fraud per KM (alerted)' : filtered_data[(filtered_data.is_approved==1) & (filtered_data.predicted_fraud == 1) & (filtered_data.is_app_fraud == 1)].shape[0]/filtered_data[(filtered_data.is_approved==1)].shape[0],

        'Fraud value per KM (alerted)' : filtered_data[(filtered_data.is_approved==1) & (filtered_data.predicted_fraud == 1) & (filtered_data.is_app_fraud == 1)].app_fraud_amount.sum()/filtered_data[(filtered_data.is_approved==1)].shape[0]
    }

    return pd.DataFrame(data=[result])


# COMMAND ----------

rule1_country_list = ['PT', 'SK', 'HU', 'SE', 'BG', 'RO', 'ES', 'PK', 'IN', 'CZ', 'TW', 'DK', 'MT', 'CH', 'GM', 'JM', 'MD', 'JO', 'DZ', 'CM']
rule1_id_type = ['National_ID', 'Other_ID']             
rule1_filter = (df_train['applicant_idcountry_issue'].isin(rule1_country_list)) & (df_train.age_at_completion<=30) & (df_train.applicant_id_type.isin(rule1_id_type))
rule1_name = '''applicant_id_country_issue in ['PT', 'SK', 'HU', 'SE', 'BG', 'RO', 'ES', 'PK', 'IN', 'CZ', 'TW', 'DK', 'MT', 'CH', 'GM', 'JM', 'MD', 'JO', 'DZ', 'CM'] & age at completion <= 30 & applicant_id_type in ['National_ID', 'Other_ID'. '''
predict_fraud_rate(df_train, rule1_filter, rule1_name)

# COMMAND ----------

rule2_icc_list = ['category.wholesale_of_machines_&_equipment', 'category.bailiff', 'category.agricultural_contractors', 'category.boot_&_shoe_shop', 'category.bathroom_&_kitchen_design', 'category.demolition', 'category.watch_&_clock_shop_(including_repair)', 'category.carpet_fitter', 'category.ceiling_contractors', 'category.hygiene_and_cleansing_services', 'category.sunglasses_shop', 'category.knitted_garment_maker', 'category.cooking_&_ironing_services', 'category.appliance_repair_shop', 'category.wine_shop', 'category.appliances_shop', 'category.swimming_pool_installers', 'category.cookery_shop_/_cooking_ingredients', 'category.metalworker', 'category.tour_guide', 'category.timber_preservation', 'category.rendering', 'category.kitchen_storage', 'category.bakery_(factory)', 'category.amusement_arcade', 'category.insurance_agent_/_broker', 'category.washing_machine_repairs_&_servicing_shop', 'category.frozen_food_shop', 'category.exhibition_stand_erector', 'category.sewing_machine_shop_/_sewing_supplies', 'category.personal_training_shop', 'category.barrister', 'category.conservatory_installers', 'category.water_freight_transport', 'category.pedicurist_&_manicurist', 'category.shed_&_carport_erector', 'category.textile_shop', 'category.body_artist_shop', 'category.insurance_motor_vehicle_inspector', 'category.data_removal', 'category.decorators_supplies', 'category.web_cafe', 'category.milkman', 'category.furnisher', 'category.patent_agent', 'category.preacher', 'category.cladding_removal', 'category.fascia_/_guttering_installation', 'category.manufacturing,_processing_&_machining_of_ceramics', 'category.tobacco'] 
rule2_filter = (df_train['company_icc'].isin(rule2_icc_list)) & (df_train['age_at_completion']<=25)
rule2_name = ''' High risk NOB & age_at_completion <= 25 '''
predict_fraud_rate(df_train, rule2_filter, rule2_name)

# COMMAND ----------

rule3_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule3_filter = (df_train['company_icc'].isin(rule3_icc_list)) & (df_train['individual_identity_address']==1) & (df_train['business_internal_checks']==1)
rule3_name = '''High risk NOB & idenity/address mismatch triggers & business_internal_checks==1'''
predict_fraud_rate(df_train, rule3_filter, rule3_name)

# COMMAND ----------

rule4_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule4_filter = (df_train['company_icc'].isin(rule4_icc_list)) & (df_train['count_failed_business_rules']>=2) & (df_train['individual_identity_address']==1)
rule4_name='''High risk NOB & #Failed business rules >= 2 & idenity/address mismatch triggers'''
predict_fraud_rate(df_train, rule4_filter, rule4_name)

# COMMAND ----------

rule5_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
print(rule5_icc_list)
rule5_filter = (df_train['company_icc'].isin(rule5_icc_list)) & (df_train['individual_identity_address']==1)
rule5_name = '''High risk NOB & idenity/address mismatch triggers'''
predict_fraud_rate(df_train, rule5_filter, rule5_name)

# COMMAND ----------

rule6_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule6_filter = (df_train['company_icc'].isin(rule6_icc_list)) & (df_train['directors_avg_age_at_completion']<=26) & (df_train['is_restricted_keyword_present']==1)
rule6_name = '''High risk NOB & Average age of directors <=26 & Restricted keyword is present'''
predict_fraud_rate(df_train, rule6_filter, rule6_name)

# COMMAND ----------

rule7_filter = (df_train['applicant_years_to_id_expiry']<=5) & (df_train['directors_avg_age_at_completion']<=25) & (df_train['individual_sanctions_pep']==1)
rule7_name = '''Years to expiry of ID <=5 & Average age of directors <=25 & PEP and Sactions checks failed'''
predict_fraud_rate(df_train, rule7_filter, rule7_name)

# COMMAND ----------

rule8_nation_list = list(df_train_transformed_[(df_train_transformed_['applicant_nationality_0_risk_encoded']==1)]['applicant_nationality_0'].unique())
print(rule8_nation_list)
rule8_filter = (df_train['applicant_email_numeric']==1) & (df_train['applicant_nationality_0'].isin(rule8_nation_list)) & (df_train['applicant_years_to_id_expiry']>=5) 
rule8_name = '''Email contains numeric & High risk nationlity & Years to expiry of ID >=5'''
predict_fraud_rate(df_train, rule8_filter, rule8_name)

# COMMAND ----------

rule9_filter = (df_train['applicant_email_numeric'] == 1) & (df_train['applicant_email_domain'] == 'outlook.com')
rule9_name = '''Risky email format'''
predict_fraud_rate(df_train, rule9_filter, rule9_name)

# COMMAND ----------

rule10_filter = (df_train['age_at_completion'] <= 20) & (df_train['company_age_at_completion'] <= 1)
rule10_name = '''Age at completion <= 20Y & Company age at completion <= 1M'''
predict_fraud_rate(df_train, rule10_filter, rule10_name)

# COMMAND ----------

rule11_filter = (df_train['company_type']=="sole-trader") & (df_train['applicant_idcountry_issue_risk']=="High")
print(list(df_train[(df_train['applicant_idcountry_issue_risk']=="High")]['applicant_idcountry_issue'].unique()))
rule11_name = '''Sole trader & Hgh risk country of ID issue'''
predict_fraud_rate(df_train, rule11_filter, rule11_name)

# COMMAND ----------

rule12_nob_list = ['category.construction'
'category.domestic_cleaner',
'category.cleaner',
'category.household_cleaning_services',
'category.transport_and_storage',
'category.industrial_cleaning_services']
rule12_filter = (df_train['company_icc'].isin(rule12_nob_list)) & (df_train['applicant_idcountry_issue'] == 'RO')
rule12_name = '''High risk NOB & Romania'''
predict_fraud_rate(df_train, rule12_filter, rule12_name)

# COMMAND ----------

rule13_id_list = ['Other_ID', 'Provisional_Licence', 'Undefined']
rule13_country_list = ['GB', 'RO', 'IN', 'PT', 'SK']
rule13_icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']
rule13_filter=(df_train.applicant_id_type.isin(rule13_id_list)) & (df_train.applicant_idcountry_issue.isin(rule13_country_list)) &  (df_train.company_icc.isin(rule13_icc_list))
rule13_name='''Risky ID type & Risky countries & Risky NOB'''
predict_fraud_rate(df_train, rule13_filter, rule13_name)

# COMMAND ----------

rule14_postcode_list = ['B', 'E', 'M', 'CV', 'IG', 'N', 'LS', 'SE', 'IP', 'NE']
rule14_icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']
rule14_filter=(df_train.applicant_postcode.isin(rule14_postcode_list)) & (df_train.individual_identity_address.isin([1])) &  (df_train.company_icc.isin(rule14_icc_list))
rule14_name = '''High risk postcode & High risk NOB & idenity/address mismatch triggers'''
predict_fraud_rate(df_train, rule14_filter, rule14_name)

# COMMAND ----------


rule15_filter =  df_train['company_status'].isin(['Liquidation', 'Dissolved', 'Undefined']) & (df_train['company_icc'].isin(list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique()))) & (df_train['applicant_idcountry_issue'].isin(list(df_train_transformed_[(df_train_transformed_['applicant_idcountry_issue_risk_encoded']==1)]['applicant_idcountry_issue'].unique())))

print(list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique()))
print(list(df_train_transformed_[(df_train_transformed_['applicant_idcountry_issue_risk_encoded']==1)]['applicant_idcountry_issue'].unique()))

rule15_name = '''High risk company status & High risk NOB & High risk country of ID issue'''
predict_fraud_rate(df_train, rule15_filter, rule15_name)

# COMMAND ----------


num_rules = 13
rule_filters = [rule1_filter,
 rule2_filter,
 rule3_filter,
 rule4_filter,
 rule5_filter,
 rule6_filter,
 rule7_filter,
 rule8_filter,
 rule9_filter,
 rule10_filter,
 rule11_filter,
 rule12_filter,
 rule13_filter,
 rule14_filter,
 rule15_filter]

rule_names = [rule1_name,
 rule2_name,
 rule3_name,
 rule4_name,
 rule5_name,
 rule6_name,
 rule7_name,
 rule8_name,
 rule9_name,
 rule10_name,
 rule11_name,
 rule12_name,
 rule13_name,
 rule14_name,
 rule15_name]

result_df = pd.DataFrame()

for rf, rn in zip(rule_filters, rule_names):
  result_df = pd.concat([result_df, predict_fraud_rate(df_train, rf, rn)])

result_df.reset_index(drop=True).set_index(pd.RangeIndex(start=1, stop=len(result_df) + 1))



# COMMAND ----------

rule1_country_list = ['PT', 'SK', 'HU', 'SE', 'BG', 'RO', 'ES', 'PK', 'IN', 'CZ', 'TW', 'DK', 'MT', 'CH', 'GM', 'JM', 'MD', 'JO', 'DZ', 'CM']
rule1_id_type = ['National_ID', 'Other_ID']             
rule1_filter = (df_test['applicant_idcountry_issue'].isin(rule1_country_list)) & (df_test.age_at_completion<=30) & (df_test.applicant_id_type.isin(rule1_id_type))
rule1_name = '''applicant_id_country_issue in ['PT', 'SK', 'HU', 'SE', 'BG', 'RO', 'ES', 'PK', 'IN', 'CZ', 'TW', 'DK', 'MT', 'CH', 'GM', 'JM', 'MD', 'JO', 'DZ', 'CM'] & age at completion <= 30 & applicant_id_type in ['National_ID', 'Other_ID'. '''

rule2_icc_list = ['category.wholesale_of_machines_&_equipment', 'category.bailiff', 'category.agricultural_contractors', 'category.boot_&_shoe_shop', 'category.bathroom_&_kitchen_design', 'category.demolition', 'category.watch_&_clock_shop_(including_repair)', 'category.carpet_fitter', 'category.ceiling_contractors', 'category.hygiene_and_cleansing_services', 'category.sunglasses_shop', 'category.knitted_garment_maker', 'category.cooking_&_ironing_services', 'category.appliance_repair_shop', 'category.wine_shop', 'category.appliances_shop', 'category.swimming_pool_installers', 'category.cookery_shop_/_cooking_ingredients', 'category.metalworker', 'category.tour_guide', 'category.timber_preservation', 'category.rendering', 'category.kitchen_storage', 'category.bakery_(factory)', 'category.amusement_arcade', 'category.insurance_agent_/_broker', 'category.washing_machine_repairs_&_servicing_shop', 'category.frozen_food_shop', 'category.exhibition_stand_erector', 'category.sewing_machine_shop_/_sewing_supplies', 'category.personal_training_shop', 'category.barrister', 'category.conservatory_installers', 'category.water_freight_transport', 'category.pedicurist_&_manicurist', 'category.shed_&_carport_erector', 'category.textile_shop', 'category.body_artist_shop', 'category.insurance_motor_vehicle_inspector', 'category.data_removal', 'category.decorators_supplies', 'category.web_cafe', 'category.milkman', 'category.furnisher', 'category.patent_agent', 'category.preacher', 'category.cladding_removal', 'category.fascia_/_guttering_installation', 'category.manufacturing,_processing_&_machining_of_ceramics', 'category.tobacco'] 
rule2_filter = (df_test['company_icc'].isin(rule2_icc_list)) & (df_test['age_at_completion']<=25)
rule2_name = ''' High risk NOB & age_at_completion <= 25 '''

rule3_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule3_filter = (df_test['company_icc'].isin(rule3_icc_list)) & (df_test['individual_identity_address']==1) & (df_test['business_internal_checks']==1)
rule3_name = '''High risk NOB & idenity/address mismatch triggers & business_internal_checks==1'''

rule4_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule4_filter = (df_test['company_icc'].isin(rule4_icc_list)) & (df_test['count_failed_business_rules']>=2) & (df_test['individual_identity_address']==1)
rule4_name='''High risk NOB & #Failed business rules >= 2 & idenity/address mismatch triggers'''

rule5_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule5_filter = (df_test['company_icc'].isin(rule5_icc_list)) & (df_test['individual_identity_address']==1)
rule5_name = '''High risk NOB & idenity/address mismatch triggers'''

rule6_icc_list = list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique())
rule6_filter = (df_test['company_icc'].isin(rule6_icc_list)) & (df_test['directors_avg_age_at_completion']<=26) & (df_test['is_restricted_keyword_present']==1)
rule6_name = '''High risk NOB & Average age of directors <=26 & Restricted keyword is present'''

rule7_filter = (df_test['applicant_years_to_id_expiry']<=5) & (df_test['directors_avg_age_at_completion']<=25) & (df_test['individual_sanctions_pep']==1)
rule7_name = '''Years to expiry of ID <=5 & Average age of directors <=25 & PEP and Sactions checks failed'''


rule8_nation_list = list(df_train_transformed_[(df_train_transformed_['applicant_nationality_0_risk_encoded']==1)]['applicant_nationality_0'].unique())
rule8_filter = (df_test['applicant_email_numeric']==1) & (df_test['applicant_nationality_0'].isin(rule8_nation_list)) & (df_test['applicant_years_to_id_expiry']>=5) 
rule8_name = '''Email contains numeric & High risk nationlity & Years to expiry of ID >=5'''

rule9_filter = (df_test['applicant_email_numeric'] == 1) & (df_test['applicant_email_domain'] == 'outlook.com')
rule9_name = '''Risky email format'''

rule10_filter = (df_test['age_at_completion'] <= 20) & (df_test['company_age_at_completion'] <= 1)
rule10_name = '''Age at completion <= 20Y & Company age at completion <= 1M'''

rule11_filter = (df_test['company_type']=="sole-trader") & (df_test['applicant_idcountry_issue_risk']=="High")
rule11_name = '''Sole trader & Hgh risk country of ID issue'''

rule12_nob_list = ['category.construction'
'category.domestic_cleaner',
'category.cleaner',
'category.household_cleaning_services',
'category.transport_and_storage',
'category.industrial_cleaning_services']
rule12_filter = (df_test['company_icc'].isin(rule12_nob_list)) & (df_test['applicant_idcountry_issue'] == 'RO')
rule12_name = '''High risk NOB & Romania'''

rule13_id_list = ['Other_ID', 'Provisional_Licence', 'Undefined']
rule13_country_list = ['GB', 'RO', 'IN', 'PT', 'SK']
rule13_icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']
rule13_filter=(df_test.applicant_id_type.isin(rule13_id_list)) & (df_test.applicant_idcountry_issue.isin(rule13_country_list)) &  (df_test.company_icc.isin(rule13_icc_list))
rule13_name='''Risky ID type & Risky countries & Risky NOB'''

rule14_postcode_list = ['B', 'E', 'M', 'CV', 'IG', 'N', 'LS', 'SE', 'IP', 'NE']
rule14_icc_list = ['category.construction', 'category.builder', 'category.cleaner', 'category.transport_and_storage', 'category.household_cleaning_services']
rule14_filter=(df_test.applicant_postcode.isin(rule14_postcode_list)) & (df_test.individual_identity_address.isin([1])) &  (df_test.company_icc.isin(rule14_icc_list))
rule14_name = '''High risk postcode & High risk NOB & idenity/address mismatch triggers'''


rule15_filter =  df_test['company_status'].isin(['Liquidation', 'Dissolved', 'Undefined']) & (df_test['company_icc'].isin(list(df_train_transformed_[(df_train_transformed_['company_icc_risk_encoded']==1)]['company_icc'].unique()))) & (df_test['applicant_idcountry_issue'].isin(list(df_train_transformed_[(df_train_transformed_['applicant_idcountry_issue_risk_encoded']==1)]['applicant_idcountry_issue'].unique())))
rule15_name = '''High risk company status & High risk NOB & High risk country of ID issue'''


rule_filters = [rule1_filter,
 rule2_filter,
 rule3_filter,
 rule4_filter,
 rule5_filter,
 rule6_filter,
 rule7_filter,
 rule8_filter,
 rule9_filter,
 rule10_filter,
 rule11_filter,
 rule12_filter,
 rule13_filter,
 rule14_filter,
 rule15_filter]

rule_names = [rule1_name,
 rule2_name,
 rule3_name,
 rule4_name,
 rule5_name,
 rule6_name,
 rule7_name,
 rule8_name,
 rule9_name,
 rule10_name,
 rule11_name,
 rule12_name,
 rule13_name,
 rule14_name,
 rule15_name]

result_df = pd.DataFrame()

for rf, rn in zip(rule_filters, rule_names):
  result_df = pd.concat([result_df, predict_fraud_rate(df_test, rf, rn)])

result_df.reset_index(drop=True).set_index(pd.RangeIndex(start=1, stop=len(result_df) + 1))


# COMMAND ----------

