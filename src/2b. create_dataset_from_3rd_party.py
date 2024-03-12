# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Feature 2
# MAGIC This notebook is used used do pull level 2 of features (from DueDIl/CH/3rd party company data)

# COMMAND ----------

# MAGIC %run ../set_up
# MAGIC

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

appf_df = spark_connector(perpetrator_query.format(from_date=start_date, to_date=end_date))
appf_df = appf_df.toDF(*[c.lower().split('.')[-1] for c in appf_df.columns])
appf_df.count()

# COMMAND ----------

appf_df.show()

# COMMAND ----------

from pyspark.sql.functions import min, max
appf_df.select(min("timestamp")).show(), appf_df.select(max("timestamp")).show()

# COMMAND ----------

from pyspark.sql.functions import min, max
appf_df.select(min("approved_at")).show(), appf_df.select(max("approved_at")).show()

# COMMAND ----------

appf_df = appf_df.drop('approved_at')

# COMMAND ----------

dfapp = appf_df.toPandas()
dfapp.shape

# COMMAND ----------

dfapp.head()

# COMMAND ----------

dfapp.index.duplicated().sum()

# COMMAND ----------

dfapp.set_index('company_id', inplace=True)


# COMMAND ----------

dfapp['is_approved'] = pd.to_numeric(dfapp['is_approved'], errors='ignore')
dfapp['is_app_fraud'] = pd.to_numeric(dfapp['is_app_fraud'], errors='ignore')
dfapp['app_fraud_amount'] = pd.to_numeric(dfapp['app_fraud_amount'], errors='ignore')

# COMMAND ----------

name_match_df = spark_connector(name_match_query.format(from_date=start_date, to_date=end_date))
name_match_df = name_match_df.toDF(*[c.lower().split('.')[-1] for c in name_match_df.columns])
name_match_df.count()

# COMMAND ----------

name_match_df = name_match_df.toPandas()
name_match_df.shape

# COMMAND ----------

name_match_df.drop_duplicates(subset=['company_id'], inplace=True)
name_match_df.set_index('company_id', inplace=True)
name_match_df.shape

# COMMAND ----------

name_match_df['applicant_name_match_score'] = name_match_df[['full_name', 'id_full_name']].apply(lambda row: fuzz.partial_token_set_ratio(
  row['full_name'], row['id_full_name']), axis=1)
name_match_df['applicant_name_mismatch_ind'] = name_match_df['applicant_name_match_score'].apply(lambda x: 1 if x < 100 else 0)

# COMMAND ----------

name_match_df[['applicant_name_mismatch_ind']].describe()

# COMMAND ----------

# del dfapp['name_mismatch_ind']
dfapp = dfapp.merge(name_match_df[['applicant_name_mismatch_ind']], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

dfapp[dfapp.is_approved==1].groupby(['applicant_name_mismatch_ind'])['is_app_fraud'].mean()

# COMMAND ----------

get_iv_class(dfapp[dfapp.is_approved==1]['is_app_fraud'], dfapp[dfapp.is_approved==1]['applicant_name_mismatch_ind'], 'name_mismatch_ind')

# COMMAND ----------

member_df = spark_connector(member_query.format(from_date=start_date, to_date=end_date))
member_df = member_df.toDF(*[c.lower().split('.')[-1] for c in member_df.columns])
member_df.count()

# COMMAND ----------

member_df = member_df.toPandas()
member_df['company_id'].duplicated().sum()

# COMMAND ----------

member_df.head()

# COMMAND ----------

if member_df['company_id'].duplicated().sum() == 0:
  member_df.set_index('company_id', inplace=True)
  print("company_id set as index")
member_df.shape

# COMMAND ----------

pd.isnull(member_df).sum()/member_df.shape[0]

# COMMAND ----------

dfapp = dfapp.merge(member_df[['comnpany_accounts_overdue', 'comnpany_accounts_next_due_at']], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

dfapp[dfapp['comnpany_accounts_overdue']==True][['timestamp', 'comnpany_accounts_overdue', 'comnpany_accounts_next_due_at']]

# COMMAND ----------

dfapp.groupby(['comnpany_accounts_overdue'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dfapp = dfapp.merge(member_df[['applicant_years_to_id_expiry']], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

dfapp.head()

# COMMAND ----------

pd.isnull(dfapp).sum()

# COMMAND ----------

dfapp['is_approved'] = pd.to_numeric(dfapp['is_approved'], errors='ignore')
dfapp['is_app_fraud'] = pd.to_numeric(dfapp['is_app_fraud'], errors='ignore')
dfapp['app_fraud_amount'] = pd.to_numeric(dfapp['app_fraud_amount'], errors='ignore')
dfapp['applicant_years_to_id_expiry'] = pd.to_numeric(dfapp['applicant_years_to_id_expiry'], errors='ignore')

# COMMAND ----------

dfapp['applicant_years_to_id_expiry'].describe(percentiles=np.linspace(0,1,101))

# COMMAND ----------

dfapp['applicant_years_to_id_expiry'] = np.clip(dfapp['applicant_years_to_id_expiry'], a_min=0, a_max=10)

# COMMAND ----------

dfapp['applicant_years_to_id_expiry'].median()

# COMMAND ----------

# dfapp['applicant_years_to_id_expiry'] = dfapp['applicant_years_to_id_expiry'].fillna(dfapp['applicant_years_to_id_expiry'].median())

# COMMAND ----------

dfapp[dfapp['is_approved']==1][['is_app_fraud', 'applicant_years_to_id_expiry']].corr('spearman')


# COMMAND ----------


dd_df = spark_connector(duedil_query.format(from_date="2021-10-01", to_date="2023-12-31"))
dd_df = dd_df.toDF(*[c.lower().split('.')[-1] for c in dd_df.columns])
dd_df.count()

# COMMAND ----------

dd_df = dd_df.toPandas()
dd_df.head()

# COMMAND ----------

dd_df['company_id'].duplicated().sum()

# COMMAND ----------

dd_df.set_index('company_id', inplace=True)

# COMMAND ----------

dd_df['duedil_hit'].mean()

# COMMAND ----------

dd_df['company_status'] = dd_df['status'].apply(lambda x: "Undefined" if pd.isnull(x) else str(x)).astype(str)
dd_df['company_status'].value_counts()

# COMMAND ----------

company_postcode = dd_df['company_postcode'].apply(lambda x: postcode_value_mapper(x, validate_and_get_postcode(x)))
company_postcode_risk = company_postcode.apply(lambda x: json.loads(x).get("output").get("value"))
dd_df['company_postcode'] = dd_df['company_postcode'].apply(lambda x: validate_and_get_postcode(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x).apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x)))
dd_df['company_postcode_risk'] = company_postcode_risk.apply(lambda x: 'High' if x == 'high' else x)

# COMMAND ----------

dd_df['company_postcode'].value_counts()

# COMMAND ----------

dd_df['company_postcode_risk'].value_counts()/dd_df.shape[0]

# COMMAND ----------

dfapp = dfapp.merge(dd_df[['company_status', 'company_postcode', 'company_postcode_risk']], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['company_status'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['company_postcode_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

def get_director_info(payload: str) -> {}:

  try:

    directorstree = json.loads(payload)
    directors = {}
    for i, officer in enumerate(directorstree):

      officer_id = directorstree[i].get("officerId", '')
      directors[officer_id] = {}

      directors[officer_id]['dd_first_name'] = directorstree[i].get("person", {}).get("firstName", '').lower()
      directors[officer_id]['dd_last_name'] = directorstree[i].get("person", {}).get("lastName", '').lower()
      dd_dob = pd.to_datetime(directorstree[i].get("person", {}).get("dateOfBirth", ''))
      directors[officer_id]['dd_dob_month'] = str(dd_dob.date().month)
      directors[officer_id]['dd_dob_year'] = str(dd_dob.date().year)
      directors[officer_id]['dd_nationalities'] = directorstree[i].get("person", {}).get("nationalities", [])
    
    return directors

  except:
    return None

# COMMAND ----------

dd_df['directors_info'] = dd_df['directorstree'].apply(lambda x: get_director_info(x))

# COMMAND ----------

pd.isnull(dd_df['directors_info']).sum()/dd_df['directors_info'].shape[0]

# COMMAND ----------

applicant_director_info = member_df[['id_first_name', 'id_last_name', 'date_of_birth']].merge(dd_df[['directors_info']], left_index=True, right_index=True).merge(dfapp[['timestamp']], left_index=True, right_index=True)
applicant_director_info.shape

# COMMAND ----------

applicant_director_info.head()

# COMMAND ----------

def get_applicant_director_nationality(row: pd.Series, similarity_threshold=90):

  if bool(row['directors_info']):
  
    applicant_id_firstname = str(row['id_first_name']).lower()
    applicant_id_lasttname = str(row['id_last_name']).lower()
    applicant_dob = pd.to_datetime(row['date_of_birth']).date()
    applicant_dob_month = str(applicant_dob.month)
    applicant_dob_year = str(applicant_dob.year)
    directors_info = row['directors_info']

    for officer in directors_info.keys():
      director = directors_info.get(officer, {})

      if (fuzz.partial_token_set_ratio(applicant_id_firstname, director.get('dd_first_name', '')) >= similarity_threshold) and \
        (fuzz.partial_token_set_ratio(applicant_id_lasttname, director.get('dd_last_name', '')) >= similarity_threshold) and \
          applicant_dob_month == director.get('dd_dob_month', '') \
            and applicant_dob_year == director.get('dd_dob_year', ''):
              return director.get('dd_nationalities', [None])
      
  return [None]

# COMMAND ----------

def get_applicant_directors_avg_age(row: pd.Series):
  
  company_created_on = pd.to_datetime(row['timestamp'])

  if bool(row['directors_info']):
  
    directors_info = row['directors_info']

    directors_age = []
    for officer in directors_info.keys():
      director = directors_info.get(officer, {})

      director_dob = pd.to_datetime(f"{director.get('dd_dob_year', '')}-{director.get('dd_dob_month', '')}-01")
      age = (company_created_on - director_dob)/np.timedelta64(1, 'Y')
      directors_age.append(age)

      return int(np.average(directors_age))
    
  else:

    applicant_dob = pd.to_datetime(row['date_of_birth'])
    age = (company_created_on - applicant_dob)/np.timedelta64(1, 'Y')

    return int(age)

      
  return None

# COMMAND ----------

applicant_director_nationality = applicant_director_info.apply(lambda row: get_applicant_director_nationality(row), axis=1)

applicant_director_nationality = applicant_director_nationality.apply(lambda x: np.unique(list(filter(lambda x: x not in [None, float('nan')], x if isinstance(x, list) else [x]))).tolist())

applicant_director_nationality.shape

# COMMAND ----------

from feature_extractors.kyc.risk_extractors.risk_extractors import ApplicantIdCountry
country_mapping = ApplicantIdCountry().country_mappings

applicant_nationality_risk = applicant_director_nationality.apply(lambda x: [country_mapping.get(country, "Undefined") for country in x] if bool(x) else ["Undefined"])

applicant_nationality_risk.value_counts()

# COMMAND ----------

applicant_dd_nationality_risk = applicant_nationality_risk.apply(lambda x: "High" if "High" in x else ("Medium" if "Medium" in x else ("Low" if "Low" in x else "Undefined")))

# COMMAND ----------

applicant_dd_nationality = pd.DataFrame(applicant_director_nationality, columns=['applicant_dd_nationality'])
applicant_dd_nationality_risk = pd.DataFrame(applicant_dd_nationality_risk, columns=['applicant_dd_nationality_risk'])
applicant_dd_nationality.shape, applicant_dd_nationality_risk.shape

# COMMAND ----------

applicant_dd_nationality_risk['applicant_dd_nationality_risk'].value_counts()/applicant_dd_nationality_risk.shape[0]

# COMMAND ----------

applicant_dd_nationality

# COMMAND ----------

applicant_directors_age = applicant_director_info.apply(lambda row: get_applicant_directors_avg_age(row), axis=1)

applicant_directors_age.shape

# COMMAND ----------

applicant_directors_age = pd.DataFrame(applicant_directors_age, columns=['directors_avg_age_at_completion'])
applicant_directors_age.describe()

# COMMAND ----------

dfapp = dfapp.merge(applicant_dd_nationality, left_index=True, right_index=True).merge(applicant_dd_nationality_risk, left_index=True, right_index=True).merge(applicant_directors_age, left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

dfapp.head()

# COMMAND ----------

dfapp['applicant_dd_nationality_risk'].value_counts()/dfapp.shape[0]

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['applicant_dd_nationality_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dd_df['company_directors_count'] = dd_df['directors_info'].apply(lambda x: len(x) if bool(x) else np.nan)
dd_df['company_directors_count'] = dd_df['company_directors_count'].apply(lambda x: "1" if x == 1 else ("2" if x == 2 else ("3+" if x >= 3 else "Undefined")))
dd_df['company_directors_count'].value_counts()

# COMMAND ----------

dfapp = dfapp.merge(dd_df['company_directors_count'], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

del dfapp['company_directors_count_y']

# COMMAND ----------

dfapp.head()

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['company_directors_count'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dd_df['company_structurelevelwise'] = dd_df['structurelevelwise'].apply(lambda x: "Undefined" if pd.isnull(x) else "1" if x == "1" else ("2" if x == "2" else "3+")).astype(str)
dd_df['company_structurelevelwise'].value_counts()

# COMMAND ----------

dfapp = dfapp.merge(dd_df[['company_structurelevelwise']], left_index=True, right_index=True)
dfapp.shape

# COMMAND ----------

del dfapp['company_structurelevelwise_x']
dfapp.rename(columns = {'company_structurelevelwise_y': 'company_structurelevelwise'}, inplace=True)

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['company_structurelevelwise'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dfapp[dfapp['is_approved']==1].groupby(['company_postcode_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

dfapp.head()

# COMMAND ----------

feature_dataset = pd.read_csv(data_location + f"app_fraud_feature_dataset_{start_date}_{end_date}.csv",
                 dtype={"company_id": "str", "membership_id": "str"})
feature_dataset.set_index('company_id', inplace=True)
feature_dataset['company_created_on'] = pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: x.date())
feature_dataset.shape

# COMMAND ----------

feature_dataset.columns.tolist()

# COMMAND ----------

feature_dataset = feature_dataset.merge(dfapp[['applicant_years_to_id_expiry', 
                                               'company_directors_count', 
                                               'applicant_dd_nationality',
                                               'applicant_dd_nationality_risk', 
                                               'company_structurelevelwise', 
                                               'company_postcode',
                                               'company_postcode_risk',
                                               'company_status', 
                                               'directors_avg_age_at_completion',
                                               'applicant_name_mismatch_ind']], 
                                        left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

feature_dataset['applicant_nationality'] = feature_dataset['applicant_nationality'].fillna("Undefined")
feature_dataset['applicant_dd_nationality'] = feature_dataset['applicant_dd_nationality'].fillna("Undefined")

# COMMAND ----------

def applicant_nationality_impute_dd(row):

  output = []
  if row['applicant_nationality'].__contains__('Undefined'):
    if isinstance(row['applicant_dd_nationality'], list):
      output = list(filter(lambda x: x not in [None, float('nan')], np.unique(row['applicant_dd_nationality']).tolist()))
    else:
      output = ["Undefined"]
  
  else:
    if isinstance(row['applicant_dd_nationality'], list):
      output = list(filter(lambda x: x not in [None, float('nan')], np.unique(row['applicant_dd_nationality'].append(row['applicant_nationality'])).tolist()))
    else:
      output = list(filter(lambda x: x not in [None, float('nan')], [row['applicant_nationality']]))
  
  return output if len(output)>0 else ["Undefined"]
    

# COMMAND ----------

applicant_nationality = feature_dataset[['applicant_nationality', 'applicant_dd_nationality']].apply(lambda row: applicant_nationality_impute_dd(row), axis=1)

# COMMAND ----------

feature_dataset['applicant_nationality'] = applicant_nationality
feature_dataset.drop(columns = ['applicant_dd_nationality'], inplace=True)

# COMMAND ----------

feature_dataset['applicant_nationality'].value_counts()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset['applicant_nationality_risk'] = feature_dataset[['applicant_nationality_risk', 'applicant_dd_nationality_risk']].apply(lambda row:
  "High" if any([row['applicant_nationality_risk'] == 'High', row['applicant_dd_nationality_risk'] == 'High']) else (
    "Medium" if any([row['applicant_nationality_risk'] == 'Medium', row['applicant_dd_nationality_risk'] == 'Medium']) else (
      "Low" if any([row['applicant_nationality_risk'] == 'Low', row['applicant_dd_nationality_risk'] == 'Low']) else "Undefined"
    )
  ),
  axis=1)

feature_dataset.drop(columns=['applicant_dd_nationality_risk'], inplace=True)

# COMMAND ----------

feature_dataset['applicant_nationality_risk'].value_counts()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_nationality_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset['applicant_idcountry_idnationality_risk'] = feature_dataset[['applicant_idcountry_issue_risk', 'applicant_nationality_risk']].apply(lambda row:
  "High" if any([row['applicant_idcountry_issue_risk'] == 'High', row['applicant_nationality_risk'] == 'High']) else (
    "Medium" if any([row['applicant_idcountry_issue_risk'] == 'Medium', row['applicant_nationality_risk'] == 'Medium']) else (
      "Low" if any([row['applicant_idcountry_issue_risk'] == 'Low', row['applicant_nationality_risk'] == 'Low']) else "Undefined"
    )
  ),
  axis=1)

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_idcountry_idnationality_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

pd.isnull(feature_dataset).sum()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset['applicant_idcountry_issue'] = feature_dataset['applicant_idcountry_issue'].fillna("Undefined")
feature_dataset['applicant_postcode'] = feature_dataset['applicant_postcode'].fillna("Undefined")

# COMMAND ----------

pd.isnull(feature_dataset).sum()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset.shape

# COMMAND ----------

import os
# os.mkdir(data_location)
feature_dataset.to_csv(data_location + f"app_fraud_feature_dataset_{start_date}_{end_date}" + ".csv")

# COMMAND ----------

