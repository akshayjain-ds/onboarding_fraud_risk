# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Scoring in Real Time
# MAGIC This notebook is used to mimic the real time producton envirment, pull the data from batch but convert each row of data to json paylod and then load the model to make predictions as if its running in real time. Mostly its for testing purpose
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

appf_df = spark_connector(perpetrator_query.format(from_date=oot_start_date, to_date=oot_end_date))
appf_df = appf_df.toDF(*[c.lower().split('.')[-1] for c in appf_df.columns])
appf_df.count()

# COMMAND ----------

appf_df.show()

# COMMAND ----------

workspace_name = 'akshayjain'
raw_service_name = 'membership_completed_v_data_source'
feature_service_name = 'uk_ifre_feature_service'
my_experiment_workspace = tecton.get_workspace(workspace_name)

# COMMAND ----------

appf_df = appf_df.drop('approved_at')

# COMMAND ----------

dfapp = appf_df.toPandas()
dfapp.shape

# COMMAND ----------

dfapp = dfapp[(pd.to_datetime(dfapp['timestamp']) >= pd.to_datetime(oot_start_date)) & 
              (pd.to_datetime(dfapp['timestamp']) <= pd.to_datetime(oot_end_date))]
dfapp.shape

# COMMAND ----------

(dfapp[km_indicator]==1).sum(), dfapp[km_indicator].mean

# COMMAND ----------

dfapp.head()

# COMMAND ----------

dfapp['rules_engine_decision'] = dfapp['rules_engine_decision'].apply(lambda x: 1 if x == 'mkyc' else 0)
dfapp['risk_engine_decision'] = dfapp['risk_category'].apply(lambda x: 1 if x == 'HIGH' else 0)
dfapp['rules_engine_decision'].mean(), dfapp['risk_engine_decision'].mean()

# COMMAND ----------

dfapp.index.duplicated().sum()

# COMMAND ----------

dfapp.set_index(id1, inplace=True)


# COMMAND ----------

dfapp[km_indicator] = pd.to_numeric(dfapp[km_indicator], errors='ignore')
dfapp[target_b] = pd.to_numeric(dfapp[target_b], errors='ignore')
dfapp[target_c] = pd.to_numeric(dfapp[target_c], errors='ignore')

# COMMAND ----------

dfapp = dfapp[dfapp[target_c]>=0]
dfapp.shape

# COMMAND ----------

dfapp[dfapp[target_c]>0][[target_c]].describe(percentiles = np.linspace(0,1,101))

# COMMAND ----------

dfapp.groupby(['is_approved'])[[target_b, target_c]].mean()

# COMMAND ----------

dfapp.head()

# COMMAND ----------

dfapp[dfapp.is_app_fraud==1]

# COMMAND ----------

ds = my_experiment_workspace.get_data_source('membership_completed_v_data_source')
member_data = ds.get_dataframe(start_time=pd.to_datetime(oot_start_date), end_time = pd.to_datetime(oot_end_date)).to_spark()

# COMMAND ----------

df = member_data.persist()

# COMMAND ----------

df = df.toPandas()
df.shape

# COMMAND ----------

df[id1].duplicated().sum()

# COMMAND ----------

df.set_index(id1, inplace=True)

# COMMAND ----------

df.head()

# COMMAND ----------

df['INDIVIDUAL_CHECKS_C_NOT_PASSED'] = df['manual_approval_triggers_rawdata'].apply(lambda x: json.dumps({"value": [t.strip() for t in x.split(',')]}))
df['COMPANY_INDUSTRY_CLASSIFICATION__RawData'] = df['company_icc_rawdata'].apply(lambda x: json.dumps({"value": x}))
df['COMPANY_SIC_CODES__RawData'] = df['company_sic_rawdata'].apply(lambda x: json.dumps({"value": x.split(',') if isinstance(x, str) else []}))
df['APPLICANT_ID_COUNTRY_ISSUE__RawData'] = df['applicant_idcountry_issue_rawdata'].apply(lambda x: json.dumps({"value": x}))
df['APPLICANT_NATIONALITY__RawData'] = df['applicant_nationality_rawdata'].apply(lambda x: json.dumps({"value": x.split() if isinstance(x, str) else []}))
df['APPLICANT_email_domain__RawData'] = df['applicant_email_rawdata'].apply(lambda x: json.dumps({"value": x}))
df['APPLICANT_POSTCODE__RawData'] = df['applicant_postcode_rawdata'].apply(lambda x: json.dumps({"value": x}))
df['APPLICANT_email_domain'] = df['applicant_email_rawdata'].apply(lambda x: x.split("@")[-1]).apply(lambda x: x.split("#")[-1])
df['COMPANY_AGE_AT_COMPLETION'] = (pd.to_datetime(df['company_created_on']).apply(lambda x : x.date()) - pd.to_datetime(df['company_incorporation_date_rawdata']).apply(lambda x : x.date()))/np.timedelta64(1, 'M')

# COMMAND ----------

df['RULE_APPLICANT_idtype_pdl'] = df[['applicant_idcountry_issue_rawdata',
                                      'applicant_id_type_rawdata',
                                      'applicant_id_subtype_rawdata']].apply(lambda row: 
                                        1 if (row['applicant_id_type_rawdata']=='DRIVING_LICENSE') and 
                                        any([
                                          row['applicant_id_subtype_rawdata']=='LEARNING_DRIVING_LICENSE',
                                             all([
                                             row['applicant_idcountry_issue_rawdata'] in ['GB', 'GBR'],
                                             row['applicant_id_subtype_rawdata'] in [["none", "None", "null", None, np.NaN]]
                                             ])]) else 0, axis=1)
df['RULE_APPLICANT_idtype_brp'] = df[['applicant_id_type_rawdata',
                                      'applicant_id_subtype_rawdata']].apply(lambda row: 
                                        1 if all([
                                          row['applicant_id_type_rawdata']=='ID_CARD',
                                          row['applicant_id_subtype_rawdata']=='RESIDENT_PERMIT_ID',
                                             ]) else 0, axis=1)
df[['RULE_APPLICANT_idtype_pdl', 'RULE_APPLICANT_idtype_brp']].mean()

# COMMAND ----------

df.head()

# COMMAND ----------

fs = my_experiment_workspace.get_feature_service(feature_service_name)
feature_dataset = fs.get_historical_features(spine=appf_df, from_source=True).to_spark()
type(feature_dataset)

# COMMAND ----------

feature_dataset = feature_dataset.persist()

# COMMAND ----------

feature_dataset = feature_dataset.toPandas()
feature_dataset.rename(columns={col: col.split('__')[-1] for col in list(feature_dataset.columns)}, inplace=True)
feature_dataset.set_index('company_id', inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

pd.isnull(feature_dataset).sum()

# COMMAND ----------

feature_dataset['days_to_app_fraud'] = feature_dataset['days_to_app_fraud'].fillna(9999)            
feature_dataset['days_fraud_to_reject'] = feature_dataset['days_fraud_to_reject'].fillna(9999)   
feature_dataset.dropna(inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset = feature_dataset[feature_dataset[target_c]>=0]
feature_dataset = feature_dataset[feature_dataset['days_to_app_fraud']>=0]
feature_dataset.shape

# COMMAND ----------

ifre_dataset = feature_dataset[['age_at_completion', 
                                'applicant_postcode', 
                                'applicant_idcountry_issue',
                                'applicant_nationality',
                                'company_keywords',
                                'company_type',
                                'applicant_id_type',
                                'company_nob']].copy()
for col in ifre_dataset.columns:
  ifre_dataset[f'{col}'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))
ifre_dataset.head()

# COMMAND ----------

ifre_features = pd.DataFrame(index=ifre_dataset.index)
ifre_features['AGE_AT_COMPLETION'] = ifre_dataset['age_at_completion']
ifre_features['APPLICANT_ID_COUNTRY_ISSUE_High'] = ifre_dataset['applicant_idcountry_issue'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['APPLICANT_ID_COUNTRY_ISSUE_Medium'] = ifre_dataset['applicant_idcountry_issue'].apply(lambda x: 1 if x == 'Medium' else 0)
ifre_features['APPLICANT_ID_COUNTRY_ISSUE_Low'] = ifre_dataset['applicant_idcountry_issue'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features['APPLICANT_ID_NATIONALITY_High'] = ifre_dataset['applicant_nationality'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['APPLICANT_ID_NATIONALITY_Medium'] = ifre_dataset['applicant_nationality'].apply(lambda x: 1 if x == 'Medium' else 0)
ifre_features['APPLICANT_ID_NATIONALITY_Low'] = ifre_dataset['applicant_nationality'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features['APPLICANT_POSTCODE_High'] = ifre_dataset['applicant_postcode'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['APPLICANT_POSTCODE_Medium'] = ifre_dataset['applicant_postcode'].apply(lambda x: 1 if x == 'Medium' else 0)
ifre_features['APPLICANT_POSTCODE_Low'] = ifre_dataset['applicant_postcode'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features['APPLICANT_ID_TYPE_Driving_Licence'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'Driving_Licence' else 0)
ifre_features['APPLICANT_ID_TYPE_Provisional_Licence'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'Provisional_Licence' else 0)
ifre_features['APPLICANT_ID_TYPE_Passport'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'Passport' else 0)
ifre_features['APPLICANT_ID_TYPE_National_ID'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'National_ID' else 0)
ifre_features['APPLICANT_ID_TYPE_Residence_Permit'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'Residence_Permit' else 0)
ifre_features['APPLICANT_ID_TYPE_Other_ID'] = ifre_dataset['applicant_id_type'].apply(lambda x: 1 if x == 'Other_ID' else 0)
ifre_features['COMPANY_TYPE_Prohibited'] = ifre_dataset['company_type'].apply(lambda x: 1 if x == 'Prohibited' else 0)
ifre_features['COMPANY_TYPE_High'] = ifre_dataset['company_type'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['COMPANY_TYPE_Low'] = ifre_dataset['company_type'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features['COMPANY_KEYWORDS_Prohibited'] = ifre_dataset['company_keywords'].apply(lambda x: 1 if x == 'Prohibited' else 0)
ifre_features['COMPANY_KEYWORDS_High'] = ifre_dataset['company_keywords'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['COMPANY_KEYWORDS_Medium'] = ifre_dataset['company_keywords'].apply(lambda x: 1 if x == 'Medium' else 0)
ifre_features['COMPANY_KEYWORDS_Low'] = ifre_dataset['company_keywords'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features['COMPANY_NOB_Prohibited'] = ifre_dataset['company_nob'].apply(lambda x: 1 if x == 'Prohibited' else 0)
ifre_features['COMPANY_NOB_High'] = ifre_dataset['company_nob'].apply(lambda x: 1 if x == 'High' else 0)
ifre_features['COMPANY_NOB_Medium'] = ifre_dataset['company_nob'].apply(lambda x: 1 if x == 'Medium' else 0)
ifre_features['COMPANY_NOB_Low'] = ifre_dataset['company_nob'].apply(lambda x: 1 if x == 'Low' else 0)
ifre_features.head()

# COMMAND ----------

feature_dataset = ifre_features.merge(dfapp[[
  'rules_engine_decision',
  'risk_engine_decision'
  ]], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset = feature_dataset.merge(df[[
  'INDIVIDUAL_CHECKS_C_NOT_PASSED',
  'COMPANY_INDUSTRY_CLASSIFICATION__RawData',
  'COMPANY_SIC_CODES__RawData',
  'APPLICANT_ID_COUNTRY_ISSUE__RawData',
  'APPLICANT_NATIONALITY__RawData',
  'APPLICANT_email_domain__RawData',
  'APPLICANT_POSTCODE__RawData',
  'APPLICANT_email_domain',
  'COMPANY_AGE_AT_COMPLETION',
  'RULE_APPLICANT_idtype_pdl',
  'RULE_APPLICANT_idtype_brp'
  ]], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset['rules_engine_decision'] = feature_dataset[['rules_engine_decision', 'RULE_APPLICANT_idtype_pdl', 'RULE_APPLICANT_idtype_brp']].max(axis=1)

# COMMAND ----------

feature_dataset.drop(columns=['RULE_APPLICANT_idtype_pdl', 'RULE_APPLICANT_idtype_brp'], 
                     inplace=True)

# COMMAND ----------

feature_dataset.tail()

# COMMAND ----------

feature_dataset = dfapp[['member_id', 'timestamp', 'risk_category',
       'is_approved','is_app_fraud', 'days_to_app_fraud', 'days_fraud_to_reject',
       'app_fraud_amount', 'app_fraud_type']].merge(feature_dataset, left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.columns

# COMMAND ----------

feature_dataset['rules_engine_decision'].mean()

# COMMAND ----------

feature_dataset[date_feature] = feature_dataset['timestamp']

# COMMAND ----------

pd.isnull(feature_dataset).sum()

# COMMAND ----------

member_df = spark_connector(member_query.format(from_date=oot_start_date, to_date=oot_end_date))
member_df = member_df.toDF(*[c.lower().split('.')[-1] for c in member_df.columns])
member_df.count()

# COMMAND ----------

member_df = member_df.toPandas()
member_df[id1].duplicated().sum()

# COMMAND ----------

if member_df[id1].duplicated().sum() == 0:
  member_df.set_index(id1, inplace=True)
  print(f"{id1} set as index")
member_df.shape

# COMMAND ----------

member_df['APPLICANT_YEARS_TO_ID_EXPIRY'] = member_df['applicant_years_to_id_expiry']
member_df['APPLICANT_YEARS_TO_ID_EXPIRY'] = np.clip(member_df['APPLICANT_YEARS_TO_ID_EXPIRY'], a_min=0, a_max=10)
feature_dataset = feature_dataset.merge(member_df[['APPLICANT_YEARS_TO_ID_EXPIRY']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

dd_df = spark_connector(duedil_query.format(from_date=oot_start_date, to_date='2024-06-30'))
dd_df = dd_df.toDF(*[c.lower().split('.')[-1] for c in dd_df.columns])
dd_df.count()

# COMMAND ----------

dd_df = dd_df.toPandas()
dd_df.head()

# COMMAND ----------

dd_df[id1].duplicated().sum()

# COMMAND ----------

dd_df.set_index(id1, inplace=True)

# COMMAND ----------

dd_df['duedil_hit'].mean()

# COMMAND ----------

feature_dataset = feature_dataset.merge(dd_df[['duedil_hit']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

dd_df['COMPANY_POSTCODE__RawData'] = dd_df['company_postcode'].apply(lambda x: json.dumps({"value": x}))

# COMMAND ----------

feature_dataset = feature_dataset.merge(dd_df[['COMPANY_POSTCODE__RawData']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

dd_df['company_structurelevelwise'] = dd_df['structurelevelwise'].apply(lambda x: "Undefined" if pd.isnull(x) else "1" if x == "1" else ("2" if x == "2" else "3+")).astype(str)
dd_df['company_structurelevelwise'].value_counts()

# COMMAND ----------

dd_df['COMPANY_STRUCTURE_LEVELWISE_1'] = dd_df['company_structurelevelwise'].apply(lambda x: 1 if x =='1' else 0)
dd_df['COMPANY_STRUCTURE_LEVELWISE_2'] = dd_df['company_structurelevelwise'].apply(lambda x: 1 if x =='2' else 0)
dd_df['COMPANY_STRUCTURE_LEVELWISE_3+'] = dd_df['company_structurelevelwise'].apply(lambda x: 1 if x =='3+' else 0)

# COMMAND ----------

feature_dataset = feature_dataset.merge(dd_df[['COMPANY_STRUCTURE_LEVELWISE_1',
                                               'COMPANY_STRUCTURE_LEVELWISE_2',
                                               'COMPANY_STRUCTURE_LEVELWISE_3+']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

# feature_dataset_original = feature_dataset.copy()
# feature_dataset = feature_dataset_original.copy()

# # if email and structurelevelwise are not available
# feature_dataset['COMPANY_STRUCTURE_LEVELWISE_1'] = 0
# feature_dataset['COMPANY_STRUCTURE_LEVELWISE_2'] = 0
# feature_dataset['COMPANY_STRUCTURE_LEVELWISE_3+'] = 0
# feature_dataset['APPLICANT_email_domain__RawData'] = '{"value": "example@domain.com"}'
# feature_dataset['APPLICANT_email_domain'] = 'domain.com'

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

feature_dataset['year_month'] = feature_dataset[date_feature].apply(lambda x: str(x.date())[:-3])

# COMMAND ----------

feature_dataset[feature_dataset[km_indicator]==1].groupby(['year_month'])[[target_b]].mean()

# COMMAND ----------

feature_dataset.groupby(['year_month'])[['duedil_hit']].mean()

# COMMAND ----------

feature_dataset[km_indicator].mean(), feature_dataset[km_indicator].sum()

# COMMAND ----------

days_to_app = feature_dataset[(feature_dataset['year_month'] <= '2024-02') & (feature_dataset['is_app_fraud']==1)][['days_to_app_fraud', 'is_app_fraud', 'app_fraud_amount']]
days_to_app.shape

# COMMAND ----------

sum_count, sum_amount = (days_to_app[['is_app_fraud', 'app_fraud_amount']].sum().ravel())
sum_count, sum_amount

# COMMAND ----------

days_to_app.sort_values(by=['days_to_app_fraud'], inplace=True)
days_to_app.head()

# COMMAND ----------

days_to_app['cum_is_app_fraud'] = 100*days_to_app['is_app_fraud'].cumsum()/sum_count
days_to_app['cum_app_fraud_amount'] = 100*days_to_app['app_fraud_amount'].cumsum()/sum_amount
days_to_app.shape

# COMMAND ----------

days_to_app.head()

# COMMAND ----------

fig = plt.figure(1, figsize=(10, 10))
plt.plot(days_to_app['days_to_app_fraud'], days_to_app['cum_is_app_fraud'], 'b', 
         label = 'Member APP Fraud 180 days post approval')
plt.plot(days_to_app['days_to_app_fraud'], days_to_app['cum_app_fraud_amount'], 'g', 
         label = 'Amount APP Fraud 180 days post approval')
plt.legend(loc='lower right')
plt.xlim([0, 180])
plt.ylim([0, 100])
plt.xlabel('days to fraud')
plt.ylabel("cumulative % fraud")
plt.xticks(np.arange(0, 190, step=10))
plt.yticks(np.arange(0, 105, step=5))
plt.grid()
plt.show()

# COMMAND ----------

feature_dataset[[*input_feature_names_from_payload()]]

# COMMAND ----------

pd.isnull(feature_dataset).sum()/feature_dataset.shape[0]

# COMMAND ----------

payload = json.loads(feature_dataset[[*input_feature_names_from_payload()]].iloc[102].to_json())
payload

# COMMAND ----------

# feature_dataset.to_csv(data_location + f"app_fraud_feature_dataset_{val_start_date}_{val_end_date}" + ".csv")

# COMMAND ----------

feature_dataset = pd.read_csv(data_location + f"app_fraud_feature_dataset_{val_start_date}_{val_end_date}.csv",
                 dtype={id1: "str", id2: "str"})
feature_dataset.set_index(id1, inplace=True)
feature_dataset[date_feature] = pd.to_datetime(feature_dataset[date_feature]).apply(lambda x: x.date())
feature_dataset['APPLICANT_ID_TYPE_None'] = 0
feature_dataset.shape

# COMMAND ----------

# import mlflow
# logged_model =f'runs:/6f5b18830a814fda8e6b46b5d85f49fa/uk_risk_engine_2024'
# mlflow_ifre_engine = mlflow.pyfunc.load_model(logged_model)
# mlflow_ifre_engine

# COMMAND ----------

import mlflow
logged_model =f'runs:/043d58444c24452faabfcb6d3a0d3e22/uk_NTT_MP_at_onboarding' 
mlflow_ntt_mp_engine = mlflow.pyfunc.load_model(logged_model)
mlflow_ntt_mp_engine

# COMMAND ----------

model = mlflow_ntt_mp_engine.unwrap_python_model().model
model

# COMMAND ----------

import sklearn
sss = sklearn.model_selection.StratifiedShuffleSplit(1, random_state=111, test_size=0.05)

# COMMAND ----------

for i, (train_index, test_index) in enumerate(sss.split(feature_dataset[[*input_feature_names_from_payload()]], feature_dataset['year_month'])): 
  print(f"Fold {i}:")
  print(f"  Train: index={train_index.shape}")
  print(f"  Test:  index={test_index.shape}")

# COMMAND ----------

# ifre_decisions = feature_dataset.iloc[test_index, :].progress_apply(lambda row: mlflow_ifre_engine.predict(row), axis=1, result_type='expand')

# COMMAND ----------

decisions = feature_dataset.iloc[test_index, :][[*input_feature_names_from_payload()] + ['APPLICANT_ID_TYPE_None']].progress_apply(lambda row: mlflow_ntt_mp_engine.predict(row), axis=1, result_type='expand')

# COMMAND ----------

decisions = decisions.merge(feature_dataset[['rules_engine_decision', 'risk_engine_decision', 
                                             km_indicator, target_b, target_c, 'year_month']], 
                            left_index=True, right_index=True)
decisions.shape

# COMMAND ----------

decisions.head()

# COMMAND ----------

decisions['raw_rating'].mean()

# COMMAND ----------

decisions['category'].value_counts()/decisions.shape[0]


# COMMAND ----------

decisions.groupby(['category'])[[target_b]].mean()*100

# COMMAND ----------

decisions[target_b].sum()

# COMMAND ----------

decisions.groupby(['category'])[[target_b]].mean()/decisions[target_b].mean()

# COMMAND ----------

np.around((100*decisions.pivot_table(index=['year_month'], columns=['category'], values = [target_b], aggfunc=['count'])/decisions.groupby(['year_month'])[['category']].count().values), decimals=2)

# COMMAND ----------

decisions[decisions['category']=="HIGH"]['reason'].value_counts()/decisions[decisions['category']=="HIGH"].shape[0]

# COMMAND ----------

# mkyc rate: 
decisions[['rules_engine_decision', 'risk_engine_decision', 'decision']].mean()

# COMMAND ----------

# mkyc rate: IFRE intersetion with NtT MP at Onboarding
decisions[(decisions['risk_engine_decision']==1) & 
          (decisions['decision']=='EDD')].shape[0]/decisions.shape[0]

# COMMAND ----------

# mkyc rate: IFRE union with NtT MP at Onboarding
decisions[(decisions['risk_engine_decision']==1) | 
          (decisions['decision']=='EDD')].shape[0]/decisions.shape[0]

# COMMAND ----------

# mkyc rate: IFRE union with Rule Engine
decisions[(decisions['risk_engine_decision']==1) | 
          (decisions['rules_engine_decision']==1)].shape[0]/decisions.shape[0]

# COMMAND ----------

# mkyc rate: IFRE union with Rule Engine union with  NtT MP at Onboarding
decisions[(decisions['risk_engine_decision']==1) | 
          (decisions['rules_engine_decision']==1) | 
          (decisions['decision']=='EDD')].shape[0]/decisions.shape[0]

# COMMAND ----------

model.feature_df[['APPLICANT_ID_COUNTRY_ISSUE__RawData', 'applicant_idcountry_issue', 'applicant_idcountry_issue_encoded']]

# COMMAND ----------

model.feature_df[['APPLICANT_ID_TYPE_National_ID', 'applicant_id_type', 'applicant_id_type_encoded']]

# COMMAND ----------

model.feature_df[['APPLICANT_NATIONALITY__RawData', 'applicant_nationality', 'applicant_nationality_0', 'applicant_nationality_encoded']]

# COMMAND ----------


model.feature_df[['APPLICANT_POSTCODE__RawData', 'applicant_postcode', 'applicant_postcode_encoded']]

# COMMAND ----------

model.feature_df[['APPLICANT_email_domain__RawData', 'applicant_email_domain', 'applicant_email_domain_encoded']]

# COMMAND ----------

model.feature_df[['COMPANY_INDUSTRY_CLASSIFICATION__RawData', 'company_icc', 'company_icc_encoded']]

# COMMAND ----------

model.feature_df[['COMPANY_SIC_CODES__RawData', 'company_sic', 'company_sic_encoded']]

# COMMAND ----------

model.feature_df[['company_nob', 'company_nob_encoded']].values

# COMMAND ----------

appf_decisions = feature_dataset[feature_dataset[target_b]==1][[*input_feature_names_from_payload()] + ['APPLICANT_ID_TYPE_None']].progress_apply(lambda row: mlflow_ntt_mp_engine.predict(row), axis=1, result_type='expand')

# appf_decisions = feature_dataset[feature_dataset[target_b]==1][[*input_feature_names_from_payload()]].progress_apply(lambda row: mlflow_ntt_mp_engine.predict(row), axis=1, result_type='expand')

# COMMAND ----------

appf_decisions = appf_decisions.merge(feature_dataset[['rules_engine_decision', 'risk_engine_decision', 
                                             km_indicator, target_b, target_c, 'app_fraud_type', 'year_month']], 
                            left_index=True, right_index=True)
appf_decisions.shape

# COMMAND ----------

cv = CountVectorizer(min_df=1, max_df=1.0, binary=True, lowercase=True, 
                     vocabulary=['purchase', 
                                'impersonation',
                                'investment',
                                'advance',
                                'romance',
                                'invoice',
                                'ceo',
                                '2nd',
                                'unknown'])
cv.fit(appf_decisions['app_fraud_type'].apply(lambda x: x.lower()))

# COMMAND ----------

scams_df = pd.DataFrame(cv.transform(appf_decisions['app_fraud_type']).toarray(), 
                        columns=cv.vocabulary,
                        index=appf_decisions.index)
scams_df.mean()

# COMMAND ----------

appf_decisions[cv.vocabulary]=scams_df
appf_decisions

# COMMAND ----------

appf_decisions.groupby(['category'])[[target_b]].sum()/appf_decisions[target_b].sum()

# COMMAND ----------

for scam in cv.vocabulary:
  print(scam, 
        np.around(appf_decisions[scam].mean()*100, decimals=2), 
        np.around(100*appf_decisions[appf_decisions['category']=='HIGH'][scam].sum()/appf_decisions[scam].sum(), decimals=2),
        np.around(100*appf_decisions[(appf_decisions['category']=='HIGH') & (appf_decisions[scam]==1)][target_c].sum()/appf_decisions[appf_decisions[scam]==1][target_c].sum(), decimals=2))

# COMMAND ----------

appf_decisions.groupby(['category'])[[target_c]].sum()/appf_decisions[target_c].sum()

# COMMAND ----------

np.around((100*appf_decisions.pivot_table(index=['year_month'], columns=['category'], values = [target_b], aggfunc=['count'])/appf_decisions.groupby(['year_month'])[['category']].count().values), decimals=2)

# COMMAND ----------

appf_decisions[appf_decisions['category']=="HIGH"]['reason'].value_counts()/appf_decisions[appf_decisions['category']=="HIGH"].shape[0]

# COMMAND ----------

