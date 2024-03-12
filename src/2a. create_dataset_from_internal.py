# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Feature 1
# MAGIC This notebook is used used do pull level 1 of features (from customer/company onboarding data)

# COMMAND ----------



# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./pre-processing

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

(dfapp['is_approved']==1).sum()

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

dfapp = dfapp[dfapp['app_fraud_amount']>=0]
dfapp.shape

# COMMAND ----------

dfapp[dfapp['app_fraud_amount']>0][['app_fraud_amount']].describe(percentiles = np.linspace(0,1,101))

# COMMAND ----------

dfapp.groupby(['is_approved'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

workspace_name = 'akshayjain'
raw_service_name = 'membership_completed_v_data_source'
feature_service_name = 'uk_ifre_feature_service'
my_experiment_workspace = tecton.get_workspace(workspace_name)

# COMMAND ----------

ds = my_experiment_workspace.get_data_source('membership_completed_v_data_source')
member_data = ds.get_dataframe(start_time=pd.to_datetime('2021-10-01'), end_time = pd.to_datetime(end_date)).to_spark()

# COMMAND ----------

df = member_data.persist()

# COMMAND ----------

df = df.toPandas()
df.shape

# COMMAND ----------

df['company_id'].duplicated().sum()

# COMMAND ----------

df.set_index('company_id', inplace=True)

# COMMAND ----------

df.head()

# COMMAND ----------

pd.isnull(df).sum()/df.shape[0]

# COMMAND ----------

df['applicant_idcountry_issue'] = df['applicant_idcountry_issue_rawdata'].apply(lambda x: validate_and_get_country(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x)
df['applicant_nationality'] = df['applicant_nationality_rawdata'].apply(lambda x: validate_and_get_country(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x)

# COMMAND ----------

df['applicant_postcode'] = df['applicant_postcode_rawdata'].apply(lambda x: validate_and_get_postcode(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x).apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x)))

# COMMAND ----------

df['company_sic'] = df['company_sic_rawdata'].apply(lambda x: validate_and_get_company_sic(x)).apply(lambda x: ["Undefined"] if str(x).__contains__("Error") or str(x).__contains__("[]") else x)

# COMMAND ----------

# High/Prohibited present
df[['is_restricted_keyword_present', 'company_keywords']] = df[['company_trading_name_rawdata']].apply(lambda x: get_keywords(x['company_trading_name_rawdata'].lower().strip()), axis=1, result_type='expand')
df['is_restricted_keyword_present'] = df['is_restricted_keyword_present']*1
df['is_restricted_keyword_present'].mean()

# COMMAND ----------

df['company_icc'] = df['company_icc_rawdata'].apply(lambda x: validate_and_get_company_icc(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x)
df['company_icc'].value_counts()

# COMMAND ----------

df['company_type'] = df[['company_type_rawdata', 'company_is_registered_rawdata']].apply(lambda row: validate_and_get_company_type(row['company_type_rawdata'], row['company_is_registered_rawdata']), axis=1).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x)
df['company_type'].value_counts()

# COMMAND ----------

df['company_age_at_completion'] = (pd.to_datetime(df['company_created_on']).apply(lambda x : x.date()) - pd.to_datetime(df['company_incorporation_date_rawdata']).apply(lambda x : x.date()))/np.timedelta64(1, 'M')
df['company_age_at_completion'].describe()

# COMMAND ----------

pd.isnull(df['company_age_at_completion']).sum()/df.shape[0]

# COMMAND ----------

# df['age_of_company_bucket'] = df['age_of_company'].apply(lambda x: "Undefined" if pd.isnull(x) else ("< 1M" if x <= 1 else ("< 6M" if x < 6 else ("< 12M" if x < 12 else ">= 12M"))))
# df['age_of_company_bucket'].value_counts()

# COMMAND ----------

df['applicant_device_type'] = df['applicant_device_type_rawdata'].apply(lambda x: str(x).lower())
df['applicant_device_type'].value_counts()

# COMMAND ----------

df['applicant_email_numeric'] = df['applicant_email_rawdata'].apply(lambda x: any(re.findall(r'\d+', x)))*1
df['applicant_email_numeric'].value_counts()/df.shape[0]

# COMMAND ----------

df['applicant_email_domain'] = df['applicant_email_rawdata'].apply(lambda x: x.split("@")[-1]).apply(lambda x: x.split("#")[-1])
df['applicant_email_domain'].value_counts()/df.shape[0]

# COMMAND ----------

# rule_applicant_singlename
rule_applicant_singlename = df[['applicant_id_firstname_rawdata', 'applicant_id_lastname_rawdata']].apply(lambda row: all([pd.isnull(row['applicant_id_firstname_rawdata']), 
                                                                                                            pd.isnull(row['applicant_id_lastname_rawdata'])])
                                                                                                       or
                                                                                                       all([bool(row['applicant_id_firstname_rawdata']), 
                                                                                                            bool(row['applicant_id_lastname_rawdata'])])
                                                                                                            , axis=1)
rule_applicant_singlename = ~rule_applicant_singlename                                                                                                           
rule_applicant_singlename.mean()                                                                                                   

# COMMAND ----------

# rule_industry_animal_breeder
rule_industry_animal_breeder = df['company_icc_rawdata'].apply(lambda x: str(x).lower().__contains__("animal_breeder"))
rule_industry_animal_breeder.mean()

# COMMAND ----------

# rule_idcountry_russia
rule_idcountry_russia = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("RU"), str(x).upper().__contains__("RUS")]))
rule_idcountry_russia.mean()

# COMMAND ----------


# rule_idcountry_ukraine
rule_idcountry_ukraine = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("UA"), str(x).upper().__contains__("UKR")]))
rule_idcountry_ukraine.mean()

# COMMAND ----------

# rule_idcountry_belarus
rule_idcountry_belarus = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("BY"), str(x).upper().__contains__("BLR")]))
rule_idcountry_belarus.mean()

# COMMAND ----------

# rule_idcountry_romania
feature_list = ['applicant_idcountry_issue_rawdata', 'applicant_id_type_rawdata', 'company_type_rawdata', 
                'applicant_postcode_rawdata', 'company_icc_rawdata', 'company_sic_rawdata']

rule_idcountry_romania = df[feature_list].apply(lambda row: all([any([str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("RO"), 
                                                                str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("ROU")])
                                                                 ,
                                                                  str(row['applicant_id_type_rawdata']).__contains__("PASSPORT"),
                                                                  any([str(row['company_type_rawdata']).__contains__("LTD"), 
                                                                       str(row['company_type_rawdata']).__contains__("null"), 
                                                                       str(row['company_type_rawdata']).__contains__("None")])
                                                                ])
                                                                and
                                                                any([
                                                                any([str(row['applicant_postcode_rawdata']).startswith("E"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("B"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("IP"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("IG"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("ST"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("CV"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("PR1"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("PR2")])
                                                                ,
                                                                any([str(row["company_icc_rawdata"]).strip().__contains__("category.construction"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.cleaner"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.transport_and_storage"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.decorator_\u0026_painter"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.painter_\u0026_decorator"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.furniture_removal"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.domestic_cleaner"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.household_cleaning_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.industrial_cleaning_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.hygiene_and_cleansing_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.carpenter_/_carpentry"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.hygiene_and_cleansing_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.plumbing_/_plumber"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.builder")])
                                                                ,
                                                                any([str(row['company_sic_rawdata']).__contains__("59111"),
                                                                    str(row['company_sic_rawdata']).__contains__("96090"),
                                                                    str(row['company_sic_rawdata']).__contains__("59112"),
                                                                    str(row['company_sic_rawdata']).__contains__("59200"),
                                                                    str(row['company_sic_rawdata']).__contains__("74209"),
                                                                    str(row['company_sic_rawdata']).__contains__("74202"),
                                                                    str(row['company_sic_rawdata']).__contains__("74201")
                                                                    ])
                                                            ])
                                                              , axis=1)
rule_idcountry_romania.mean()

# COMMAND ----------

# rule_idcountry_portugal
feature_list = ['applicant_idcountry_issue_rawdata', 'applicant_id_type_rawdata', 'company_is_registered_rawdata', 
                'applicant_postcode_rawdata', 'applicant_email_rawdata', 'applicant_device_type_rawdata']

rule_idcountry_portugal = df[feature_list].apply(lambda row: all([
                                                                any([str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("PT"), 
                                                                     str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("PRT")])
                                                                ,
                                                                any([str(row['applicant_id_type_rawdata']).__contains__("PASSPORT"),
                                                                    str(row['applicant_id_type_rawdata']).__contains__("ID_CARD")])
                                                                ,
                                                                row['company_is_registered_rawdata']
                                                                ,
                                                                any([str(row['applicant_postcode_rawdata']).startswith("M3"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M6"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M7"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M8"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G20"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G21"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G22"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G31"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G51"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G51"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("ML1"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("PA1")])
                                                                ,
                                                                str(row['applicant_email_rawdata']).lower().__contains__("@gmail.com")
                                                                ,
                                                                str(row['applicant_device_type_rawdata']).lower().__contains__("ios")
                                                                ])
                                                              ,axis=1)
rule_idcountry_portugal.mean()

# COMMAND ----------

# rule_company_bank
feature_list = ['company_trading_name_rawdata', 'company_sic_rawdata', 'company_icc_rawdata']
rule_company_bank = df[feature_list].apply(lambda row: any([str(row['company_trading_name_rawdata']).lower().__contains__("bank"), 
                                                                                                 str(row['company_trading_name_rawdata']).lower().__contains__("banking"),
                                                                                                 str(row['company_trading_name_rawdata']).lower().__contains__("banker"),
                                                                                                 str(row['company_icc_rawdata']).lower().__contains__("category.bank_(the_business_you_own_is_a_bank)"),
                                                                                                 str(row['company_sic_rawdata']).lower().__contains__("64191"), 
                                                                                                 str(row['company_sic_rawdata']).lower().__contains__("64110")]),
                                                                                axis=1)
rule_company_bank.mean()

# COMMAND ----------

# rule_company_bank
rule_company_bank = df['company_trading_name_rawdata'].apply(lambda x: any([str(x).lower().__contains__("bank"), 
                                                                                  str(x).lower().__contains__("banking"),
                                                                                  str(x).lower().__contains__("banker"),
                                                                                  str(x).lower().__contains__("category.bank_(the_business_you_own_is_a_bank)"),
                                                                                  str(x).lower().__contains__("64191"), 
                                                                                  str(x).lower().__contains__("64110")]))
rule_company_bank.mean()

# COMMAND ----------

rules_dataset = df[['company_created_on']]
rules_dataset['rule_applicant_singlename'] = rule_applicant_singlename*1
rules_dataset['rule_industry_animal_breeder'] = rule_industry_animal_breeder*1
rules_dataset['rule_idcountry_russia'] = rule_idcountry_russia*1
rules_dataset['rule_idcountry_ukraine'] = rule_idcountry_ukraine*1
rules_dataset['rule_idcountry_belarus'] = rule_idcountry_belarus*1
rules_dataset['rule_idcountry_romania'] = rule_idcountry_romania*1
rules_dataset['rule_idcountry_portugal'] = rule_idcountry_portugal*1
rules_dataset['rule_company_bank'] = rule_company_bank*1
rules_dataset['rc_rule_in_fbr'] = rules_dataset.max(axis=1)
rules_dataset['count_rc_rule_in_fbr'] = rules_dataset.sum(axis=1)
rules_dataset.drop(columns = ['company_created_on'], inplace=True)
rules_dataset.shape

# COMMAND ----------

rules_dataset.head()

# COMMAND ----------

rules_dataset.rc_rule_in_fbr.mean(), rules_dataset.count_rc_rule_in_fbr.mean()

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

feature_dataset.dropna(inplace=True)
feature_dataset.shape

# COMMAND ----------

for col in [
  'age_at_completion',
  'applicant_postcode',
  'applicant_idcountry_issue',
  'applicant_nationality',
  'company_keywords',
  'company_type',
  'applicant_id_type',
  'company_icc',
  'company_sic',
  'company_nob',
  'manual_approval_triggers',
  'manual_approval_triggers_unexpected',
  'company_industry_bank',
  'company_keywords_bank',
  'applicant_fraud_pass',
  'director_fraud_pass',
  'shareholder_fraud_pass']:
  
  # feature_dataset[f'{col}_input'] = feature_dataset[col].apply(lambda x: json.loads(x).get("input"))
  # feature_dataset[f'{col}_value'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))
  # feature_dataset[f'{col}_error'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("error"))
  # feature_dataset.drop(columns=[col], inplace=True)
  if col in ['applicant_postcode',  
             'applicant_idcountry_issue',
             'applicant_nationality',
             'company_keywords',
             'company_type',
             'company_icc',
             'company_sic',
             'company_nob']:
    feature_dataset[f'{col}_risk'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))
    feature_dataset.drop(columns=[f'{col}'], inplace=True)

  else:
    feature_dataset[f'{col}'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))

feature_dataset.drop(columns=['timestamp'], inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.tail()

# COMMAND ----------

feature_dataset['is_app_fraud'] = feature_dataset['is_app_fraud'].astype(int)
feature_dataset['app_fraud_amount'] = feature_dataset['app_fraud_amount'].astype(float)

# COMMAND ----------

feature_dataset = feature_dataset[feature_dataset['app_fraud_amount'] >= 0]
feature_dataset.shape

# COMMAND ----------

feature_dataset = feature_dataset.merge(rules_dataset, left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset['fraud_fail'] = 1 - feature_dataset[['applicant_fraud_pass', 'director_fraud_pass', 'shareholder_fraud_pass']].min(axis=1)*1
feature_dataset['count_rc_rule_in_fbr'] = feature_dataset['count_rc_rule_in_fbr'] + feature_dataset['fraud_fail']


# COMMAND ----------

feature_dataset.drop(columns = [ 'applicant_fraud_pass', 'director_fraud_pass', 'shareholder_fraud_pass', 
                                'company_industry_bank','company_keywords_bank', 
                                'company_icc_risk', 'company_sic_risk'
                                ], inplace=True)


# COMMAND ----------

feature_dataset = feature_dataset.merge(df[[
  'applicant_idcountry_issue', 'applicant_nationality', 'is_restricted_keyword_present', 
  'applicant_postcode', 'company_sic', 'company_icc', 'company_type',
  'applicant_device_type', 'applicant_email_domain', 'applicant_email_numeric', 
  'company_age_at_completion'
  ]], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.tail()

# COMMAND ----------

feature_dataset['applicant_nationality_risk'].value_counts()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_nationality_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_idcountry_issue_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['company_age_at_completion'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_device_type'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset['applicant_email_domain'] = feature_dataset['applicant_email_domain'].apply(lambda x: x if x in ['gmail.com',
 'hotmail.com',
 'outlook.com',
 'yahoo.com',
 'hotmail.co.uk',
 'icloud.com',
 'yahoo.co.uk',
 'live.co.uk'] else "other")

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_email_domain'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_postcode_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['company_nob_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['company_keywords_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['company_type_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['applicant_id_type'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['company_nob_risk'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['fraud_fail'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['rc_rule_in_fbr'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['is_restricted_keyword_present'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

mat = feature_dataset['manual_approval_triggers'].apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_BLOCKLIST ') for st in ['bl001', 'bl002', 'bl004', 'bl005', 'bl006', 'bl007', 'bl008']])[0])

mat = mat.apply(lambda x: str(x).upper().replace('applicant_address_fail'.upper(), ' INDIVIDUAL_IDENTITY_ADDRESS '))

mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_SANCTIONS_PEP ') for st in ['applicant_pep', 'sanction_fail']])[0])


mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_ID_SCAN ') for st in ['facematch_verification_failed', 'id_scan_images_download_failed', 
                                                                                                                       'id_scan_verification_failed', 'idscan_mismatch']])[0])

mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' BUSINESS_INTERNAL_CHECKS ') for st in ['missing_country_shareholder', 'director_mismatch',
                                                                                                                             'missing_number_or_name_registered_address', 
                                                                                                                             'missing_shareholder', 'missing_sic_codes']])[0])

                                                                                                       

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer

# COMMAND ----------

triggers_to_use = ['individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks']

triggers_vec = CountVectorizer(binary=True, vocabulary=triggers_to_use)                                              
triggers_vec.fit(mat)


# COMMAND ----------

triggers_df = pd.DataFrame(triggers_vec.transform(mat).toarray(), 
                           columns=triggers_vec.get_feature_names_out().tolist(), 
                           index=feature_dataset.index)

triggers_df.shape

# COMMAND ----------

triggers_df.mean()

# COMMAND ----------

import seaborn as sb
heat = sb.heatmap(triggers_df.corr())

# COMMAND ----------

feature_dataset[triggers_df.columns.tolist()] = triggers_df

# COMMAND ----------

feature_dataset['count_failed_business_rules'] = feature_dataset['count_rc_rule_in_fbr'] + feature_dataset[triggers_df.columns.tolist()].sum(axis=1)
feature_dataset['count_failed_business_rules'].value_counts()

# COMMAND ----------

feature_dataset['count_failed_business_rules'] = np.clip(feature_dataset['count_failed_business_rules'], a_min=0, a_max=5)
feature_dataset['count_failed_business_rules'].value_counts()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['individual_blocklist'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['individual_identity_address'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['individual_sanctions_pep'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['individual_id_scan'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['business_internal_checks'])[['is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['count_failed_business_rules'])[['is_app_fraud', 'app_fraud_amount']].mean()


# COMMAND ----------

feature_dataset.drop(columns = ['manual_approval_triggers',
       'manual_approval_triggers_unexpected'], inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

feature_dataset['is_approved'].mean(), feature_dataset['is_approved'].sum()

# COMMAND ----------

feature_dataset['company_created_on'].min(), feature_dataset['company_created_on'].max()

# COMMAND ----------

feature_dataset['year_month'] = pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: str(x.date())[:7])

# COMMAND ----------

feature_dataset[~feature_dataset['year_month'].isin(['2021-10', '2021-11', '2021-12'])]['is_app_fraud'].mean()

# COMMAND ----------

feature_dataset[feature_dataset['is_approved']==1].groupby(['year_month'])['is_app_fraud'].mean()

# COMMAND ----------

pd.isnull(feature_dataset).sum()

# COMMAND ----------

import os
# os.mkdir(data_location)
feature_dataset.to_csv(data_location + f"app_fraud_feature_dataset_{start_date}_{end_date}" + ".csv")

# COMMAND ----------

