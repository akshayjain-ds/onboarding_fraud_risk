# Databricks notebook source
data_location = "/dbfs/dbfs/Shared/Decisioning/Strawberries/App_fraud/"
artefact_location = "/Workspace/Shared/Decisioning/Strawberries/uk_app_fraud_model/app_fraud_engine_training_v2/artefacts/"

# COMMAND ----------

start_date, end_date = '2022-01-01', '2023-07-01'
oot_start_date, oot_end_date = '2023-07-02', '2024-02-29'
start_date, end_date, oot_start_date, oot_end_date

# COMMAND ----------

def input_feature_names_from_payload():

  return (
    'INDIVIDUAL_CHECKS_C_NOT_PASSED',
    # 'APPLICANT_device_type',
    'APPLICANT_ID_TYPE_National_ID',
    'APPLICANT_ID_TYPE_Passport',
    'APPLICANT_ID_TYPE_Driving_Licence',
    'APPLICANT_ID_TYPE_Residence_Permit',
    'APPLICANT_ID_TYPE_Other_ID',
    'APPLICANT_ID_TYPE_Provisional_Licence',
    'APPLICANT_ID_COUNTRY_ISSUE__RawData',
    'APPLICANT_NATIONALITY__RawData',
    'AGE_AT_COMPLETION',
    'APPLICANT_email_domain',
    'COMPANY_KEYWORDS_High',
    'COMPANY_KEYWORDS_Prohibited',
    'COMPANY_SIC_CODES__RawData',
    'COMPANY_INDUSTRY_CLASSIFICATION__RawData',
    'APPLICANT_POSTCODE__RawData',
    'COMPANY_POSTCODE__RawData',
    'APPLICANT_YEARS_TO_ID_EXPIRY',
    'COMPANY_STRUCTURE_LEVELWISE_1',
    'COMPANY_STRUCTURE_LEVELWISE_2',
    'COMPANY_STRUCTURE_LEVELWISE_3+',
    'COMPANY_AGE_AT_COMPLETION',
    'APPLICANT_email_domain__RawData'
    )

def get_input_categorical_features() -> tuple :
  return ('applicant_postcode', 
          'applicant_idcountry_issue',
          'applicant_id_type',
          # 'company_type',
          'company_icc',
          'company_postcode',
          'applicant_email_domain',
          # 'applicant_device_type',
          'company_structurelevelwise',
          # 'company_status',

          'applicant_nationality',
          'company_sic'
          )


def get_decision_categorical_features() -> tuple :
  return ('applicant_postcode_encoded', 
          'applicant_idcountry_issue_encoded',
          'applicant_id_type_encoded',
          # 'company_type_encoded',
          'company_postcode_encoded',
          'applicant_email_domain_encoded',
          # 'applicant_device_type_encoded',
          'company_structurelevelwise_encoded',
          # 'company_status_encoded',
          'company_nob_encoded',

          'applicant_nationality_encoded',
          )

def get_input_numerical_features() -> tuple:
    """ This returns the list of input features with new mappings.
    """
    return ('age_at_completion', 
            'company_age_at_completion',
            'applicant_years_to_id_expiry',
            'is_restricted_keyword_present',
            'individual_identity_address', 
            'applicant_email_numeric'
            # 'directors_avg_age_at_completion',
            # 'fraud_fail', 
            # 'individual_id_scan', 
            # 'rc_rule_in_fbr',
            # 'individual_blocklist',
            # 'individual_sanctions_pep',
            # 'business_internal_checks',
            # 'count_failed_business_rules'
            )
    
def get_decision_numerical_features() -> tuple:
  """ This returns the list of input features with new mappings.
  """
  return ('age_at_completion_encoded', 
          'company_age_at_completion_encoded',
          'applicant_years_to_id_expiry_encoded',
          'is_restricted_keyword_present',
          'individual_identity_address', 
          'applicant_email_numeric'
          # 'directors_avg_age_at_completion',
          # 'fraud_fail', 
          # 'individual_id_scan', 
          # 'rc_rule_in_fbr',
          # 'individual_blocklist',
          # 'individual_sanctions_pep',
          # 'business_internal_checks',
          # 'count_failed_business_rules'
          )

def get_input_features() -> tuple:
    """ This returns the list of input features with new mappings.
    """
    return tuple([*get_input_categorical_features()] + [*get_input_numerical_features()])
  
def get_decision_features() -> tuple:
    """ This returns the list of input features with new mappings.
    """
    return tuple([*get_decision_categorical_features()] + [*get_decision_numerical_features()])
  
len([*input_feature_names_from_payload()]), len([*get_input_features()]), len([*get_decision_features()])

# COMMAND ----------

id1 = "company_id"
id2 = "membership_id"
km_indicator = "is_approved"
date_feature = "company_created_on"
target_b = "is_app_fraud"
target_c = "app_fraud_amount"
train_start_date, train_end_date = "2022-01-01", "2022-12-31"
test_start_date, test_end_date = "2023-01-01", "2023-03-31"
val_start_date, val_end_date = "2023-04-01", "2023-06-30"