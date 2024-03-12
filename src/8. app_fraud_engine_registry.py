# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Model Defination and Registry
# MAGIC This notebook will be used to read all the model artfecats that were saved in earlier notebooks, wrap them in a model class definition and then register the mlflow model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

import pickle
from collections import OrderedDict
from io import IOBase
from typing import Dict
from ds_python_utils.frozendict import FrozenDict
import json

from kyc_decisioning_common.decision_engine.helpers import dict_to_dataframe, round_to_int
from kyc_decisioning_common.decision_engine.ver_2.base import RiskEngineT
from kyc_decisioning_common.decision_engine.ver_2.de_types import (
    EngineInputOutputT, DataFrameInputT, FeatureNames, PathOrFileLike, is_pathlike,
)
from kyc_decisioning_common.decision_engine.ver_2.enums import (
    LiveModelInfo,
    VerificationType,
    RiskCategory,
)
from kyc_decisioning_common.decision_engine.errors import RiskEngineError


# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

with open(artefact_location + "app_km_glmm_encoders.pkl", 'rb') as f:
  glmm_encoders = pickle.load(f)
glmm_encoders

# COMMAND ----------

with open(artefact_location + "app_km_optb_encoders.pkl", 'rb') as f:
  optb_encoders = pickle.load(f)
optb_encoders

# COMMAND ----------

with open(artefact_location + "cal_rm_appf_model.pkl", 'rb') as f:
  cal_xgb_model = pickle.load(f)
cal_xgb_model

# COMMAND ----------

model_params = json.dumps(cal_xgb_model.estimator.get_params(), default=np_encoder)
with open(artefact_location + "model_params.json", "w") as f:
    f.write(model_params)

# COMMAND ----------

with open(artefact_location + "appf_rm_shap_explainer.pkl", 'rb') as f:
  shap_explainer = pickle.load(f)
shap_explainer

# COMMAND ----------

with open(artefact_location + "appf_rm_thresholds.json", 'r') as f:
  thresholds_json = json.load(f)
thresholds_json

# COMMAND ----------

medium_high_threshold, low_medium_threshold = thresholds_json.get("Medium -> High").get("threshold"), thresholds_json.get("Low -> Medium").get("threshold")
print(medium_high_threshold, low_medium_threshold)


# COMMAND ----------



# COMMAND ----------

class NTTMP_ModelError(Exception):
    # user defined exceptions must inherit from the base Exception class
    # if you want to be able to catch them explicitly
    # https://stackoverflow.com/questions/31713054/cant-catch-mocked-exception-because-it-doesnt-inherit-baseexception
    pass


class BaseNTTMPModel:
    """

    """
    
    APPROVED_OUTCOME = 'BAU'
    MKYC_OUTCOME = 'EDD'

    def __init__(self,
                input_feature_names_raw,
                input_numerical_engine_features,
                decision_categorical_engine_features,
                decision_numerical_engine_features,
                input_categorical_engine_features,
                 **kwargs):
      
        self.kwargs = kwargs

        self.MODEL_VERSION = '1.0.0'
        
        self.input_feature_names_raw = input_feature_names_raw

        self.input_categorical_engine_features = input_categorical_engine_features
        self.input_numerical_engine_features = input_numerical_engine_features
        self.input_feature_names = tuple([*self.input_categorical_engine_features] + [*self.input_numerical_engine_features])
      
        self.decision_categorical_engine_features = decision_categorical_engine_features
        self.decision_numerical_engine_features = decision_numerical_engine_features
        self.decision_feature_names = tuple([*self.decision_categorical_engine_features] + [*self.decision_numerical_engine_features])

        self.glmm_encoders = {}
        self.optb_encoders = {}
    
        self.model_class = xgb.XGBClassifier
        self.caliberation_class = CalibratedClassifierCV
        self.shap_explainer = shap.TreeExplainer
        self.model_obj = None
        self.model_calibration_obj = None
        self.shap_explainer_obj = None

        self.THRESHOLD_LOW_MEDIUM = float('nan')
        self.THRESHOLD_MEDIUM_HIGH = float('nan')

        # store risk category gradients
        self.RISK_GRADIENT_LOW = float('nan')
        self.RISK_GRADIENT_HIGH = float('nan')

        self.MIN_RISK_SCORE = 0
        self.MAX_RISK_SCORE = 1000
        self.TM_THRESHOLD_ABSOLUTE = 200

        self.RISK_SCORE_MULTIPLIER = 1000
        self.TM_THRESHOLD = None
        
        self.verification_type = VerificationType.white_list
        self.STANDARDISED_MAX_LOW_RISK_SCORE = 399
        self.STANDARDISED_MAX_MED_RISK_SCORE = 601
        self.STANDARDISED_MAX_HIGH_RISK_SCORE = 999
        self.FAILSAFE_RISK_SCORE = 1000
        self.training_complete = False
    
    @classmethod
    def _kyc_string_outcome(cls, bool_outcome: bool) -> str:
        """Return 'approved' if bool(bool_outcome) is True, else return 'mkyc'."""
        return cls.APPROVED_OUTCOME if bool_outcome else cls.MKYC_OUTCOME
    
    @classmethod
    def np_encoder(cls, object):
        if isinstance(object, np.generic):
            return object.item()
    
    def validate_and_get_country(self, raw_input: str, required_len=2):

        try:
            if not isinstance(raw_input, str):
                return repr(TypeError("expected value to be a string with country code in it"))
            if raw_input is None:
                raw_input = "None"
            try:
                assert raw_input.strip()
            except AssertionError:
                return repr(TypeError("expected value to be a string with country code in it"))

        except Exception as ex:
            return repr(Exception(f"causing_value={raw_input}, ErrorReason={repr(ex)}"))

        alpha_count = len(raw_input)
        if alpha_count not in [2, 3]:
            return repr(ValueError(f"country_code={raw_input} is not alpha_2 or alpha_3"))
        alpha_scheme = {2: 'alpha_2', 3: 'alpha_3'}
        country = pycountry.countries.get(**{alpha_scheme[alpha_count]: raw_input})

        try:
            country_code = getattr(country, alpha_scheme[required_len]) if country else None
            assert country_code
        except AssertionError:
            return repr(ValueError(f"[{raw_input}] is not a valid country code"))

        return country_code
    
    def validate_and_get_postcode(self, raw_input: str):

        if raw_input is None or (isinstance(raw_input, str) and not raw_input.strip()):
            return repr(ValueError("empty value for the postcode is not allowed"))

        if not raw_input or not isinstance(raw_input, str):
            return repr(TypeError(f"postcode must be a string. Got {type(raw_input).__name__}"))

        raw_input = raw_input.strip()

        if len(raw_input) <= 4:
            outward = raw_input.strip()
        else:
            raw_input = raw_input.replace(' ', '')
            inward_len = 3
            outward, inward = raw_input[:-inward_len].strip(), raw_input[-inward_len:].strip()

        if not 2 <= len(outward) <= 4:
            return repr(ValueError(
                f'Postcode {raw_input} is not valid - outward is expected to be b/w 2-4 chars. got outward_code={outward}'))

        return outward
    
    def validate_and_get_company_sic(self, raw_input: list):

        if isinstance(raw_input, list):
            return raw_input
        elif isinstance(raw_input, str):
            return [code.strip() for code in raw_input.split(',') if code.strip()]
        elif isinstance(raw_input, float):
            if np.isnan(raw_input):
                return []
        elif raw_input is None:
            return []
        else:
            ex = TypeError(f"Unexpected type <'{type(raw_input).__name__}'> of "
                            f"sic_codes={raw_input!r}")
            return repr(Exception(f"causing_value={raw_input}, ErrorReason={repr(ex)}"))

    def transform_features(self, input_data: pd.DataFrame) -> pd.DataFrame:

        input_data['INDIVIDUAL_CHECKS_C_NOT_PASSED'] = input_data['INDIVIDUAL_CHECKS_C_NOT_PASSED'].apply(lambda x: json.loads(x).get("value", []))
        input_data['individual_identity_address'] = input_data['INDIVIDUAL_CHECKS_C_NOT_PASSED'].apply(lambda x: any(filter(lambda check_type: check_type in ['INDIVIDUAL_IDENTITY_ADDRESS'], x))*1)
        input_data['individual_identity_address']

        input_data['APPLICANT_ID_COUNTRY_ISSUE__RawData'] = input_data['APPLICANT_ID_COUNTRY_ISSUE__RawData'].apply(lambda x: json.loads(x).get("value", "Undefined")).fillna("Undefined")
        input_data['applicant_idcountry_issue'] = input_data['APPLICANT_ID_COUNTRY_ISSUE__RawData'].apply(lambda x: "Undefined" if x in ["none", "None", "null", None, np.NaN] or str(x).__contains__("Error") else x)
        input_data['applicant_idcountry_issue'] = input_data['applicant_idcountry_issue'].apply(lambda x: self.validate_and_get_country(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x)
        input_data['applicant_idcountry_issue']
        
        input_data['APPLICANT_NATIONALITY__RawData'] = input_data['APPLICANT_NATIONALITY__RawData'].apply(lambda x: list(filter(lambda y: y not in ["none", "None", "null", None, np.NaN], json.loads(x).get("value", ["Undefined"]))) if json.loads(x).get("value") is not None else ["Undefined"])
        input_data['applicant_nationality'] = input_data['APPLICANT_NATIONALITY__RawData'].apply(lambda x: ["Undefined"] if len(x) == 0 else x)
        input_data['applicant_nationality'] = input_data['applicant_nationality'].apply(lambda x: [self.validate_and_get_country(c) for c in x]).apply(lambda x: ["Undefined" if c.__contains__("Error") else c for c in x])
        
        input_data['COMPANY_SIC_CODES__RawData'] = input_data['COMPANY_SIC_CODES__RawData'].apply(lambda x: list(filter(lambda y: y not in ["none", "None", "null", None, np.NaN], json.loads(x).get("value", ["Undefined"]))))
        input_data['COMPANY_SIC_CODES__RawData'] = input_data['COMPANY_SIC_CODES__RawData'].apply(lambda x: self.validate_and_get_company_sic(x))
        input_data['company_sic'] = input_data['COMPANY_SIC_CODES__RawData'].apply(lambda x: ["Undefined"] if len(x) == 0 else x)

        input_data['COMPANY_INDUSTRY_CLASSIFICATION__RawData'] = input_data['COMPANY_INDUSTRY_CLASSIFICATION__RawData'].apply(lambda x: json.loads(x).get("value", "Undefined"))
        input_data['company_icc'] = input_data['COMPANY_INDUSTRY_CLASSIFICATION__RawData'].apply(lambda x: "Undefined" if x in ["none", "None", "null", None, np.NaN] else x)

        input_data['APPLICANT_POSTCODE__RawData'] = input_data['APPLICANT_POSTCODE__RawData'].apply(lambda x: json.loads(x).get("value", "Undefined"))
        input_data['applicant_postcode'] = input_data['APPLICANT_POSTCODE__RawData'].apply(lambda x: "Undefined" if x in ["none", "None", "null", None, np.NaN] else x)
        input_data['applicant_postcode'] = input_data['applicant_postcode'].apply(lambda x: self.validate_and_get_postcode(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x).apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x)))

        input_data['COMPANY_POSTCODE__RawData'] = input_data['COMPANY_POSTCODE__RawData'].apply(lambda x: json.loads(x).get("value", "Undefined"))
        input_data['company_postcode'] = input_data['COMPANY_POSTCODE__RawData'].apply(lambda x: "Undefined" if x in ["none", "None", "null", None, np.NaN] else x)
        input_data['company_postcode'] = input_data['company_postcode'].apply(lambda x: self.validate_and_get_postcode(x)).apply(lambda x: "Undefined" if str(x).__contains__("Error") else x).apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x)))
        
        input_data['APPLICANT_email_domain'.lower()] = input_data['APPLICANT_email_domain'].apply(lambda x: x.split("@")[-1].split("#")[-1].lower() if isinstance(x, str) else "other").apply(lambda x: x if x in ['gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com', 'icloud.com', 'hotmail.co.uk', 'yahoo.co.uk', 'live.co.uk'] else 'other')

        input_data['APPLICANT_email_domain__RawData'] = input_data['APPLICANT_email_domain__RawData'].apply(lambda x: json.loads(x).get("value", "Undefined") if json.loads(x).get("value") is not None else "Undefined")
        input_data['applicant_email_numeric'] = input_data['APPLICANT_email_domain__RawData'].apply(lambda x: any(re.findall(r'\d+', x)))*1

        # input_data['APPLICANT_device_type'.lower()] = input_data['APPLICANT_device_type'].apply(lambda x: x.lower() if isinstance(x, str) else "Undefined")

        input_data['APPLICANT_YEARS_TO_ID_EXPIRY'.lower()] = input_data['APPLICANT_YEARS_TO_ID_EXPIRY'].apply(lambda x: np.NaN if x in ["none", "None", "null", None, np.NaN] else x)

        input_data['AGE_AT_COMPLETION'.lower()] = input_data['AGE_AT_COMPLETION'].apply(lambda x: np.NaN if x in ["none", "None", "null", None, np.NaN] else x)

        input_data['COMPANY_AGE_AT_COMPLETION'.lower()] = input_data['COMPANY_AGE_AT_COMPLETION'].apply(lambda x: np.NaN if x in ["none", "None", "null", None, np.NaN] else x)

        input_data['is_restricted_keyword_present'] = input_data[['COMPANY_KEYWORDS_Prohibited', 'COMPANY_KEYWORDS_High']].max(axis=1)

        input_data['applicant_id_type'] = input_data[['APPLICANT_ID_TYPE_Passport',
                                                    'APPLICANT_ID_TYPE_Driving_Licence',
                                                    'APPLICANT_ID_TYPE_Other_ID',
                                                    'APPLICANT_ID_TYPE_National_ID',
                                                    'APPLICANT_ID_TYPE_Provisional_Licence',
                                                    'APPLICANT_ID_TYPE_Residence_Permit']].apply(
                lambda row: 'Passport' if row['APPLICANT_ID_TYPE_Passport'] == 1 
                else ('Driving_Licence' if row['APPLICANT_ID_TYPE_Driving_Licence'] == 1 
                        else ('Provisional_Licence' if row['APPLICANT_ID_TYPE_Provisional_Licence'] == 1 
                            else ('Residence_Permit' if row['APPLICANT_ID_TYPE_Residence_Permit'] == 1 
                                    else ('National_ID' if row['APPLICANT_ID_TYPE_National_ID'] == 1
                                        else ('Other_ID' if row['APPLICANT_ID_TYPE_Other_ID'] == 1
                                                else 'Undefined'
                                            )
                                        )
                                )
                            )
                    ), axis=1)

        input_data['company_structurelevelwise'] = input_data[['COMPANY_STRUCTURE_LEVELWISE_1',
                                                'COMPANY_STRUCTURE_LEVELWISE_2',
                                                'COMPANY_STRUCTURE_LEVELWISE_3+']].apply(
                lambda row: '1' if row['COMPANY_STRUCTURE_LEVELWISE_1'] == 1 
                else ('2' if row['COMPANY_STRUCTURE_LEVELWISE_2'] == 1 
                        else ('3+' if row['COMPANY_STRUCTURE_LEVELWISE_3+'] == 1 
                            else 'Undefined'
                            )
                ), axis=1)

        nationality_count = input_data['applicant_nationality'].apply(lambda x: len(x)).max()
        sic_count = input_data['company_sic'].apply(lambda x: len(x)).max()

        input_data = input_data.merge(input_data[['company_sic']].apply(lambda row: {f"company_sic_{i}": j for i, j in enumerate(row['company_sic'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

        input_data = input_data.merge(input_data[['applicant_nationality']].apply(lambda row: {f"applicant_nationality_{i}": j for i, j in enumerate(row['applicant_nationality'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

        for key, encoder in {key: self.glmm_encoders[key] for key in [*self.input_categorical_engine_features]}.items():
  
            if key not in ['applicant_nationality', 'company_sic']:

                input_data[f"{key}_encoded"] = encoder.transform(input_data[f"{key}"])

            elif key in ['applicant_nationality']:
                
                cols = [f'{key}_{i}' for i in range(nationality_count)]
                for i, col in enumerate(cols):

                    input_data[f'{col}_encoded'] = encoder.transform(input_data[col].rename(f"{key}"))

                input_data[f'{key}_encoded'] = input_data[[f'{col}_encoded' for col in cols]].max(axis=1)

            elif key in ['company_sic']:

                cols = [f'{key}_{i}' for i in range(sic_count)]
                for i, col in enumerate(cols):

                    input_data[f'{col}_encoded'] = encoder.transform(input_data[col].rename(f"{key}"))

                input_data[f"{key}_encoded"] = input_data[[f'{col}_encoded' for col in cols]].max(axis=1)

            else:

                pass
        
        input_data['company_nob'] = json.dumps(input_data[['company_icc', 'company_sic']].squeeze().to_dict())

        input_data['company_nob_encoded'] = input_data[['company_icc_encoded', 'company_sic_encoded']].max(axis=1)

        for key, value in {key: self.optb_encoders[key.replace("_encoded", "")] for key in 
         [*self.decision_categorical_engine_features] + ['age_at_completion', 'company_age_at_completion', 'applicant_years_to_id_expiry']}.items():
            
            if key.__contains__("encoded"):
                input_data[f"{key}"] = value.transform(input_data[f"{key}"], 
                                                     metric="woe", metric_missing='empirical')
            
            else:
                input_data[f"{key}_encoded"] = value.transform(input_data[f"{key}"], 
                                                     metric="woe", metric_missing='empirical')

        
        return input_data
    
    def shap_reason(self, feature_df: pd.DataFrame) -> pd.DataFrame:
      
      shap_values = self.shap_explainer_obj.shap_values(feature_df[[*self.decision_feature_names]])
      
      if isinstance(shap_values, np.ndarray):
        shap_values = pd.DataFrame(shap_values, index = feature_df.index, columns=self.decision_feature_names)

      else:
        shap_values = pd.DataFrame(shap_values.values, index = feature_df.index, columns=[*self.decision_feature_names])
      
      reason = shap_values.apply(lambda row: [*self.decision_feature_names][row.argmax()], axis=1)
      
      return reason
      
    
    def verify_input_features(self, input_features: pd.DataFrame) -> bool:
        """
        This method verifies if transform_feature method returns the features which matches with the
        decision_engine_features
        :param decision_features is the dataframe which returned by transform_feature method.
        """
        return set(self.input_feature_names_raw).issubset(set(input_features.columns))
    
    def verify_decision_features(self, decision_features: pd.DataFrame) -> bool:
        """
        This method verifies if transform_feature method returns the features which matches with the
        decision_engine_features
        :param decision_features is the dataframe which returned by transform_feature method.
        """
        return set(self.decision_feature_names).issubset(set(decision_features.columns))

    def predict_proba(self, feature_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        This function takes the KYC data and implements the risk engine V2
        model, and outputs both the integer score and the 'low, med, high' scores

        :param pd.DataFrame feature_df: dataframe of KYC data
        :return: pd.Series: dataframe of predicted probabilites
        """
            
        if not self.verify_decision_features(decision_features=feature_df):

            feature_df = self.transform_features(feature_df)
            self.feature_df = feature_df

            if not self.verify_decision_features(decision_features=feature_df):
                raise NTTMP_ModelError('Decision feature error in predict_proba')
        
        pred_proba_np = self.model_calibration_obj.predict_proba(feature_df[[*self.decision_feature_names]])

        labels_pred_ser = pd.Series(pred_proba_np[:, 1], index=feature_df.index)
        reason = self.shap_reason(feature_df)

        labels_pred_ser.rename('yPredScore')
        reason.rename('shap_reason')
        labels_pred_ser_scaled = labels_pred_ser.apply(lambda x: round(x * self.RISK_SCORE_MULTIPLIER))
        
#         return labels_pred_ser_scaled
        return (labels_pred_ser_scaled, reason)
        

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        This function performs the overall prediction to assess recall of the model
        (where the med/high threshold is the defacto decision boundary)

        :param pd.DataFrame feature_df: input features
        :return pd.Series: series containing the prediction result (0/1)
        """
        
#         prediction_df = self.predict_proba(feature_df)
        prediction_df = self.predict_proba(feature_df)[0]
  
        predict_df = prediction_df.apply(lambda x: 1 if x >= self.THRESHOLD_MEDIUM_HIGH else 0)

        return predict_df.rename('prediction')

    def predict_risk_category(self, feature_df):

        if len(self.input_feature_names) != 0:
            feature_df = feature_df[list(self.input_feature_names)]

        # output of model is sum of rows of DF (since it's a linear model)
        prediction_df = self.predict_proba(feature_df)[0]
#         prediction_df = self.predict_proba(feature_df)

        return prediction_df.apply(self._assign_risk_category)

    def _assign_risk_category(self, risk_rating) -> RiskCategory:
        """

        :param risk_rating:
        :return:
        """

        if risk_rating < self.MIN_RISK_SCORE:
            raise RuntimeError('risk score too low')
        elif risk_rating > self.MAX_RISK_SCORE:
            raise RuntimeError('risk score too high')
        elif risk_rating >= self.THRESHOLD_MEDIUM_HIGH:
            return RiskCategory.high.value
        elif risk_rating >= self.THRESHOLD_LOW_MEDIUM:
            return RiskCategory.medium.value
        elif risk_rating < self.THRESHOLD_LOW_MEDIUM:
            return RiskCategory.low.value
        else:
            raise RuntimeError('_assign_rejection_category error')

    def feature_names(self) -> FeatureNames:
        """Return a tuple of strings enumerating all feature names the model expects.
        Requires a trained model for this to work, otherwise raises
        ``XGBoostError('need to call fit or load_model first')`` .
        """
        return self.decision_feature_names

    def verify_input(self, features: EngineInputOutputT) -> None:
        """Pass silently if input data is valid, otherwise raise InvalidInputError.
                For sub-engines, currently does nothing."""


    def _assign_risk_category_standardised(self, risk_rating: int) -> RiskCategory:

        if risk_rating <= self.STANDARDISED_MAX_LOW_RISK_SCORE:
            return RiskCategory.low.value
        elif self.STANDARDISED_MAX_LOW_RISK_SCORE < risk_rating <= self.STANDARDISED_MAX_MED_RISK_SCORE:
            return RiskCategory.medium.value
        elif self.STANDARDISED_MAX_MED_RISK_SCORE < risk_rating <= self.STANDARDISED_MAX_HIGH_RISK_SCORE:
            return RiskCategory.high.value
        elif self.FAILSAFE_RISK_SCORE == risk_rating:
            return RiskCategory.high.value
        else:
            raise RuntimeError(f'{__name__}: UNREACHABLE CODE')
            

    def _standardise_risk_score(self, risk_score: int):
        """
        """
        if self._assign_risk_category(risk_score) == RiskCategory.low.value:
            normalised_risk_score = risk_score / self.THRESHOLD_LOW_MEDIUM
            new_risk_score_scale = self.STANDARDISED_MAX_LOW_RISK_SCORE - (self.TM_THRESHOLD_ABSOLUTE + 1) #plus one to be sure of being above threshold
            re_scaled_risk_score = new_risk_score_scale * normalised_risk_score
            return round(re_scaled_risk_score + (self.TM_THRESHOLD_ABSOLUTE + 1))
        # ensures risk score between "STANDARDISED_MAX_LOW_RISK_SCORE + 1" and "STANDARDISED_MAX_MED_RISK_SCORE" inclusive
        elif self._assign_risk_category(risk_score) == RiskCategory.medium.value:
            normalised_risk_score = (risk_score - self.THRESHOLD_LOW_MEDIUM) / ((self.THRESHOLD_MEDIUM_HIGH - 1) - self.THRESHOLD_LOW_MEDIUM)
            new_risk_score_scale = self.STANDARDISED_MAX_MED_RISK_SCORE - (self.STANDARDISED_MAX_LOW_RISK_SCORE + 1)
            re_scaled_risk_score = new_risk_score_scale * normalised_risk_score
            return round(re_scaled_risk_score + (self.STANDARDISED_MAX_LOW_RISK_SCORE + 1))
        # ensures risk score between "STANDARDISED_MAX_MED_RISK_SCORE + 1" and "STANDARDISED_MAX_HIGH_RISK_SCORE" inclusive
        elif self._assign_risk_category(risk_score) == RiskCategory.high.value:
            normalised_risk_score = (risk_score - self.THRESHOLD_MEDIUM_HIGH) / (self.RISK_SCORE_MULTIPLIER - self.THRESHOLD_MEDIUM_HIGH)
            new_risk_score_scale = self.STANDARDISED_MAX_HIGH_RISK_SCORE - (self.STANDARDISED_MAX_MED_RISK_SCORE + 1)
            re_scaled_risk_score = new_risk_score_scale * normalised_risk_score
            return round(re_scaled_risk_score + (self.STANDARDISED_MAX_MED_RISK_SCORE + 1))
        else:
            raise RuntimeError(f'{__name__}: UNREACHABLE CODE')

    def _decide(self, features_df: DataFrameInputT) -> Tuple[float, str]:
        """Call the ML model and then extract and return the risk score as float."""

        
#         predict_ndarray = self.predict_proba(features_df)
        predict_ndarray, predict_reason = self.predict_proba(features_df)
#         print('predict_ndarray',predict_ndarray)
         
#         predict_ser = pd.Series(predict_ndarray[:, 1], index=features_df.index)  # get the first column
        
#         return predict_ndarray.to_list()[0]
        return predict_ndarray.to_list()[0], predict_reason.tolist()[0]
        

    def _reduce_feature_vector(self, feature_vector: EngineInputOutputT) -> EngineInputOutputT:
        """Return an ordered dict of features following the order in this engine's
        booster feature_names list, eliminating any non-required features.
        """
        
        return OrderedDict((name, feature_vector[name]) for name in self.input_feature_names_raw)
        
    def decide(self, features: EngineInputOutputT, *args) -> EngineInputOutputT:

        ordered_vector = self._reduce_feature_vector(features)
        input_df = dict_to_dataframe(ordered_vector)
        self.input_df = input_df
  
        if not self.verify_input_features(input_features=input_df):
            raise NTTMP_ModelError('Error with input features')

#         risk_rating_raw = self._decide(input_df)
        risk_rating_raw, reason_feature = self._decide(input_df)

        # reason_value = input_df[reason_feature.replace("_encoded", "")][0]

        reason = reason_feature.replace("_encoded", "").upper()

        risk_rating_standardised = self._standardise_risk_score(risk_rating_raw)
        
        risk_engine_decision = self._kyc_string_outcome(risk_rating_standardised <= self.STANDARDISED_MAX_MED_RISK_SCORE)
        
        if self._assign_risk_category(risk_rating_raw) != self._assign_risk_category_standardised(risk_rating_standardised):
            raise APPFEngineError(f'rejection category difference between raw and standard scores')
            
        risk_category = self._assign_risk_category_standardised(risk_rating_standardised)

        return dict(
            raw_rating = round_to_int(risk_rating_raw),
            rating = round_to_int(risk_rating_standardised),
            category = risk_category,
            decision = risk_engine_decision,
            reason = [reason if risk_engine_decision=="EDD" else None]
        )




# COMMAND ----------

payload = {
      "INDIVIDUAL_CHECKS_C_NOT_PASSED": "{\"value\": [\"INDIVIDUAL_IDENTITY_ADDRESS\", \"INDIVIDUAL_SANCTIONS_PEP\", \"IDSCAN_MISMATCH\"]}",
      "APPLICANT_device_type": "android",
      "APPLICANT_ID_TYPE_National_ID": 1,

      "APPLICANT_ID_TYPE_Passport": 0,
      "APPLICANT_ID_TYPE_Driving_Licence": 0,
      "APPLICANT_ID_TYPE_Residence_Permit": 0,
      "APPLICANT_ID_TYPE_Other_ID": 0,
      "APPLICANT_ID_TYPE_Provisional_Licence": 0,
      "APPLICANT_ID_COUNTRY_ISSUE__RawData": "{\"value\": \"GB\"}",
      "AGE_AT_COMPLETION": 18,
      "APPLICANT_email_domain__RawData": "{\"value\": \"example_1@outlook.com\"}",
      "APPLICANT_email_domain": "@outlook.com",
      "COMPANY_KEYWORDS_High": 1,
      "COMPANY_KEYWORDS_Prohibited": 0,
      "COMPANY_SIC_CODES__RawData": "{\"value\": [\"01210\", \"01220\"]}",
      "COMPANY_INDUSTRY_CLASSIFICATION__RawData": "{\"value\": \"category.accident_investigator\"}",
      "APPLICANT_POSTCODE__RawData": "{\"value\": \"HP13\"}",
      "COMPANY_POSTCODE__RawData": "{\"value\": \"WC2H 9JQ\"}",
      "APPLICANT_YEARS_TO_ID_EXPIRY": 7,
      "COMPANY_STRUCTURE_LEVELWISE_1": 1,
      "COMPANY_STRUCTURE_LEVELWISE_2": 0,
      "COMPANY_STRUCTURE_LEVELWISE_3+": 0,
      "COMPANY_AGE_AT_COMPLETION": None,
      "APPLICANT_NATIONALITY__RawData": "{\"value\": [\"IN\", \"GBR\"]}"
  }
sample_input = json.dumps(payload)
sample_input = json.loads(sample_input)
sample_input

# COMMAND ----------


nttmp_engine = BaseNTTMPModel(
  input_feature_names_raw = input_feature_names_from_payload(),
  input_numerical_engine_features = get_input_numerical_features(),
  input_categorical_engine_features = get_input_categorical_features(),
  decision_numerical_engine_features = get_decision_numerical_features(),
  decision_categorical_engine_features = get_decision_categorical_features(),
)

nttmp_engine.training_complete = True
nttmp_engine.MODEL_VERSION = '1.0.0'
nttmp_engine.glmm_encoders = copy.deepcopy(glmm_encoders)
nttmp_engine.optb_encoders = copy.deepcopy(optb_encoders)
nttmp_engine.model_obj = copy.deepcopy(cal_xgb_model.base_estimator)
nttmp_engine.model_calibration_obj = copy.deepcopy(cal_xgb_model)
nttmp_engine.shap_explainer_obj = copy.deepcopy(shap_explainer)
nttmp_engine.THRESHOLD_MEDIUM_HIGH =  medium_high_threshold
nttmp_engine.THRESHOLD_LOW_MEDIUM = low_medium_threshold

# COMMAND ----------

nttmp_engine.input_feature_names

# COMMAND ----------

nttmp_engine.decision_feature_names

# COMMAND ----------

nttmp_engine.feature_names()

# COMMAND ----------

sample_output = nttmp_engine.decide(sample_input)
sample_output

# COMMAND ----------


json.loads(nttmp_engine.input_df.to_json(orient='records'))

# COMMAND ----------

json.loads(nttmp_engine.feature_df.to_json(orient='records'))

# COMMAND ----------

import mlflow
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self, wrapped_class):

      self.model = wrapped_class

  def fit(self, X, y, w):

      pass

  def predict(self, context, model_input):

      return self.model.decide(model_input)

# COMMAND ----------

def open_model_params() -> {}:
  
   model_params = open(artefact_location + 'model_params.json', 'r')

   return json.load(model_params)

def log_eda_report():

  mlflow.log_artifact(artefact_location + "appf_data_pofile.html", "Explorary Data Analysis")

def log_feature_correlation_plot():

  mlflow.log_artifact(artefact_location + "feature_correlation_df.html", "Feature Correlation")

  mlflow.log_artifact(artefact_location + "feature_correlation_plot.png", "Feature Correlation")

def log_plot_roc():

  mlflow.log_artifact(artefact_location + "rm_val_roc_auc.png", "ROC AUC")

  mlflow.log_artifact(artefact_location + "km_val_roc_auc.png", "ROC AUC")

def log_precision_recall():

  mlflow.log_artifact(artefact_location + "rm_precision_recall_curve.png", "Precision Recall")

  mlflow.log_artifact(artefact_location + "km_precision_recall_curve.png", "Precision Recall")

def log_calibaration_curve():

  mlflow.log_artifact(artefact_location + "rm_calibaration_curve.png", "Calibration Curve")

  mlflow.log_artifact(artefact_location + "km_calibaration_curve.png", "Calibration Curve")
  
def log_feature_importance_bar_plot() -> None:

  mlflow.log_artifact(artefact_location + "feature_importance_bar_plot.png", "Summary of the effects(SHAP) of features")
    
def log_feature_importance_beeswarm_plot() -> None:

  mlflow.log_artifact(artefact_location + "feature_importance_beasworm_plot.png", "Summary of the effects(SHAP) of features")
    
def log_partial_dependence_plot() -> None:

  mlflow.log_artifact(artefact_location + "feature_partial_dependence_plot.png", "Summary of the effects(SHAP) of features")

def log_waterfall_plot() -> None:

  mlflow.log_artifact(artefact_location + "feature_importance_waterfall_plot.png", "Summary of the effects(SHAP) of features")

def open_performance_metrics() -> {}:

  performance_metrics = open(artefact_location + 'performance_metrics.json', 'r')

  return json.load(performance_metrics)

def open_precision_recall_group_size() -> {}:

  precision_recall_group_size = open(artefact_location + 'precision_recall_group_size.json', 'r')

  return json.load(precision_recall_group_size)

def open_fold_metrics() -> {}:

  fold_metrics = open(artefact_location + '5fold_cv_metrics.json', 'r')

  return json.load(fold_metrics) 

# COMMAND ----------

artefact_location

# COMMAND ----------

nttmp_model_mlflow = MLFlowWrapper(nttmp_engine)
run_name="uk_NTT_MP_at_onboarding"

tags = {
  "THRESHOLD_LOW_MEDIUM": nttmp_engine.THRESHOLD_LOW_MEDIUM,
  "THRESHOLD_MEDIUM_HIGH": nttmp_engine.THRESHOLD_MEDIUM_HIGH,
  }

with mlflow.start_run(run_name=run_name) as run:
  
  mlflow.set_tags(tags)

  log_eda_report()
  log_feature_correlation_plot()
  log_plot_roc()
  log_precision_recall()
  log_feature_importance_bar_plot()
  log_feature_importance_beeswarm_plot()
  log_calibaration_curve()
  log_partial_dependence_plot()
  log_waterfall_plot()
                     
  mlflow.pyfunc.log_model(run_name, python_model = nttmp_model_mlflow, 
                          pip_requirements=[
                            "cffi==1.14.6",
                            "cloudpickle==2.2.0",
                            "colorama==0.4.6",
                            "defusedxml==0.7.1",
                            "googleapis-common-protos==1.61.0",
                            "ipython==7.32.0",
                            "kyc-decisioning-common==0.5.1",
                            "psutil==5.8.0",
                            "tornado==6.1",
                            "optbinning==0.18.0",
                            "shap==0.41.0",
                            "category-encoders==2.5.1.post0",
                            "scikit-learn==1.2.1",
                            "pandas==1.3.5",
                            "xgboost==1.3.3",
                            "pycountry"]
                          )
  mlflow_run_id = run.info.run_uuid
  artifactURI = run.info.artifact_uri

  mlflow.log_dict({"model_params": open_model_params()}, 
                  artifact_file='model_params.json')
  
  mlflow.log_dict({
    'input_feature_names': nttmp_engine.input_feature_names, 
    'decision_feature_names': nttmp_engine.decision_feature_names
  }, artifact_file='feature_names.json')
  
  mlflow.log_dict({
    'input_feature_vector': sample_input, 
    'nttmp_decision': sample_output
  }, artifact_file='sample_input_output.json')
  
  performance_metrics = open_performance_metrics()
  mlflow.log_metrics({"AUC_RM_train": performance_metrics.get("AUC").get("RM").get("train")})
  mlflow.log_metrics({"AUC_RM_test": performance_metrics.get("AUC").get("RM").get("test")})
  mlflow.log_metrics({"AUC_RM_val": performance_metrics.get("AUC").get("RM").get("val")})
  mlflow.log_metrics({"AUC_KM_train": performance_metrics.get("AUC").get("KM").get("train")})
  mlflow.log_metrics({"AUC_KM_test": performance_metrics.get("AUC").get("KM").get("test")})
  mlflow.log_metrics({"AUC_KM_val": performance_metrics.get("AUC").get("KM").get("val")})
  
  mlflow.log_dict({'precision_recall_group_size': open_precision_recall_group_size()}, 
                  artifact_file='precision_recall_group_size.json')
  
  mlflow.log_dict({'fold_metrics': open_fold_metrics()}, 
                  artifact_file='5fold_metrics.json')

  mlflow.log_dict({'thresholds': thresholds_json}, 
                  artifact_file='thresholds.json')

  print('mlflow_run_id', mlflow_run_id)
  print('artifactURI', artifactURI)

# COMMAND ----------

import mlflow
logged_model =f'runs:/{mlflow_run_id}/uk_NTT_MP_at_onboarding' 
mlflow_ntt_mp_engine = mlflow.pyfunc.load_model(logged_model)
mlflow_ntt_mp_engine

# COMMAND ----------

mlflow_ntt_mp_engine.predict(sample_input)

# COMMAND ----------

