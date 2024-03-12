# Databricks notebook source
def validate_and_get_country(raw_input: str, required_len=2):
  import pycountry

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

# COMMAND ----------

from feature_extractors.kyc.risk_extractors.mappings.postcodes import PostcodeMapping

def validate_and_get_postcode(raw_input: str):
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

def postcode_value_mapper(raw_input: str, processed_value: str, mapping: dict = PostcodeMapping.get_postcode_mapping_v2()) -> str:
    import json

    if processed_value.__contains__("Error") or processed_value.__contains__("Exception"):
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": "Undefined",
                                      "error": repr(processed_value)}})

    if len(processed_value) == 0:
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": "Low",
                                      "error": None}})

    if processed_value in mapping:
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": mapping[processed_value],
                                      "error": None}})
    else:
        return postcode_value_mapper(raw_input, processed_value[:-1], mapping)

# COMMAND ----------

def validate_and_get_company_sic(raw_input: list):
  import numpy as np

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

# COMMAND ----------

def keywords_mapping() -> dict:
    return {
        "adult": "Prohibited",
        "ammo": "High",
        "ammunition": "Prohibited",
        "beautician": "High",
        "beauty": "High",
        "build": "High",
        "builder": "High",
        "builders": "High",
        "building": "High",
        "casino": "Prohibited",
        "cbd": "Prohibited",
        "clean": "High",
        "cleaner": "High",
        "cleaners": "High",
        "cleaning": "High",
        "clothing": "High",
        "construct": "High",
        "construction": "High",
        "enterprise": "High",
        "fetish": "Prohibited",
        "fx": "Prohibited",
        "garment": "High",
        "gun": "High",
        "hair": "High",
        "investment": "High",
        "investments": "High",
        "loan": "High",
        "metal": "High",
        "mining": "High",
        "oil": "Medium",
        "paint": "High",
        "painter": "High",
        "painters": "High",
        "painting": "High",
        "payday": "High",
        "pharmaceutical": "High",
        "recycling": "Medium",
        "salon": "High",
        "scrap": "High",
        "services": "High",
        "sex": "Prohibited",
        "sole-trader": "High",
        "soletrader": "High",
        "textiles": "High",
        "trade": "High",
        "trader": "High",
        "traders": "High",
        "trading": "High",
        "transfer": "High",
        "travel": "High",
        "valet": "High",
        "elite": "High",
        "girls": "High",
        "bullion": "High",
        "sapphire": "High",
        "ruby": "High",
        "platinum": "High",
        "ore": "High",
        "ores": "High",
        "bike": "High",
        "motorbike": "High",
        "plant": "High",
        "vacation": "High",
        "flight": "High",
        "trips": "High",
        "rag": "High",
        "cosmetics": "High",
        "nails": "High",
        "nail": "High",
        "massage": "High",
        "xxx": "Prohibited",
        "x-rated": "Prohibited",
        "insurance": "High",
        "mortgage": "Prohibited",
        "broker": "Prohibited",
        "bond": "High",
        "lend": "High",
        "funding": "High",
        "gemstones": "High",
        "invest": "High",
        "second hand": "High",
        "fruit": "High",
        "vegetable": "Low",
        "agency": "High",
        "diamond": "High",
        "disposal": "Medium",
        "drugs": "Medium",
        "extraction": "Medium",
        "fashion": "Medium",
        "fly": "Medium",
        "send": "Medium",
        "sunbed": "Medium",
        "waste": "Prohibited",
        "gem": "High",
        "gems": "Medium",
        "gold": "High",
        "hemp": "Prohibited",
        "holding": "Prohibited",
        "holdings": "Prohibited",
        "silver": "High",
        "weapon": "Prohibited",
        "luxury": "High",
        "deluxe": "Medium",
        "antiquities": "Medium",
        "antiques": "Medium",
        "tours": "Medium",
        "discover": "Medium",
        "tourist": "Medium",
        "destination": "Medium",
        "airlines": "Prohibited",
        "airline": "Prohibited",
        "auction": "Prohibited",
        "automotive": "Prohibited",
        "autos": "Prohibited",
        "bet": "High",
        "betting": "Prohibited",
        "bitcoin": "Prohibited",
        "bomb": "Prohibited",
        "bureau": "Prohibited",
        "capital": "High",
        "car": "High",
        "cars": "High",
        "motors": "Medium",
        "vehicle": "High",
        "carwash": "Prohibited",
        "crypto": "Prohibited",
        "currency": "Prohibited",
        "escort": "Prohibited",
        "exchange": "Prohibited",
        "forex": "Prohibited",
        "gamble": "Prohibited",
        "money": "Prohibited",
        "trust": "Prohibited",
        "syria": "Prohibited",
        "iraq": "Prohibited",
        "cic": "Prohibited",
        "c.i.c": "Prohibited",
        "church": "Prohibited",
        "churches": "Prohibited",
        "charity": "Prohibited",
        "charities": "Prohibited",
        "foundation": "Prohibited",
        "voluntary": "Prohibited",
        "association": "Prohibited",
        "society": "Prohibited",
        "community": "Prohibited",
        "collective": "Prohibited",
        "charitable": "Prohibited",
        "religion": "Prohibited",
        "religious": "Prohibited",
        "christian": "Prohibited",
        "muslim": "Prohibited",
        "hindu": "Prohibited",
        "mission": "Prohibited",
        "mosque": "Prohibited",
        "mosques": "Prohibited",
        "islam": "Prohibited",
        "islamic": "Prohibited",
        "memorial house": "Prohibited",
        "culture and education centre": "Prohibited",
        "brotherhood": "Prohibited",
        "saint": "Prohibited",
        "cross": "Prohibited",
        "christ": "Prohibited",
        "quran": "Prohibited",
        "sunnah": "Prohibited",
        "c.i.o": "Prohibited",
        "welfare": "Prohibited",
        "ethic": "Prohibited",
        "ethical": "Prohibited",
        "aid": "Prohibited",
        "church council": "Prohibited",
        "parochial": "Prohibited",
        "ecclesiastical": "Prohibited",
        "parish": "Prohibited",
        "hallows": "Prohibited",
        "poverty": "Prohibited",
        "choir": "Prohibited",
        "chorus": "Prohibited",
        "poor": "Prohibited",
        "reverend": "Prohibited",
        "need": "Prohibited",
        "in need": "Prohibited",
        "apostolic": "Prohibited",
        "baptist": "Prohibited",
        "humanity": "Prohibited",
        "gospel": "Prohibited",
        "memorial": "Prohibited",
        "spiritual": "Prohibited",
        "holy": "Prohibited",
        "grace": "Prohibited",
        "godly": "Prohibited",
        "god": "Prohibited",
        "miracle": "Prohibited",
        "volunteer":"Prohibited",
        "st mary": "Prohibited",
        "synagogue": "Prohibited",
        "bible": "Prohibited",
        "christianity": "Prohibited",
        "judaism": "Prohibited",
        "hinduism": "Prohibited",
        "spirituality": "Prohibited",
        "paganism": "Prohibited",
        "sikhism": "Prohibited",
        "faith": "Prohibited",
        "protestantism": "Prohibited",
        "atheism": "Prohibited",
        "buddhism": "Prohibited",
        "piety": "Prohibited",
        "creed": "Prohibited",
        "agnostic": "Prohibited",
        "sacred": "Prohibited",
        "theology": "Prohibited",
        "evangelicalism": "Prohibited",
        "orthodox": "Prohibited",
        "catholicism": "Prohibited",
        "prayer": "Prohibited",
        "polytheism": "Prohibited",
        "doctrinal": "Prohibited",
        "devout": "Prohibited",
        "divine": "Prohibited",
        "priest": "Prohibited",
        "chapel": "Prohibited",
        "mass": "Prohibited",
        "denomination": "Prohibited",
        "shrine": "Prohibited",
        "cathedral": "Prohibited",
        "sect": "Prohibited",
        "worship": "Prohibited",
        "temple": "Prohibited",
        "abbey": "Prohibited",
        "basilica": "Prohibited",
        "bethel": "Prohibited",
        "chancel": "Prohibited",
        "tabernacle": "Prohibited",
        "cult": "Prohibited",
        "clan": "Prohibited",
        "following": "Prohibited",
        "communion": "Prohibited",
        "confession": "Prohibited",
        "dogma": "Prohibited",
        "tenet": "Prohibited",
        "masjid": "Prohibited",
        "flock": "Prohibited",
        "sacrifice": "Prohibited",
        "shelter": "Prohibited",
        "safe house": "Prohibited",
        "altar": "Prohibited",
        "grave": "Prohibited",
        "reliquary": "Prohibited",
        "relic": "Prohibited",
        "pantheon": "Prohibited",
        "sin": "Prohibited",
        "kirk": "Prohibited",
        "bishop": "Prohibited",
        "methodist": "Prohibited",
        "sanctify": "Prohibited",
        "monastery": "Prohibited",
        "jewish": "Prohibited",
        "gothic": "Prohibited",
        "jesus": "Prohibited",
        "blessed": "Prohibited",
        "unhollowed": "Prohibited",
        "motherhouse": "Prohibited",
        "impious": "Prohibited",
        "pagan": "Prohibited",
        "consecrated": "Prohibited",
        "empathy": "Prohibited",
        "concur": "Prohibited",
        "passion": "Prohibited",
        "sympathize": "Prohibited",
        "dark web": "Prohibited",
        "darkweb": "Prohibited",
        "dark-web": "Prohibited",
        "firearm": "Prohibited",
        "firearms": "Prohibited",
        "gambling": "Prohibited",
        "military": "Prohibited",
        "msb": "Prohibited",
        "politic": "Prohibited",
        "politics": "Prohibited",
        "tank": "Prohibited",
        "tanks": "Prohibited"
    }

import re
keywords = list(map(str.lower, [k for k, v in keywords_mapping().items() if v in ['High', 'Prohibited']]))

def get_keywords(x: str, names: list = keywords):

  filtered_list = [re.search(r"\b{}\b".format(k), x) for k in names]
  filtered_list = [item.string for item in filtered_list if item is not None]
  return (any(filtered_list), np.unique(filtered_list).tolist())

# COMMAND ----------

def validate_and_get_company_icc(raw_input: str):
  import math
  icc = raw_input

  if icc is None or (isinstance(icc, float) and math.isnan(icc)):
      return repr(None)

  if not isinstance(icc, str):
      return repr(TypeError(f"Expected type <str, NoneType or float('nan')>,"
                            f" got {icc!r} of type <{type(icc).__name__}>"))
  return icc.lower().strip()

# COMMAND ----------

def validate_and_get_company_type(raw_input: str, is_registered_company: bool):
  on_company_type_none_override = 'sole-trader'.lower().strip()

  if any([raw_input is None,
          str(raw_input).__contains__("null"),
          str(raw_input).__contains__("None")]):
      if not is_registered_company:
          return on_company_type_none_override
      else:
          return repr(ValueError("company.type=None & registeredCompany=True is invalid state!"))

  return raw_input.lower().strip()

# COMMAND ----------

import json
class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return super(NpEncoder, self).default(obj)

# COMMAND ----------

def np_encoder(object):
  if isinstance(object, np.generic):
      return object.item()