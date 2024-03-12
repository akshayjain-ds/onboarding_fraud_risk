# Databricks notebook source
# MAGIC %md ##APP Fraud Engine: Queries
# MAGIC This notebook will be called by other notebooks to run queries on snowflake and pull data

# COMMAND ----------

perpetrator_query = """ 
select 
    c.member_id::varchar as member_id,
    c.company_id::varchar as company_id, 
    c.created_at::timestamp as timestamp,
    d.risk_category,
    d.failed_business_rules,
    case when d.failed_business_rules in ('["ORE_RISK_RATING"]', '[]') then 'approved' else 'mkyc' end as rules_engine_decision,
    d.final_decision,
    m.approved_at_clean::timestamp as approved_at,
    case when m.approved_at_clean is not null then 1 else 0 end as is_approved,
    max(case when appf.days_to_fraud <= 180 then 1 else 0 end) as is_app_fraud,
    min(appf.days_to_fraud) as days_to_app_fraud,
    min(appf.days_to_reject) as days_fraud_to_reject,
    sum(case when appf.days_to_fraud <= 180 then appf.amount else 0 end) as app_fraud_amount,
    LISTAGG(distinct case when appf.days_to_fraud <= 180 then appf.app_fraud_type end, ',') AS app_fraud_type
from 
    (select 
        try_parse_json(data):identity:membership_id::varchar as member_id,
        try_parse_json(data):metadata:created_at::timestamp as created_at,
        replace(try_parse_json(data):decision:risk_category::varchar, '"') as risk_category,
        replace(try_parse_json(data):decision:final_decision::varchar, '"') as final_decision,
        try_parse_json(data):decision:risk_rating as risk_rating,
        try_parse_json(data):decision:failed_business_rules::varchar as failed_business_rules,
        rank() OVER (PARTITION BY member_id 
                     ORDER BY created_at::timestamp DESC) AS rnk
    from raw.kyc_decisioning.decisions 
    where try_parse_json(data):metadata:created_at::date between '{from_date}'::date - interval '3 months' and '{to_date}'::date
    ) d 
    inner join
    KYX_PROD.PRES_KYX.memberships m
    on d.member_id = m.member_id and d.rnk=1
    inner join 
    KYX_PROD.PRES_KYX.companies c 
    on m.member_id::varchar = c.member_id::varchar
    left join 
    (WITH jira AS 
    (
    SELECT DISTINCT
        ticket_key,
        a.ticket_id,
        a.company_id,
        issue_description,
        fraud_report_kustomer_link,
        CASE WHEN fraud_report_kustomer_link IS NOT NULL THEN 'Bank' ELSE reported_by END AS reported_by,
        REGEXP_SUBSTR(issue_description, 'https?://tide.kustomerapp.com.*', 1, 1, 'e') AS link1,
        REGEXP_SUBSTR(fraud_report_kustomer_link, 'https?://tide.kustomerapp.com.*', 1, 1, 'e') AS link_rp,
        SPLIT_PART(link1, '|https:', 1) AS link2,
        REGEXP_SUBSTR(link2,'https.*\/event\/([a-z0-9]+)' ,1, 1, 'e',1 ) AS kustomer_ticket_id_2,
        REGEXP_SUBSTR(fraud_report_kustomer_link,'https.*\/event\/([a-z0-9]+)' ,1, 1, 'e',1 ) AS kustomer_ticket_id_1,
        CASE WHEN fraud_type_new = '419 emails and letters' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Advance fee fraud' THEN 'Advance Fee'
            WHEN fraud_type_new = 'Cryptocurrency investment fraud' THEN 'Investment'
            WHEN fraud_type_new = 'HMRC scam' THEN 'Impersonation (Other)'
            WHEN fraud_type_new = 'Impersonation scam' THEN 'Impersonation (Other)'
            WHEN fraud_type_new = 'Investment scam' THEN 'Investment'
            WHEN fraud_type_new = 'Invoice scams' THEN 'Invoice and Mandate'
            WHEN fraud_type_new = 'Mail boxes and multiple post redirections' THEN 'Invoice and Mandate'
            WHEN fraud_type_new = 'Mule' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Not provided' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Other' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Purchase scam' THEN 'Purchase scam'
            WHEN fraud_type_new = 'Pyramid scheme fraud' THEN 'Investment'
            WHEN fraud_type_new = 'Romance scam' THEN 'Romance scam'
            WHEN fraud_type_new = 'Safe account scam' THEN 'Impersonation (Police/Bank)'
            WHEN fraud_type_new = 'Telephone banking scam' THEN 'Impersonation (Other)'
            END AS main_category,
        fraud_type_category,
        COALESCE(main_category,fraud_type_category) AS app_fraud_type,
        CASE WHEN app_fraud_type IN ('Unknown type/Other',
                                    'Advance Fee',
                                    'Investment',
                                    'Impersonation (Other)',
                                    'Invoice and Mandate',
                                    'Purchase scam',
                                    'Romance scam',
                                    'Impersonation (Police/Bank)'
                                    ) THEN 'Yes' ELSE 'No' END AS app_flag, --Yes No Flag
        ----changes here
        COALESCE(app_fraud,app_flag) AS is_app,                         
        returned_funds_transaction_ref,
        ticket_created_at
        
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS" a
    LEFT JOIN "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKET_CHANGES" b 
        ON b.change_field = 'Status' 
        AND b.new_value = 'Confirmed'
    WHERE YEAR(ticket_created_at) >= 2022
        AND project_key = 'RCM'
        AND is_subtask <>1
        AND jira_ticket_status NOT IN ('Duplicate','Duplicates')
        AND COALESCE(fraud_type_new,fraud_type_category) IS NOT NULL
    )
    , jira_22 AS 
    (
    SELECT 
        jira_fincrime.ticket_key,
        jira_fincrime.company_id,
        jira_fincrime.ticket_id,
        reported_by,
        date_of_refund,
        fraud_caught,
        CASE WHEN fraud_type_new = '419 emails and letters' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Advance fee fraud' THEN 'Advance Fee'
            WHEN fraud_type_new = 'Cryptocurrency investment fraud' THEN 'Investment'
            WHEN fraud_type_new = 'HMRC scam' THEN 'Impersonation (Other)'
            WHEN fraud_type_new = 'Impersonation scam' THEN 'Impersonation (Other)'
            WHEN fraud_type_new = 'Investment scam' THEN 'Investment'
            WHEN fraud_type_new = 'Invoice scams' THEN 'Invoice and Mandate'
            WHEN fraud_type_new = 'Mail boxes and multiple post redirections' THEN 'Invoice and Mandate'
            WHEN fraud_type_new = 'Mule' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Not provided' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Other' THEN 'Unknown type/Other'
            WHEN fraud_type_new = 'Purchase scam' THEN 'Purchase scam'
            WHEN fraud_type_new = 'Pyramid scheme fraud' THEN 'Investment'
            WHEN fraud_type_new = 'Romance scam' THEN 'Romance scam'
            WHEN fraud_type_new = 'Safe account scam' THEN 'Impersonation (Police/Bank)'
            WHEN fraud_type_new = 'Telephone banking scam' THEN 'Impersonation (Other)'
            ELSE fraud_type_new END AS app_fraud_type,
            CASE WHEN fraud_type_new  IN ('Cryptocurrency investment fraud', 
                                        'HMRC scam', 
                                        'Impersonation scam', 
                                        'Investment scam', 
                                        'Invoice scams', 
                                        'Other', 
                                        'Romance scam',
                                        'Purchase scam', 
                                        'Advance fee fraud',
                                        'Pyramid scheme fraud',
                                        'Other',
                                        '419 emails and letters',
                                        'Safe account scam') THEN 'Yes' ELSE 'No' END AS is_app, --Yes No Flag here for 1 or 0
        MIN(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) AS reported_date
    
    FROM member_support_prod.pres_member_support.jira_tickets  AS jira_fincrime
    LEFT JOIN member_support_prod.pres_member_support.jira_ticket_changes 
        ON jira_fincrime.ticket_id = jira_ticket_changes.ticket_id 
        AND jira_ticket_changes.change_field = 'Number of fraud reports'
    WHERE jira_fincrime.project_key = 'RCM' 
        AND jira_fincrime.number_of_fraud_report IS NOT NULL
        AND jira_fincrime.is_subtask <> 1 
        AND (DATE(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) 
            >= DATE('2022-01-01') )
        AND jira_fincrime.jira_ticket_status not in ('Duplicate', 'Duplicates')
    group by 1,2,3,4,5,6,7,8
    )

-- select * from jira;
    ,banks AS 
    (
    SELECT DISTINCT sortcode,
            UPPER(TRIM(bank_name)) AS bank_name
    FROM "PAYMENT_SERVICES_PROD"."PRES_PAYMENT_SERVICES"."PAYMENT"
    )


    , kustomer AS (
    SELECT a.* , 
        DATE(COALESCE(b.created_at, c.created_at, ticket_created_at)) AS reported_date
    FROM jira a 
    LEFT JOIN member_support_prod.pres_member_support.kustomer_conversations b 
        ON a.kustomer_ticket_id_1 = b.ticket_id
    LEFT JOIN member_support_prod.pres_member_support.kustomer_conversations c 
        ON a.kustomer_ticket_id_2 = c.ticket_id

    )


    , txns AS (
    SELECT 
            ticket_key,
            REGEXP_REPLACE(txn_ref_all, '[\\s,"\"]', '') as txn_ref_all
    FROM (
    SELECT ticket_key, TRIM(one_fraud_report_transaction_reference) AS txn_ref_all 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key,  TRIM(two_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, TRIM(three_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key,  TRIM(four_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key,  TRIM(five_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, TRIM(six_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, TRIM(seven_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, trim(eight_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, trim(nine_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"
    UNION 
    SELECT ticket_key, trim(ten_fraud_report_transaction_reference) 
    FROM "MEMBER_SUPPORT_PROD"."PRES_MEMBER_SUPPORT"."JIRA_TICKETS"

    )
    WHERE txn_ref_all IS NOT NULL

    )

    -- select * from txns;
    , tickets AS
    (
    SELECT DISTINCT
        a.company_id,
        a.parent_key,
        CASE WHEN a.parent_key IS  NULL THEN  a.ticket_key ELSE a.parent_key END AS ticket_key_1,
        a.ticket_created_at,
        b.txn_ref,
        dt_account_name,
        LEFT(dt_account_identification,6)  AS sort_code,
        b.transaction_at,
        b.amount
    FROM 
    (SELECT * FROM member_support_prod.pres_member_support.jira_tickets WHERE project_key = 'RCM') a 
    LEFT JOIN txns t 
        ON a.ticket_key = t.ticket_key
    LEFT JOIN member_support_prod.pres_member_support.jira_tickets sub 
        ON a.parent_key = sub.ticket_key
    JOIN "PAYMENT_SERVICES_PROD"."PRES_PAYMENT_SERVICES"."CLEARED_TRANSACTIONS" b 
        ON t.txn_ref_all = b.txn_ref
    JOIN "PAYMENT_SERVICES_PROD"."PRES_PAYMENT_SERVICES"."PAYMENTS_REALTIME_TRANSACTIONS" rt 
        ON b.txn_ref = rt.transaction_reference
        AND rt.status = 'CLEARED'
    WHERE transaction_type IN ('PaymentIn','FasterPaymentIn')
    AND local_instrument <> 'UK.OBIE.SWIFT'
    )

    -- select * from tickets;

    , data AS 
    (
    SELECT *
    FROM (

    SELECT  t.ticket_key_1,
            t.company_id,
            t.ticket_created_at,
            d.reported_by,
            t.txn_ref,
            t.amount,
            t.sort_code,
            t.transaction_at,
            d.app_fraud_type,
            d.is_app,
            DATE(d.reported_date) AS reported_date
    FROM tickets t
    JOIN  kustomer d 
        ON  t.ticket_key_1 = d.ticket_key 


    UNION 

    SELECT  t2.ticket_key_1,
            t2.company_id,
            t2.ticket_created_at,
            d2.reported_by,
            t2.txn_ref,
            t2.amount,
            t2.sort_code,
            t2.transaction_at,
            d2.app_fraud_type,       
            d2.is_app,
            DATE(d2.reported_date) AS reported_date
    FROM tickets t2
    JOIN jira_22 d2 
        ON  t2.ticket_key_1 = d2.ticket_key 

    )
    QUALIFY ROW_NUMBER() OVER(PARTITION BY txn_ref ORDER BY ticket_created_at DESC) = 1
    )

    -- select * from data;
    select distinct
            a.company_id::varchar as company_id,
            a.transaction_at,
            d.approved_at_clean,
            a.app_fraud_type,
            amount,
            datediff('days', d.approved_at_clean, a.transaction_at) as days_to_fraud,
            datediff('days', a.transaction_at, d.rejected_at_clean) as days_to_reject
    from data a
    left join banks bb on a.sort_code = bb.sortcode
    left join kyx_prod.pres_kyx.companies c on to_varchar(a.company_id) = c.company_id
    left join kyx_prod.pres_kyx.memberships d on c.member_id = d.member_id
    where is_app = 'Yes') appf 
    on c.company_id::varchar = appf.company_id::varchar
where 
  m.is_completed=1 
  and nvl(m.approved_at_clean::date, c.created_at::date) <= '{to_date}'::date
  and 
  ((c.created_at::date between '{from_date}'::date and '{to_date}'::date)
  or 
  ((m.approved_at_clean::date between '{from_date}'::date and '{to_date}'::date) 
  and (m.approved_at_clean::date between c.created_at::date and c.created_at::date + interval '3 months'))
  )
group by 1, 2, 3, 4, 5, 6, 7, 8, 9
"""

# COMMAND ----------

sar_query = """ 
SELECT 
    to_varchar(c.company_id) as company_id, 
    to_varchar(m.member_id) as member_id, 
    max(case when datediff('days', m.approved_at_clean, A.sar_created_date) <= 90 then 1 else 0 end) as sar_label
FROM 
    (SELECT  
        REGEXP_SUBSTR(TRIM(jira_tickets.ticket_summary),'[0-9]{{4,}}') AS company_id, 
        TO_DATE(DATE_TRUNC('DAY', MIN(jira_ticket_changes.change_at))) AS sar_created_date
    FROM    
        TIDE.PRES_JIRA.JIRA_TICKETS AS jira_tickets
        LEFT JOIN TIDE.PRES_JIRA.JIRA_TICKET_CHANGES AS jira_ticket_changes 
        ON jira_tickets.TICKET_ID = jira_ticket_changes.TICKET_ID
    WHERE   jira_tickets.PROJECT_KEY = 'RCM' AND TRIM(jira_tickets.issue_type) IN ('TM alert', 'Risk case') AND
          (jira_tickets.JIRA_TICKET_STATUS IS NULL OR jira_tickets.JIRA_TICKET_STATUS <> 'Duplicates') AND
          (NOT (jira_tickets.is_subtask = 1 ) OR (jira_tickets.is_subtask = 1 ) IS NULL) AND
          jira_ticket_changes.NEW_VALUE IN ('SAR', 'Tide Review', 'PPS Review', 'Submit to NCA', 'NCA Approval', 
                                            'NCA Refusal', 'Clear funds', 'Off-board','customer')
    GROUP BY 1
    ) A 
    JOIN 
    KYX_PROD.PRES_KYX.COMPANIES c 
    ON A.company_id = c.company_id 
    JOIN 
    KYX_PROD.PRES_KYX.MEMBERSHIPS m 
    ON c.member_id = m.member_id
WHERE 
    m.is_completed = 1 
group by 1, 2"""

# COMMAND ----------

decision_query = """
select distinct
    c.company_id,
    a.risk_category,
    a.final_decision,
    a.failed_business_rules
from (select 
        try_parse_json(data):identity:membership_id as member_id,
        try_parse_json(data):metadata:created_at::timestamp as created_at,
        replace(try_parse_json(data):metadata:versions:risk_engine_version::varchar, '"') as risk_engine_version, 
        replace(try_parse_json(data):decision:risk_category::varchar, '"') as risk_category,
        replace(try_parse_json(data):decision:final_decision::varchar, '"') as final_decision,
        try_parse_json(data):decision:risk_rating as risk_rating,
        try_parse_json(data):decision:failed_business_rules::varchar as failed_business_rules,
        try_parse_json(data):decision:risk_engine_features as features,
        rank() OVER (PARTITION BY member_id 
                     ORDER BY created_at::timestamp DESC) AS rnk
      from raw.kyc_decisioning.decisions 
      where try_parse_json(data):metadata:created_at::date between '{from_date}'::date and '{to_date}'::date
     ) a 
     left join 
     kyx_prod.pres_kyx.memberships m 
     on a.member_id = m.member_id 
     left join 
     kyx_prod.pres_kyx.companies c 
     on a.member_id = c.member_id
where 
    a.rnk = 1 
    and m.is_completed = 1
"""

# COMMAND ----------

member_query = """
with company as
(
select 
  c.company_id::varchar as company_id,
  m.member_id::varchar as member_id,
  c.created_at::timestamp as created_at,
  c.accounts_next_due_at as comnpany_accounts_next_due_at, 
  c.is_accounts_overdue as comnpany_accounts_overdue, 
  c.company_status,
  rank() OVER (PARTITION BY c.company_id ORDER BY c.created_at::timestamp DESC) AS rnk
from 
  KYX_PROD.PRES_KYX.memberships m 
  inner join
  KYX_PROD.PRES_KYX.companies c
  on c.member_id = m.member_id
where 
  m.is_completed=1 
  and nvl(m.approved_at_clean::date, c.created_at::date) <= '{to_date}'::date
  and 
  ((c.created_at::date between '{from_date}'::date and '{to_date}'::date)
  or 
  ((m.approved_at_clean::date between '{from_date}'::date and '{to_date}'::date) 
  and (m.approved_at_clean::date between c.created_at::date and c.created_at::date + interval '3 months'))
  )
),
async as
(select 
  member_id, id_type, id_subtype, id_country, id_first_name, id_last_name, id_expiry_at,
  rank() OVER (PARTITION BY member_id ORDER BY created_at DESC) AS rnk
from 
  KYX_PROD.CHNL_KYX.VERIFICATION_ID_SCAN
where 
  verification_type in ('ID_SCAN','ID_SCAN_VALIDATIONS') 
  and verification_usage IN ('REGISTRATION', 'ACCOUNT_RECOVERY') 
  ),
dob as 
(SELECT
    member_id,
    date_of_birth,
    rank() OVER (PARTITION BY member_id ORDER BY created_at::timestamp) AS rnk
FROM 
    KYX_PROD.PRES_KYX.users
)
select distinct
  c.company_id, 
  c.created_at,
  c.comnpany_accounts_next_due_at, 
  c.comnpany_accounts_overdue,
  async.id_expiry_at, 
  async.id_first_name,
  async.id_last_name,
  dob.date_of_birth,
  datediff('years', c.created_at::date, async.id_expiry_at::date) as applicant_years_to_id_expiry
from 
  company c 
  left join 
  async
  on c.member_id = async.member_id 
  and c.rnk = 1 and async.rnk = 1
  left join 
  dob 
  on c.member_id = dob.member_id and dob.rnk=1
"""

# COMMAND ----------

duedil_query = """select distinct * from 
(select distinct
  c.company_id::varchar as company_id, 
  c.member_id::varchar as member_id,
  c.created_at::timestamp as created_at,
  case when COMPANIES_HOUSE_NUMBER = to_varchar(try_parse_json(json_col):data:companyNumber) then 1 else 0 end as duedil_hit,
  COMPANIES_HOUSE_NUMBER,
  try_parse_json(json_col):metadata:created_at::timestamp as duedil_created_at,
  try_parse_json(json_col):data as duedil_payload,
  try_parse_json(json_col):data:address:postcode::varchar as company_postcode,
  try_parse_json(json_col):data:address:countryCode::varchar as company_countrycode,
  try_parse_json(json_col):data:charitableIdentityCount as charitableIdentityCount,
  try_parse_json(json_col):data:financialSummary as financialSummary,
  try_parse_json(json_col):data:incorporationDate as incorporationDate,
  try_parse_json(json_col):data:numberOfEmployees as numberOfEmployees,
  try_parse_json(json_col):data:recentStatementDate as recentStatementDate,
  try_parse_json(json_col):data:majorShareholders as majorShareholders,
  try_parse_json(json_col):data:directorsTree as directorsTree,
  try_parse_json(json_col):data:shareholderTree as shareholderTree,
  try_parse_json(json_col):data:personsOfSignificantControl as personsOfSignificantControl,
  try_parse_json(json_col):data:structureDepth as structureDepth,
  try_parse_json(json_col):data:structureLevelWise:"1" as structureLevelWise,
  try_parse_json(json_col):data:status::varchar as status,
  rank() OVER (PARTITION BY COMPANIES_HOUSE_NUMBER ORDER BY duedil_created_at) AS rnk
from 
  KYX_PROD.PRES_KYX.companies c
  inner join 
  KYX_PROD.PRES_KYX.memberships m
  on c.member_id = m.member_id and m.is_completed=1
  left join 
  (select json_col from TIDE.DUEDIL_INTEGRATION.uk_registered_companies
  union
  select data as json_col from RAW.KYC_DECISIONING.COMPANY_DETAILS) d
  on COMPANIES_HOUSE_NUMBER = to_varchar(try_parse_json(json_col):data:companyNumber)
where c.created_at::date between '{from_date}'::date and '{to_date}'::date
order by company_id, created_at)
where rnk = 1
QUALIFY ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY duedil_hit desc) = 1"""

# COMMAND ----------

name_match_query = """
select 
    company_id,
    payload,
    concat_ws(' ', lower(nvl(first_name, '')), lower(nvl(middle_name, '')), lower(nvl(last_name, ''))) as full_name,
    concat_ws(' ', lower(nvl(id_first_name, '')), lower(nvl(id_middle_name, '')), lower(nvl(id_last_name, ''))) as id_full_name
from (select 
      company_id::varchar as company_id, 
      try_parse_json(payload) as payload, 
      try_parse_json(payload):applicant:firstName::varchar as first_name,
      try_parse_json(payload):applicant:middleName::varchar as middle_name,
      try_parse_json(payload):applicant:lastName::varchar as last_name,
      try_parse_json(payload):applicant:idScan:firstName::varchar as id_first_name,
      try_parse_json(payload):applicant:idScan:middleName::varchar as id_middle_name,
      try_parse_json(payload):applicant:idScan:lastName::varchar as id_last_name,
      rank() OVER (PARTITION BY company_id ORDER BY "timestamp" DESC) AS rnk
from tide.event_service.event
where 
  event_type in ('application/vnd.tide.membership-completed.v1', 'application/vnd.tide.calculate-risk.v1')
  and "timestamp"::date between '{from_date}'::date and '{to_date}'::date
  )
where
  rnk = 1
"""


# COMMAND ----------

multi_business_query = """select 
    c.company_id,
    -- m.member_id,
    max(case when map.count_companies > 0 then 1 else 0 end) as is_existing_user
from 
    kyx_prod.pres_kyx.companies c
    inner join
    kyx_prod.pres_kyx.memberships m
    on c.member_id = m.member_id
    left join
    kyx_prod.pres_kyx.users u
    on c.member_id = u.member_id
    left join
    (SELECT distinct
        i.user_id, 
        count( distinct b.legacy_id)-1 as count_companies
    FROM 
        kyx_prod.individual_business_svc_hourly_public.relationship r
        inner join 
        kyx_prod.individual_business_svc_hourly_public.individual i
            on r.entity_id = i.id
        inner join 
        kyx_prod.individual_business_svc_hourly_public.business b
            on r.entity_dest_id = b.id
    where 
        i.created_on::date <= b.created_on::date
        and i.user_id is not null
    group by i.user_id
    ) map 
    on 
    u.user_id = map.user_id
where 
    m.is_completed=1
    and c.created_at::date between '{from_date}'::date and '{to_date}'::date
group by 1"""