SET threads TO {THREAD_COUNT};
SET memory_limit = '{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

WITH sym_grp AS (
  SELECT 
        '{MODEL_LEVEL_ID}' AS context_keys
      , {LEFT_PREFIX}.predictor_name
      , {LEFT_PREFIX}.predictor_type
      , {LEFT_PREFIX}.symbolic_value AS bin_contents
      , AVG({LEFT_PREFIX}.shap_coeff) AS contribution
      , MIN({LEFT_PREFIX}.shap_coeff) AS contribution_0
      , MAX({LEFT_PREFIX}.shap_coeff) AS contribution_100
      , COUNT(*) AS frequency
  FROM {TABLE_N} AS {LEFT_PREFIX} 
  WHERE {WHERE_CONDITION}
  GROUP BY {LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.predictor_type, {LEFT_PREFIX}.symbolic_value
)
SELECT
  context_keys
, predictor_name
, predictor_type
, bin_contents
, ROW_NUMBER() OVER(PARTITION BY context_keys, predictor_name ORDER BY frequency DESC) AS bin_order
, contribution
, contribution_0
, contribution_100
, frequency
FROM sym_grp

