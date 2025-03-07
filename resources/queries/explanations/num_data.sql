SET threads TO {THREAD_COUNT};
SET memory_limit = '{MEMORY_LIMIT}GB';
SET enable_progress_bar = {ENABLE_PROGRESS_BAR};

WITH
    quantiles AS (
       SELECT
        *
        ,NTILE(10) OVER (PARTITION BY (predictor_name, context_keys) ORDER BY numeric_value ASC) AS decile
        FROM {TABLE_N} AS {LEFT_PREFIX}
        WHERE {WHERE_CONDITION}
    ),
    grouped_data AS (
        SELECT
            {LEFT_PREFIX}.context_keys
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.predictor_type
            , {LEFT_PREFIX}.decile
            , AVG({LEFT_PREFIX}.shap_coeff) AS contribution
            , MIN({LEFT_PREFIX}.shap_coeff) AS contribution_0
            , MAX({LEFT_PREFIX}.shap_coeff) AS contribution_100
            , COUNT(*) AS frequency
            , MIN({LEFT_PREFIX}.numeric_value) AS minimum
            , MAX({LEFT_PREFIX}.numeric_value) AS maximum
        FROM quantiles AS {LEFT_PREFIX}
        GROUP BY {LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.predictor_type, {LEFT_PREFIX}.decile, {LEFT_PREFIX}.context_keys
    ),
    intervals AS (
        SELECT
            {LEFT_PREFIX}.context_keys
            , {LEFT_PREFIX}.predictor_name
            , {LEFT_PREFIX}.decile
            , LAG(maximum) OVER (PARTITION BY ({LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.context_keys) ORDER BY {LEFT_PREFIX}.decile) AS min_interval
            , LEAD(minimum) OVER (PARTITION BY ({LEFT_PREFIX}.predictor_name, {LEFT_PREFIX}.context_keys) ORDER BY {LEFT_PREFIX}.decile) AS max_interval
        FROM grouped_data as {LEFT_PREFIX}
    )
SELECT
    {LEFT_PREFIX}.context_keys
    , {LEFT_PREFIX}.predictor_name
    , {LEFT_PREFIX}.predictor_type
    , {LEFT_PREFIX}.decile AS bin_order
    , {LEFT_PREFIX}.contribution
    , {LEFT_PREFIX}.contribution_0
    , {LEFT_PREFIX}.contribution_100
    , {LEFT_PREFIX}.frequency
    , CASE 
        WHEN {RIGHT_PREFIX}.min_interval IS NULL
            THEN '<=' || CAST(CAST(({LEFT_PREFIX}.maximum + {RIGHT_PREFIX}.max_interval) / 2.0 AS DECIMAL) AS VARCHAR)
        WHEN {RIGHT_PREFIX}.max_interval IS NULL
            THEN '>' || CAST(CAST(({LEFT_PREFIX}.minimum + {RIGHT_PREFIX}.min_interval) / 2.0 AS DECIMAL) AS VARCHAR)
        ELSE '[' || CAST(CAST(({LEFT_PREFIX}.minimum + {RIGHT_PREFIX}.min_interval) / 2.0 AS DECIMAL) AS VARCHAR) || ':' || CAST(CAST(({LEFT_PREFIX}.maximum + {RIGHT_PREFIX}.max_interval) / 2.0 AS DECIMAL) AS VARCHAR) || ']'
    END AS bin_contents
FROM grouped_data AS {LEFT_PREFIX}
JOIN intervals AS {RIGHT_PREFIX}
ON {LEFT_PREFIX}.predictor_name={RIGHT_PREFIX}.predictor_name AND {LEFT_PREFIX}.decile={RIGHT_PREFIX}.decile AND {LEFT_PREFIX}.context_keys = {RIGHT_PREFIX}.context_keys