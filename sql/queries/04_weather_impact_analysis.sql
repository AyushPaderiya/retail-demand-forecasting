-- ============================================================================
-- Weather Impact on Sales Analysis
-- ============================================================================
-- Purpose: Analyze correlation between weather conditions and sales performance
-- Tableau Use: Scatter plots, conditional formatting, weather impact cards
-- ============================================================================

WITH daily_store_sales AS (
    SELECT 
        s.date,
        s.store_id,
        SUM(s.total_amount) AS daily_revenue,
        SUM(s.quantity) AS daily_units,
        COUNT(DISTINCT s.transaction_id) AS daily_transactions,
        COUNT(DISTINCT s.customer_id) AS daily_customers
    FROM sales_transactions s
    GROUP BY s.date, s.store_id
),

weather_sales AS (
    SELECT 
        ds.date,
        ds.store_id,
        ds.daily_revenue,
        ds.daily_units,
        ds.daily_transactions,
        ds.daily_customers,
        w.temp_high_f,
        w.temp_low_f,
        (w.temp_high_f + w.temp_low_f) / 2 AS avg_temp_f,
        w.precipitation_inches,
        w.conditions
    FROM daily_store_sales ds
    JOIN weather w ON ds.date = w.date AND ds.store_id = w.store_id
),

-- Analysis by weather condition type
condition_analysis AS (
    SELECT 
        conditions AS weather_condition,
        COUNT(*) AS observation_count,
        ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
        ROUND(AVG(daily_units), 2) AS avg_daily_units,
        ROUND(AVG(daily_transactions), 2) AS avg_daily_transactions,
        ROUND(AVG(daily_customers), 2) AS avg_daily_customers,
        ROUND(STDDEV(daily_revenue), 2) AS revenue_std_dev
    FROM weather_sales
    GROUP BY conditions
),

-- Analysis by temperature bands
temperature_bands AS (
    SELECT 
        CASE 
            WHEN avg_temp_f < 32 THEN 'Freezing (<32°F)'
            WHEN avg_temp_f < 50 THEN 'Cold (32-50°F)'
            WHEN avg_temp_f < 70 THEN 'Mild (50-70°F)'
            WHEN avg_temp_f < 85 THEN 'Warm (70-85°F)'
            ELSE 'Hot (>85°F)'
        END AS temperature_band,
        COUNT(*) AS observation_count,
        ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
        ROUND(AVG(daily_units), 2) AS avg_daily_units,
        ROUND(AVG(daily_transactions), 2) AS avg_daily_transactions
    FROM weather_sales
    GROUP BY 
        CASE 
            WHEN avg_temp_f < 32 THEN 'Freezing (<32°F)'
            WHEN avg_temp_f < 50 THEN 'Cold (32-50°F)'
            WHEN avg_temp_f < 70 THEN 'Mild (50-70°F)'
            WHEN avg_temp_f < 85 THEN 'Warm (70-85°F)'
            ELSE 'Hot (>85°F)'
        END
),

-- Precipitation impact
precipitation_impact AS (
    SELECT 
        CASE 
            WHEN precipitation_inches = 0 THEN 'No Rain'
            WHEN precipitation_inches < 0.1 THEN 'Light Rain (<0.1")'
            WHEN precipitation_inches < 0.5 THEN 'Moderate Rain (0.1-0.5")'
            ELSE 'Heavy Rain (>0.5")'
        END AS precipitation_level,
        COUNT(*) AS observation_count,
        ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
        ROUND(AVG(daily_transactions), 2) AS avg_daily_transactions,
        ROUND(AVG(daily_customers), 2) AS avg_daily_customers
    FROM weather_sales
    GROUP BY 
        CASE 
            WHEN precipitation_inches = 0 THEN 'No Rain'
            WHEN precipitation_inches < 0.1 THEN 'Light Rain (<0.1")'
            WHEN precipitation_inches < 0.5 THEN 'Moderate Rain (0.1-0.5")'
            ELSE 'Heavy Rain (>0.5")'
        END
)

-- Combine all analyses (use UNION ALL for Tableau compatibility)
SELECT 
    'By Condition' AS analysis_type,
    weather_condition AS category,
    observation_count,
    avg_daily_revenue,
    avg_daily_units,
    avg_daily_transactions,
    avg_daily_customers,
    revenue_std_dev
FROM condition_analysis

UNION ALL

SELECT 
    'By Temperature' AS analysis_type,
    temperature_band AS category,
    observation_count,
    avg_daily_revenue,
    avg_daily_units,
    avg_daily_transactions,
    NULL AS avg_daily_customers,
    NULL AS revenue_std_dev
FROM temperature_bands

UNION ALL

SELECT 
    'By Precipitation' AS analysis_type,
    precipitation_level AS category,
    observation_count,
    avg_daily_revenue,
    NULL AS avg_daily_units,
    avg_daily_transactions,
    avg_daily_customers,
    NULL AS revenue_std_dev
FROM precipitation_impact

ORDER BY analysis_type, avg_daily_revenue DESC;
