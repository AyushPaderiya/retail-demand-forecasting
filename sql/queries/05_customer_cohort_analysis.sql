-- ============================================================================
-- Customer Cohort Retention Analysis
-- ============================================================================
-- Purpose: Analyze customer retention patterns by acquisition cohort
-- Tableau Use: Cohort heatmaps, retention curves, LTV dashboards
-- ============================================================================

WITH customer_first_purchase AS (
    -- Identify each customer's first purchase (cohort assignment)
    SELECT 
        customer_id,
        MIN(date) AS first_purchase_date,
        DATE_TRUNC('month', MIN(date)) AS cohort_month
    FROM sales_transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),

customer_activity AS (
    -- Track each customer's activity by month
    SELECT 
        s.customer_id,
        DATE_TRUNC('month', s.date) AS activity_month,
        SUM(s.total_amount) AS monthly_spend,
        COUNT(DISTINCT s.transaction_id) AS monthly_transactions
    FROM sales_transactions s
    WHERE s.customer_id IS NOT NULL
    GROUP BY s.customer_id, DATE_TRUNC('month', s.date)
),

cohort_activity AS (
    -- Join cohort info with activity data
    SELECT 
        cfp.cohort_month,
        ca.activity_month,
        -- Calculate months since first purchase (cohort period)
        EXTRACT(YEAR FROM ca.activity_month - cfp.cohort_month) * 12 +
        EXTRACT(MONTH FROM ca.activity_month - cfp.cohort_month) AS cohort_period,
        cfp.customer_id,
        ca.monthly_spend,
        ca.monthly_transactions
    FROM customer_first_purchase cfp
    JOIN customer_activity ca ON cfp.customer_id = ca.customer_id
),

cohort_sizes AS (
    -- Count customers per cohort
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM customer_first_purchase
    GROUP BY cohort_month
),

retention_matrix AS (
    -- Build the retention matrix
    SELECT 
        ca.cohort_month,
        ca.cohort_period,
        cs.cohort_size AS original_cohort_size,
        COUNT(DISTINCT ca.customer_id) AS active_customers,
        SUM(ca.monthly_spend) AS cohort_revenue,
        SUM(ca.monthly_transactions) AS cohort_transactions,
        
        -- Retention rate
        ROUND(
            COUNT(DISTINCT ca.customer_id)::DECIMAL / cs.cohort_size * 100, 
            2
        ) AS retention_rate_pct,
        
        -- Average spend per active customer
        ROUND(
            SUM(ca.monthly_spend) / NULLIF(COUNT(DISTINCT ca.customer_id), 0), 
            2
        ) AS avg_spend_per_customer
        
    FROM cohort_activity ca
    JOIN cohort_sizes cs ON ca.cohort_month = cs.cohort_month
    GROUP BY ca.cohort_month, ca.cohort_period, cs.cohort_size
)

SELECT 
    cohort_month,
    cohort_period,
    original_cohort_size,
    active_customers,
    retention_rate_pct,
    cohort_revenue,
    cohort_transactions,
    avg_spend_per_customer,
    
    -- Cumulative revenue per customer (proxy for LTV)
    SUM(cohort_revenue) OVER (
        PARTITION BY cohort_month 
        ORDER BY cohort_period
    ) / original_cohort_size AS cumulative_rev_per_customer,
    
    -- Period-over-period retention change
    retention_rate_pct - LAG(retention_rate_pct, 1) OVER (
        PARTITION BY cohort_month 
        ORDER BY cohort_period
    ) AS retention_change_ppt

FROM retention_matrix
WHERE cohort_period <= 12  -- Limit to first 12 months for readability
ORDER BY cohort_month, cohort_period;
