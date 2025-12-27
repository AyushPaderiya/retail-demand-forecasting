-- ============================================================================
-- Monthly Sales Trends with Year-over-Year Growth
-- ============================================================================
-- Purpose: Analyze monthly revenue trends with MoM and YoY comparisons
-- Tableau Use: Line chart showing revenue trends, YoY growth KPI cards
-- ============================================================================

WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', s.date) AS sale_month,
        EXTRACT(YEAR FROM s.date) AS sale_year,
        EXTRACT(MONTH FROM s.date) AS sale_month_num,
        SUM(s.total_amount) AS total_revenue,
        SUM(s.quantity) AS total_units_sold,
        COUNT(DISTINCT s.transaction_id) AS transaction_count,
        COUNT(DISTINCT s.customer_id) AS unique_customers,
        AVG(s.total_amount) AS avg_transaction_value
    FROM sales_transactions s
    GROUP BY 
        DATE_TRUNC('month', s.date),
        EXTRACT(YEAR FROM s.date),
        EXTRACT(MONTH FROM s.date)
),

with_comparisons AS (
    SELECT 
        ms.*,
        -- Previous month metrics (MoM)
        LAG(total_revenue, 1) OVER (ORDER BY sale_month) AS prev_month_revenue,
        -- Same month last year (YoY)
        LAG(total_revenue, 12) OVER (ORDER BY sale_month) AS prev_year_revenue,
        -- Rolling 3-month average
        AVG(total_revenue) OVER (
            ORDER BY sale_month 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m_avg_revenue
    FROM monthly_sales ms
)

SELECT 
    sale_month,
    sale_year,
    sale_month_num,
    total_revenue,
    total_units_sold,
    transaction_count,
    unique_customers,
    ROUND(avg_transaction_value, 2) AS avg_transaction_value,
    rolling_3m_avg_revenue,
    
    -- Month-over-Month Growth
    ROUND(
        CASE 
            WHEN prev_month_revenue > 0 
            THEN ((total_revenue - prev_month_revenue) / prev_month_revenue) * 100
            ELSE NULL 
        END, 2
    ) AS mom_growth_pct,
    
    -- Year-over-Year Growth
    ROUND(
        CASE 
            WHEN prev_year_revenue > 0 
            THEN ((total_revenue - prev_year_revenue) / prev_year_revenue) * 100
            ELSE NULL 
        END, 2
    ) AS yoy_growth_pct

FROM with_comparisons
ORDER BY sale_month;
