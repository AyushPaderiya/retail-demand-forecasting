-- ============================================================================
-- ETL Script: Create Summary Tables for Tableau Dashboards
-- ============================================================================
-- Purpose: Pre-aggregate data into summary tables for fast Tableau queries
-- Schedule: Run daily/weekly to refresh materialized views
-- ============================================================================

-- ============================================================================
-- DAILY SALES SUMMARY
-- ============================================================================
-- Aggregates transactions to daily level by store and product

DROP TABLE IF EXISTS summary_daily_sales;

CREATE TABLE summary_daily_sales AS
SELECT 
    s.date,
    s.store_id,
    s.product_id,
    st.region,
    st.store_format,
    p.category,
    p.subcategory,
    
    -- Sales metrics
    SUM(s.quantity) AS total_quantity,
    SUM(s.total_amount) AS total_revenue,
    COUNT(DISTINCT s.transaction_id) AS transaction_count,
    COUNT(DISTINCT s.customer_id) AS unique_customers,
    
    -- Averages
    AVG(s.total_amount) AS avg_transaction_value,
    AVG(s.quantity) AS avg_basket_size,
    
    -- Discount metrics
    SUM(CASE WHEN s.discount_applied THEN 1 ELSE 0 END) AS discounted_transactions,
    AVG(CASE WHEN s.discount_applied THEN s.discount_percent ELSE 0 END) AS avg_discount_pct

FROM sales_transactions s
JOIN stores st ON s.store_id = st.store_id
JOIN products p ON s.product_id = p.product_id
GROUP BY 
    s.date, s.store_id, s.product_id,
    st.region, st.store_format,
    p.category, p.subcategory;

-- Create indexes for Tableau performance
CREATE INDEX idx_summary_daily_date ON summary_daily_sales(date);
CREATE INDEX idx_summary_daily_store ON summary_daily_sales(store_id);
CREATE INDEX idx_summary_daily_region ON summary_daily_sales(region);
CREATE INDEX idx_summary_daily_category ON summary_daily_sales(category);


-- ============================================================================
-- WEEKLY PERFORMANCE SUMMARY
-- ============================================================================

DROP TABLE IF EXISTS summary_weekly_performance;

CREATE TABLE summary_weekly_performance AS
SELECT 
    DATE_TRUNC('week', s.date) AS week_start,
    EXTRACT(YEAR FROM s.date) AS year,
    EXTRACT(WEEK FROM s.date) AS week_num,
    s.store_id,
    st.store_name,
    st.region,
    
    -- Revenue metrics
    SUM(s.total_amount) AS weekly_revenue,
    SUM(s.quantity) AS weekly_units,
    COUNT(DISTINCT s.transaction_id) AS weekly_transactions,
    COUNT(DISTINCT s.customer_id) AS weekly_customers,
    
    -- Averages
    AVG(s.total_amount) AS avg_transaction_value,
    
    -- Store efficiency
    SUM(s.total_amount) / st.size_sqft AS revenue_per_sqft

FROM sales_transactions s
JOIN stores st ON s.store_id = st.store_id
GROUP BY 
    DATE_TRUNC('week', s.date),
    EXTRACT(YEAR FROM s.date),
    EXTRACT(WEEK FROM s.date),
    s.store_id, st.store_name, st.region, st.size_sqft;

CREATE INDEX idx_summary_weekly_week ON summary_weekly_performance(week_start);
CREATE INDEX idx_summary_weekly_store ON summary_weekly_performance(store_id);


-- ============================================================================
-- MONTHLY KPI SUMMARY
-- ============================================================================

DROP TABLE IF EXISTS summary_monthly_kpis;

CREATE TABLE summary_monthly_kpis AS
WITH monthly_data AS (
    SELECT 
        DATE_TRUNC('month', date) AS month,
        SUM(total_amount) AS revenue,
        SUM(quantity) AS units,
        COUNT(DISTINCT transaction_id) AS transactions,
        COUNT(DISTINCT customer_id) AS customers
    FROM sales_transactions
    GROUP BY DATE_TRUNC('month', date)
)
SELECT 
    month,
    revenue,
    units,
    transactions,
    customers,
    
    -- Month-over-month changes
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY month)) 
        / NULLIF(LAG(revenue, 1) OVER (ORDER BY month), 0) * 100, 
        2
    ) AS mom_growth_pct,
    
    -- Year-over-year changes
    LAG(revenue, 12) OVER (ORDER BY month) AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue, 12) OVER (ORDER BY month)) 
        / NULLIF(LAG(revenue, 12) OVER (ORDER BY month), 0) * 100, 
        2
    ) AS yoy_growth_pct,
    
    -- Running totals (YTD)
    SUM(revenue) OVER (
        PARTITION BY EXTRACT(YEAR FROM month) 
        ORDER BY month
    ) AS ytd_revenue

FROM monthly_data
ORDER BY month;

CREATE INDEX idx_summary_monthly_month ON summary_monthly_kpis(month);


-- ============================================================================
-- CUSTOMER SEGMENTS SUMMARY
-- ============================================================================

DROP TABLE IF EXISTS summary_customer_segments;

CREATE TABLE summary_customer_segments AS
WITH customer_metrics AS (
    SELECT 
        customer_id,
        MIN(date) AS first_purchase,
        MAX(date) AS last_purchase,
        COUNT(DISTINCT transaction_id) AS total_orders,
        SUM(total_amount) AS lifetime_revenue,
        AVG(total_amount) AS avg_order_value
    FROM sales_transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
)
SELECT 
    customer_id,
    first_purchase,
    last_purchase,
    total_orders,
    lifetime_revenue,
    ROUND(avg_order_value, 2) AS avg_order_value,
    last_purchase - first_purchase AS customer_tenure_days,
    
    -- RFM Segmentation
    CASE 
        WHEN lifetime_revenue >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY lifetime_revenue) FROM customer_metrics) THEN 'High Value'
        WHEN lifetime_revenue >= (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lifetime_revenue) FROM customer_metrics) THEN 'Medium Value'
        ELSE 'Low Value'
    END AS value_segment,
    
    CASE 
        WHEN total_orders >= 10 THEN 'Loyal'
        WHEN total_orders >= 3 THEN 'Regular'
        ELSE 'New'
    END AS frequency_segment

FROM customer_metrics;

CREATE INDEX idx_summary_customer_segment ON summary_customer_segments(value_segment);


-- ============================================================================
-- REFRESH TIMESTAMP
-- ============================================================================

DROP TABLE IF EXISTS etl_refresh_log;

CREATE TABLE IF NOT EXISTS etl_refresh_log (
    refresh_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    refresh_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count BIGINT
);

-- Log this refresh
INSERT INTO etl_refresh_log (table_name, row_count)
VALUES 
    ('summary_daily_sales', (SELECT COUNT(*) FROM summary_daily_sales)),
    ('summary_weekly_performance', (SELECT COUNT(*) FROM summary_weekly_performance)),
    ('summary_monthly_kpis', (SELECT COUNT(*) FROM summary_monthly_kpis)),
    ('summary_customer_segments', (SELECT COUNT(*) FROM summary_customer_segments));

SELECT * FROM etl_refresh_log ORDER BY refresh_timestamp DESC LIMIT 10;
