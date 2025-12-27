-- ============================================================================
-- Store Performance Ranking with Same-Store Sales Growth
-- ============================================================================
-- Purpose: Rank stores by performance with regional comparisons
-- Tableau Use: Heatmaps, bar charts with rankings, regional drill-down
-- ============================================================================

WITH store_metrics AS (
    SELECT 
        st.store_id,
        st.store_name,
        st.region,
        st.store_format,
        st.size_sqft,
        st.open_date,
        SUM(s.total_amount) AS total_revenue,
        SUM(s.quantity) AS total_units_sold,
        COUNT(DISTINCT s.transaction_id) AS transaction_count,
        COUNT(DISTINCT s.customer_id) AS unique_customers,
        COUNT(DISTINCT s.date) AS active_days
    FROM stores st
    LEFT JOIN sales_transactions s ON st.store_id = s.store_id
    GROUP BY 
        st.store_id, st.store_name, st.region, 
        st.store_format, st.size_sqft, st.open_date
),

regional_benchmarks AS (
    SELECT 
        region,
        AVG(total_revenue) AS avg_regional_revenue,
        AVG(total_revenue / NULLIF(size_sqft, 0)) AS avg_revenue_per_sqft
    FROM store_metrics
    GROUP BY region
),

ranked_stores AS (
    SELECT 
        sm.*,
        rb.avg_regional_revenue,
        rb.avg_revenue_per_sqft AS regional_avg_rev_per_sqft,
        
        -- Revenue per square foot
        ROUND(sm.total_revenue / NULLIF(sm.size_sqft, 0), 2) AS revenue_per_sqft,
        
        -- Average daily revenue
        ROUND(sm.total_revenue / NULLIF(sm.active_days, 0), 2) AS avg_daily_revenue,
        
        -- Overall ranking
        RANK() OVER (ORDER BY sm.total_revenue DESC) AS overall_rank,
        DENSE_RANK() OVER (ORDER BY sm.total_revenue DESC) AS overall_dense_rank,
        
        -- Regional ranking
        RANK() OVER (PARTITION BY sm.region ORDER BY sm.total_revenue DESC) AS regional_rank,
        
        -- Format ranking
        RANK() OVER (PARTITION BY sm.store_format ORDER BY sm.total_revenue DESC) AS format_rank,
        
        -- Performance quartile
        NTILE(4) OVER (ORDER BY sm.total_revenue DESC) AS performance_quartile,
        
        -- Percentile rank
        ROUND(PERCENT_RANK() OVER (ORDER BY sm.total_revenue) * 100, 1) AS percentile_rank

    FROM store_metrics sm
    JOIN regional_benchmarks rb ON sm.region = rb.region
)

SELECT 
    store_id,
    store_name,
    region,
    store_format,
    size_sqft,
    open_date,
    total_revenue,
    total_units_sold,
    transaction_count,
    unique_customers,
    revenue_per_sqft,
    avg_daily_revenue,
    overall_rank,
    regional_rank,
    format_rank,
    performance_quartile,
    percentile_rank,
    
    -- Performance vs regional average
    ROUND(
        ((total_revenue - avg_regional_revenue) / NULLIF(avg_regional_revenue, 0)) * 100, 
        2
    ) AS pct_vs_regional_avg,
    
    -- Efficiency score (revenue per sqft vs regional average)
    ROUND(
        (revenue_per_sqft / NULLIF(regional_avg_rev_per_sqft, 0)) * 100,
        1
    ) AS efficiency_index

FROM ranked_stores
ORDER BY overall_rank;
