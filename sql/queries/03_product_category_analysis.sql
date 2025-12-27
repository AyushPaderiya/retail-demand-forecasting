-- ============================================================================
-- Product Category Performance Analysis
-- ============================================================================
-- Purpose: Analyze category performance, contribution, and cross-sell opportunities
-- Tableau Use: Treemaps, pie charts, category comparison dashboards
-- ============================================================================

WITH category_sales AS (
    SELECT 
        p.category,
        p.subcategory,
        SUM(s.total_amount) AS category_revenue,
        SUM(s.quantity) AS category_units,
        COUNT(DISTINCT s.transaction_id) AS category_transactions,
        COUNT(DISTINCT p.product_id) AS product_count,
        AVG(s.total_amount) AS avg_transaction_value,
        AVG(p.unit_price) AS avg_unit_price
    FROM products p
    JOIN sales_transactions s ON p.product_id = s.product_id
    GROUP BY p.category, p.subcategory
),

total_metrics AS (
    SELECT 
        SUM(category_revenue) AS grand_total_revenue,
        SUM(category_units) AS grand_total_units,
        SUM(category_transactions) AS grand_total_transactions
    FROM category_sales
),

category_analysis AS (
    SELECT 
        cs.*,
        tm.grand_total_revenue,
        
        -- Revenue contribution percentage
        ROUND((cs.category_revenue / tm.grand_total_revenue) * 100, 2) AS revenue_contribution_pct,
        
        -- Units contribution percentage
        ROUND((cs.category_units::DECIMAL / tm.grand_total_units) * 100, 2) AS units_contribution_pct,
        
        -- Revenue per unit
        ROUND(cs.category_revenue / NULLIF(cs.category_units, 0), 2) AS revenue_per_unit,
        
        -- Average basket size
        ROUND(cs.category_units::DECIMAL / NULLIF(cs.category_transactions, 0), 2) AS avg_basket_size,
        
        -- Running total for Pareto analysis
        SUM(cs.category_revenue) OVER (
            ORDER BY cs.category_revenue DESC
        ) AS cumulative_revenue,
        
        -- Cumulative percentage
        ROUND(
            SUM(cs.category_revenue) OVER (ORDER BY cs.category_revenue DESC) 
            / tm.grand_total_revenue * 100, 
            2
        ) AS cumulative_pct,
        
        -- Category rank
        RANK() OVER (ORDER BY cs.category_revenue DESC) AS category_rank

    FROM category_sales cs
    CROSS JOIN total_metrics tm
)

SELECT 
    category,
    subcategory,
    category_revenue,
    category_units,
    category_transactions,
    product_count,
    ROUND(avg_transaction_value, 2) AS avg_transaction_value,
    ROUND(avg_unit_price, 2) AS avg_unit_price,
    revenue_per_unit,
    avg_basket_size,
    revenue_contribution_pct,
    units_contribution_pct,
    cumulative_revenue,
    cumulative_pct,
    category_rank,
    
    -- ABC Classification based on cumulative revenue
    CASE 
        WHEN cumulative_pct <= 80 THEN 'A - High Value'
        WHEN cumulative_pct <= 95 THEN 'B - Medium Value'
        ELSE 'C - Low Value'
    END AS abc_classification

FROM category_analysis
ORDER BY category_rank;
