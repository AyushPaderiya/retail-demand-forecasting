-- ============================================================================
-- ETL Script: Load CSV Data into Database Tables
-- ============================================================================
-- Purpose: Bulk load all CSV files from the data generation into database
-- Usage: Run after creating schema tables, adjust file paths as needed
-- ============================================================================

-- NOTE: Adjust the file paths to match your environment
-- For PostgreSQL, use COPY command
-- For other databases, use their respective bulk load syntax

-- ============================================================================
-- 1. LOAD DIMENSION TABLES (Load these first due to FK constraints)
-- ============================================================================

-- Load Stores
COPY stores(
    store_id, store_name, region, store_format, size_sqft, 
    open_date, latitude, longitude, manager_count, employee_count
)
FROM '/path/to/artifacts/data/stores.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Load Products
COPY products(
    product_id, product_name, category, subcategory, brand,
    unit_price, unit_cost, is_perishable, shelf_life_days
)
FROM '/path/to/artifacts/data/products.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Load Calendar
COPY calendar(
    date, year, month, day, day_of_week, day_name, week_of_year,
    quarter, is_weekend, is_holiday, holiday_name, days_to_holiday,
    days_from_holiday, season, is_month_start, is_month_end,
    is_quarter_start, is_quarter_end
)
FROM '/path/to/artifacts/data/calendar.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Load Customers
COPY customers(
    customer_id, first_name, last_name, email, phone,
    registration_date, preferred_store_id, loyalty_tier, lifetime_value
)
FROM '/path/to/artifacts/data/customers.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- ============================================================================
-- 2. LOAD TRANSACTIONAL TABLES
-- ============================================================================

-- Load Weather Data
COPY weather(
    weather_id, date, store_id, temp_high_f, temp_low_f,
    precipitation_inches, humidity_pct, wind_speed_mph, conditions
)
FROM '/path/to/artifacts/data/weather.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Load Promotions
COPY promotions(
    promotion_id, promotion_name, start_date, end_date,
    discount_percent, promotion_type, store_id, product_id
)
FROM '/path/to/artifacts/data/promotions.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Load Sales Transactions (largest table - consider chunking for very large files)
COPY sales_transactions(
    transaction_id, date, store_id, product_id, customer_id,
    quantity, unit_price, discount_applied, discount_percent, total_amount
)
FROM '/path/to/artifacts/data/sales_transactions.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- ============================================================================
-- 3. POST-LOAD VALIDATION
-- ============================================================================

-- Verify row counts
SELECT 'stores' AS table_name, COUNT(*) AS row_count FROM stores
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'calendar', COUNT(*) FROM calendar
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'weather', COUNT(*) FROM weather
UNION ALL
SELECT 'promotions', COUNT(*) FROM promotions
UNION ALL
SELECT 'sales_transactions', COUNT(*) FROM sales_transactions;

-- Check for orphaned foreign keys
SELECT 'orphaned_store_ids' AS check_name, COUNT(*) AS issue_count
FROM sales_transactions s
WHERE NOT EXISTS (SELECT 1 FROM stores st WHERE st.store_id = s.store_id)

UNION ALL

SELECT 'orphaned_product_ids', COUNT(*)
FROM sales_transactions s
WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = s.product_id);
