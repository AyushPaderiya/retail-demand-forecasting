-- ============================================================================
-- ETL Script: Incremental Load for Sales Transactions
-- ============================================================================
-- Purpose: Load only NEW transactions since last ETL run (incremental)
-- Schedule: Run daily to keep data current without full reload
-- ============================================================================

-- ============================================================================
-- 1. CREATE TRACKING TABLE (run once during setup)
-- ============================================================================

CREATE TABLE IF NOT EXISTS etl_watermarks (
    table_name VARCHAR(100) PRIMARY KEY,
    last_loaded_date DATE,
    last_loaded_id BIGINT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initialize watermark for sales_transactions
INSERT INTO etl_watermarks (table_name, last_loaded_date, last_loaded_id)
VALUES ('sales_transactions', '1900-01-01', 0)
ON CONFLICT (table_name) DO NOTHING;


-- ============================================================================
-- 2. STAGING TABLE FOR NEW DATA
-- ============================================================================

DROP TABLE IF EXISTS staging_sales_transactions;

CREATE TABLE staging_sales_transactions (
    transaction_id BIGINT,
    date DATE,
    store_id INTEGER,
    product_id INTEGER,
    customer_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10, 2),
    discount_applied BOOLEAN,
    discount_percent INTEGER,
    total_amount DECIMAL(12, 2),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- 3. LOAD NEW DATA INTO STAGING
-- ============================================================================
-- Option 1: Load from CSV file (uncomment and adjust path)
-- COPY staging_sales_transactions (
--     transaction_id, date, store_id, product_id, customer_id,
--     quantity, unit_price, discount_applied, discount_percent, total_amount
-- )
-- FROM '/path/to/new_sales_data.csv'
-- WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Option 2: Load from external staging table (if using data warehouse pattern)
-- INSERT INTO staging_sales_transactions (
--     transaction_id, date, store_id, product_id, customer_id,
--     quantity, unit_price, discount_applied, discount_percent, total_amount
-- )
-- SELECT 
--     transaction_id, date, store_id, product_id, customer_id,
--     quantity, unit_price, discount_applied, discount_percent, total_amount
-- FROM external_staging_sales  -- External table or foreign data wrapper
-- WHERE transaction_id > (
--     SELECT COALESCE(last_loaded_id, 0) 
--     FROM etl_watermarks 
--     WHERE table_name = 'sales_transactions'
-- );

-- NOTE: Choose one of the above options and uncomment.
-- For this demo, we'll skip actual data loading (staging table will be empty)


-- ============================================================================
-- 4. DATA QUALITY CHECKS ON STAGING
-- ============================================================================

-- Check for nulls in required fields
DO $$
DECLARE
    null_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO null_count
    FROM staging_sales_transactions
    WHERE transaction_id IS NULL 
       OR date IS NULL 
       OR store_id IS NULL 
       OR product_id IS NULL;
    
    IF null_count > 0 THEN
        RAISE EXCEPTION 'Found % rows with NULL required fields', null_count;
    END IF;
END $$;

-- Check for invalid foreign keys
DO $$
DECLARE
    invalid_stores INTEGER;
    invalid_products INTEGER;
BEGIN
    SELECT COUNT(*) INTO invalid_stores
    FROM staging_sales_transactions s
    WHERE NOT EXISTS (SELECT 1 FROM stores st WHERE st.store_id = s.store_id);
    
    SELECT COUNT(*) INTO invalid_products
    FROM staging_sales_transactions s
    WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = s.product_id);
    
    IF invalid_stores > 0 OR invalid_products > 0 THEN
        RAISE EXCEPTION 'Invalid FKs: % stores, % products', invalid_stores, invalid_products;
    END IF;
END $$;

-- Check for duplicates
DO $$
DECLARE
    dup_count INTEGER;
BEGIN
    SELECT COUNT(*) - COUNT(DISTINCT transaction_id) INTO dup_count
    FROM staging_sales_transactions;
    
    IF dup_count > 0 THEN
        RAISE EXCEPTION 'Found % duplicate transaction IDs', dup_count;
    END IF;
END $$;


-- ============================================================================
-- 5. MERGE STAGING INTO TARGET (UPSERT)
-- ============================================================================

INSERT INTO sales_transactions (
    transaction_id, date, store_id, product_id, customer_id,
    quantity, unit_price, discount_applied, discount_percent, total_amount
)
SELECT 
    transaction_id, date, store_id, product_id, customer_id,
    quantity, unit_price, discount_applied, discount_percent, total_amount
FROM staging_sales_transactions
ON CONFLICT (transaction_id) DO UPDATE SET
    quantity = EXCLUDED.quantity,
    total_amount = EXCLUDED.total_amount,
    discount_percent = EXCLUDED.discount_percent;


-- ============================================================================
-- 6. UPDATE WATERMARK
-- ============================================================================

UPDATE etl_watermarks
SET 
    last_loaded_date = (SELECT MAX(date) FROM staging_sales_transactions),
    last_loaded_id = (SELECT MAX(transaction_id) FROM staging_sales_transactions),
    updated_at = CURRENT_TIMESTAMP
WHERE table_name = 'sales_transactions';


-- ============================================================================
-- 7. LOG ETL RUN
-- ============================================================================

INSERT INTO etl_refresh_log (table_name, row_count)
VALUES ('sales_transactions_incremental', (SELECT COUNT(*) FROM staging_sales_transactions));


-- ============================================================================
-- 8. CLEANUP
-- ============================================================================

TRUNCATE TABLE staging_sales_transactions;

-- Show summary
SELECT 
    'Incremental Load Complete' AS status,
    (SELECT last_loaded_date FROM etl_watermarks WHERE table_name = 'sales_transactions') AS latest_date,
    (SELECT row_count FROM etl_refresh_log WHERE table_name = 'sales_transactions_incremental' ORDER BY refresh_timestamp DESC LIMIT 1) AS rows_loaded;
