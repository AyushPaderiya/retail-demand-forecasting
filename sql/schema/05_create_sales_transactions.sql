-- Create sales_transactions table
-- Main fact table containing all sales transactions

DROP TABLE IF EXISTS sales_transactions CASCADE;

CREATE TABLE sales_transactions (
    transaction_id BIGINT PRIMARY KEY,
    date DATE NOT NULL REFERENCES calendar(date),
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    customer_id INTEGER,  -- Nullable for anonymous transactions
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    discount_applied BOOLEAN DEFAULT FALSE,
    discount_percent INTEGER DEFAULT 0,
    total_amount DECIMAL(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common query patterns
CREATE INDEX idx_sales_date ON sales_transactions(date);
CREATE INDEX idx_sales_store ON sales_transactions(store_id);
CREATE INDEX idx_sales_product ON sales_transactions(product_id);
CREATE INDEX idx_sales_customer ON sales_transactions(customer_id);

-- Composite indexes for forecasting queries
CREATE INDEX idx_sales_store_date ON sales_transactions(store_id, date);
CREATE INDEX idx_sales_product_date ON sales_transactions(product_id, date);
CREATE INDEX idx_sales_store_product_date ON sales_transactions(store_id, product_id, date);

-- Partition hint: For production, consider partitioning by date
-- CREATE TABLE sales_transactions (...) PARTITION BY RANGE (date);

COMMENT ON TABLE sales_transactions IS 'Main fact table with all sales transactions';
COMMENT ON COLUMN sales_transactions.transaction_id IS 'Unique identifier for each transaction';
COMMENT ON COLUMN sales_transactions.customer_id IS 'Customer ID (nullable for anonymous purchases)';
