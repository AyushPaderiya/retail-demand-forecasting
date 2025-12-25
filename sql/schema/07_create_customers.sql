-- Create customers table
-- Customer dimension table

DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    segment VARCHAR(50) NOT NULL,
    first_purchase_date DATE,
    lifetime_value DECIMAL(12, 2),
    purchase_frequency_monthly DECIMAL(5, 1),
    avg_basket_size DECIMAL(10, 2),
    age_group VARCHAR(20),
    preferred_channel VARCHAR(20),
    email_opt_in BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_customers_segment ON customers(segment);
CREATE INDEX idx_customers_first_purchase ON customers(first_purchase_date);
CREATE INDEX idx_customers_channel ON customers(preferred_channel);

COMMENT ON TABLE customers IS 'Customer dimension with segment and behavior attributes';
COMMENT ON COLUMN customers.segment IS 'Customer segment: Regular, Premium, VIP, New, Inactive';
COMMENT ON COLUMN customers.lifetime_value IS 'Total customer lifetime value in dollars';
