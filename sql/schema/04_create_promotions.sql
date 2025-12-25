-- Create promotions table
-- Promotions fact table

DROP TABLE IF EXISTS promotions CASCADE;

CREATE TABLE promotions (
    promotion_id BIGINT PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    promotion_type VARCHAR(50) NOT NULL,
    discount_percent INTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_featured BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common query patterns
CREATE INDEX idx_promotions_store ON promotions(store_id);
CREATE INDEX idx_promotions_product ON promotions(product_id);
CREATE INDEX idx_promotions_dates ON promotions(start_date, end_date);
CREATE INDEX idx_promotions_type ON promotions(promotion_type);

-- Composite index for lookups during sales generation
CREATE INDEX idx_promotions_lookup ON promotions(store_id, product_id, start_date, end_date);

COMMENT ON TABLE promotions IS 'Promotions and discounts applied to store-product combinations';
COMMENT ON COLUMN promotions.discount_percent IS 'Discount percentage (5-50%)';
