-- Create products table
-- Products dimension table containing product catalog

DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100) NOT NULL,
    brand VARCHAR(100),
    unit_price DECIMAL(10, 2) NOT NULL,
    unit_cost DECIMAL(10, 2),
    weight_kg DECIMAL(8, 2),
    is_perishable BOOLEAN DEFAULT FALSE,
    shelf_life_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common query patterns
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_subcategory ON products(subcategory);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_perishable ON products(is_perishable);

COMMENT ON TABLE products IS 'Product catalog dimension table';
COMMENT ON COLUMN products.product_id IS 'Unique identifier for each product';
COMMENT ON COLUMN products.is_perishable IS 'Whether product is perishable (affects demand patterns)';
