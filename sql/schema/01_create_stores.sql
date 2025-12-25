-- Create stores table
-- Stores dimension table containing store attributes

DROP TABLE IF EXISTS stores CASCADE;

CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL,
    store_format VARCHAR(50) NOT NULL,
    size_sqft INTEGER,
    open_date DATE,
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    manager_count INTEGER,
    employee_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common query patterns
CREATE INDEX idx_stores_region ON stores(region);
CREATE INDEX idx_stores_format ON stores(store_format);
CREATE INDEX idx_stores_open_date ON stores(open_date);

COMMENT ON TABLE stores IS 'Store dimension table containing store attributes and location';
COMMENT ON COLUMN stores.store_id IS 'Unique identifier for each store';
COMMENT ON COLUMN stores.region IS 'Geographic region: North, South, East, West, Central';
COMMENT ON COLUMN stores.store_format IS 'Store format: Supermarket, Hypermarket, Express, Online';
