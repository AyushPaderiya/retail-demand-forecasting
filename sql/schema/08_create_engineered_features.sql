-- Create engineered_features table
-- Pre-computed features for ML training

DROP TABLE IF EXISTS engineered_features CASCADE;

CREATE TABLE engineered_features (
    feature_id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    store_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    
    -- Target variable
    daily_quantity INTEGER,
    daily_revenue DECIMAL(12, 2),
    
    -- Lag features
    sales_lag_7 DECIMAL(12, 2),
    sales_lag_14 DECIMAL(12, 2),
    sales_lag_30 DECIMAL(12, 2),
    qty_lag_7 INTEGER,
    qty_lag_14 INTEGER,
    qty_lag_30 INTEGER,
    
    -- Rolling features
    rolling_mean_7 DECIMAL(12, 2),
    rolling_std_7 DECIMAL(12, 2),
    rolling_mean_14 DECIMAL(12, 2),
    rolling_mean_30 DECIMAL(12, 2),
    rolling_min_7 DECIMAL(12, 2),
    rolling_max_7 DECIMAL(12, 2),
    
    -- Date features
    day_of_week INTEGER,
    week_of_year INTEGER,
    month INTEGER,
    quarter INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    days_to_holiday INTEGER,
    season VARCHAR(20),
    
    -- Store features
    store_region VARCHAR(50),
    store_format VARCHAR(50),
    store_size_sqft INTEGER,
    
    -- Product features
    product_category VARCHAR(100),
    product_subcategory VARCHAR(100),
    product_price DECIMAL(10, 2),
    is_perishable BOOLEAN,
    
    -- Weather features
    temp_high_f INTEGER,
    temp_low_f INTEGER,
    precipitation_inches DECIMAL(5, 2),
    weather_conditions VARCHAR(50),
    
    -- Promotion features
    has_promotion BOOLEAN,
    promotion_discount INTEGER,
    
    -- Entity aggregates
    store_avg_daily_sales_30d DECIMAL(12, 2),
    product_avg_daily_sales_30d DECIMAL(12, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(date, store_id, product_id)
);

-- Create indexes for ML queries
CREATE INDEX idx_features_date ON engineered_features(date);
CREATE INDEX idx_features_store ON engineered_features(store_id);
CREATE INDEX idx_features_product ON engineered_features(product_id);
CREATE INDEX idx_features_lookup ON engineered_features(store_id, product_id, date);

COMMENT ON TABLE engineered_features IS 'Pre-computed features for ML model training';
