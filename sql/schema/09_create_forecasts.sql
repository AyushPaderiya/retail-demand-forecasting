-- Create forecasts table
-- Store model predictions

DROP TABLE IF EXISTS forecasts CASCADE;

CREATE TABLE forecasts (
    forecast_id BIGSERIAL PRIMARY KEY,
    forecast_date DATE NOT NULL,  -- Date when forecast was made
    target_date DATE NOT NULL,    -- Date being forecasted
    store_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    horizon_days INTEGER NOT NULL,  -- 7, 14, or 30
    
    -- Predictions
    predicted_quantity DECIMAL(12, 2),
    predicted_revenue DECIMAL(14, 2),
    
    -- Confidence intervals
    prediction_lower DECIMAL(12, 2),
    prediction_upper DECIMAL(12, 2),
    
    -- Model info
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Actuals (updated after target_date passes)
    actual_quantity INTEGER,
    actual_revenue DECIMAL(12, 2),
    
    -- Error metrics (calculated after actuals available)
    absolute_error DECIMAL(12, 2),
    percentage_error DECIMAL(8, 4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(forecast_date, target_date, store_id, product_id, model_name)
);

-- Create indexes
CREATE INDEX idx_forecasts_forecast_date ON forecasts(forecast_date);
CREATE INDEX idx_forecasts_target_date ON forecasts(target_date);
CREATE INDEX idx_forecasts_store ON forecasts(store_id);
CREATE INDEX idx_forecasts_product ON forecasts(product_id);
CREATE INDEX idx_forecasts_model ON forecasts(model_name);
CREATE INDEX idx_forecasts_horizon ON forecasts(horizon_days);

-- Composite index for API queries
CREATE INDEX idx_forecasts_lookup ON forecasts(store_id, product_id, horizon_days, forecast_date DESC);

COMMENT ON TABLE forecasts IS 'Model predictions with actuals for accuracy tracking';
COMMENT ON COLUMN forecasts.forecast_date IS 'Date when the forecast was generated';
COMMENT ON COLUMN forecasts.target_date IS 'Future date being forecasted';
COMMENT ON COLUMN forecasts.horizon_days IS 'Forecast horizon: 7, 14, or 30 days';
