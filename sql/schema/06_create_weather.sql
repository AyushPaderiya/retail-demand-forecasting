-- Create weather table
-- Weather data by store location and date

DROP TABLE IF EXISTS weather CASCADE;

CREATE TABLE weather (
    weather_id BIGINT PRIMARY KEY,
    date DATE NOT NULL REFERENCES calendar(date),
    store_id INTEGER NOT NULL REFERENCES stores(store_id),
    temp_high_f INTEGER,
    temp_low_f INTEGER,
    precipitation_inches DECIMAL(5, 2),
    humidity_percent INTEGER,
    wind_speed_mph DECIMAL(5, 1),
    conditions VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(date, store_id)
);

-- Create indexes
CREATE INDEX idx_weather_date ON weather(date);
CREATE INDEX idx_weather_store ON weather(store_id);
CREATE INDEX idx_weather_conditions ON weather(conditions);

-- Composite index for joining with sales
CREATE INDEX idx_weather_store_date ON weather(store_id, date);

COMMENT ON TABLE weather IS 'Daily weather data by store location';
COMMENT ON COLUMN weather.temp_high_f IS 'Daily high temperature in Fahrenheit';
COMMENT ON COLUMN weather.conditions IS 'Weather condition: Clear, Cloudy, Rain, Snow, etc.';
