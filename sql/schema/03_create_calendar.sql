-- Create calendar table
-- Date dimension table with temporal features

DROP TABLE IF EXISTS calendar CASCADE;

CREATE TABLE calendar (
    date DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,  -- 0=Monday, 6=Sunday
    day_name VARCHAR(20) NOT NULL,
    week_of_year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    holiday_name VARCHAR(100),
    is_month_start BOOLEAN NOT NULL,
    is_month_end BOOLEAN NOT NULL,
    days_to_holiday INTEGER,
    season VARCHAR(20) NOT NULL
);

-- Create indexes for common query patterns
CREATE INDEX idx_calendar_year_month ON calendar(year, month);
CREATE INDEX idx_calendar_week ON calendar(year, week_of_year);
CREATE INDEX idx_calendar_is_holiday ON calendar(is_holiday);
CREATE INDEX idx_calendar_is_weekend ON calendar(is_weekend);
CREATE INDEX idx_calendar_season ON calendar(season);

COMMENT ON TABLE calendar IS 'Date dimension table with temporal features for forecasting';
COMMENT ON COLUMN calendar.day_of_week IS 'Day of week: 0=Monday through 6=Sunday';
COMMENT ON COLUMN calendar.days_to_holiday IS 'Days until next major holiday';
