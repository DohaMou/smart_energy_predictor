CREATE TABLE dim_location (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    region TEXT,
    latitude NUMERIC(8,5),
    longitude NUMERIC(8,5)
);

CREATE TABLE raw_enedis_30min (
    id BIGSERIAL PRIMARY KEY,
    payload JSONB,
    source_file TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE consumption_hourly (
    ts TIMESTAMP NOT NULL,
    location_id INT NOT NULL REFERENCES dim_location(id),
    profile TEXT NOT NULL,
    energy_kwh NUMERIC(14,3) NOT NULL,
    PRIMARY KEY (ts, location_id, profile)
);

CREATE TABLE weather_hourly (
    ts TIMESTAMP NOT NULL,
    location_id INT NOT NULL REFERENCES dim_location(id),
    temperature_c NUMERIC(5,2),
    humidity_pct NUMERIC(5,2),
    wind_speed_kmh NUMERIC(6,2),
    PRIMARY KEY (ts, location_id)
);

CREATE TABLE features_hourly (
    ts TIMESTAMP NOT NULL,
    location_id INT NOT NULL REFERENCES dim_location(id),
    energy_kwh NUMERIC(14,3),
    temperature_c NUMERIC(5,2),
    wind_speed_kmh NUMERIC(6,2),
    hour SMALLINT,
    day_of_week SMALLINT,
    month SMALLINT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    lag_24h NUMERIC(14,3),
    lag_168h NUMERIC(14,3),
    rolling_mean_24h NUMERIC(14,3),
    PRIMARY KEY (ts, location_id)
);

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    location_id INT NOT NULL REFERENCES dim_location(id),
    target_ts TIMESTAMP NOT NULL,
    predicted_kwh NUMERIC(14,3),
    lower_bound NUMERIC(14,3),
    upper_bound NUMERIC(14,3),
    model_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    location_id INT NOT NULL REFERENCES dim_location(id),
    ts TIMESTAMP NOT NULL,
    alert_type TEXT,
    severity TEXT,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO dim_location (name, region, latitude, longitude)
VALUES ('Calais', 'Hauts-de-France', 50.9513, 1.8587);
