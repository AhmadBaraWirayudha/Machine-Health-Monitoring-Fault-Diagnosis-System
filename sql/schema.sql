-- =============================================================
-- Condition Monitoring Database Schema
-- Project: Machine Health Monitoring & Fault Diagnosis System
-- Engine : SQLite (compatible with PostgreSQL with minor edits)
-- =============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- -------------------------------------------------------------
-- Assets
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id        TEXT PRIMARY KEY,
    asset_name      TEXT NOT NULL,
    asset_type      TEXT,                   -- motor, pump, fan, compressor
    location        TEXT,
    manufacturer    TEXT,
    install_date    TEXT,                   -- ISO 8601: 2022-06-15
    rated_power_kW  REAL,
    rated_speed_rpm REAL,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Inspections
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT    NOT NULL REFERENCES assets(asset_id),
    inspect_date    TEXT    NOT NULL,
    inspector       TEXT,
    method          TEXT,                   -- vibration, temperature, oil, visual
    condition_found TEXT,                   -- brief field note
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Sensor Readings  (raw values per inspection)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id        INTEGER NOT NULL REFERENCES inspections(inspection_id),
    air_temp_K           REAL,
    process_temp_K       REAL,
    rotational_speed_rpm REAL,
    torque_Nm            REAL,
    tool_wear_min        REAL,
    power_W              REAL,
    temp_diff_K          REAL,
    current_A            REAL,
    vibration_velocity   REAL,             -- mm/s RMS (ISO 10816)
    recorded_at          TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Feature Records  (extracted diagnostic features)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_records (
    feature_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id    INTEGER NOT NULL REFERENCES inspections(inspection_id),
    -- Time domain
    rms              REAL,
    kurtosis         REAL,
    crest_factor     REAL,
    peak_to_peak     REAL,
    skewness         REAL,
    variance_val     REAL,
    -- Frequency domain
    fft_peak_freq    REAL,
    fft_peak_amp     REAL,
    band_energy_low  REAL,
    band_energy_mid  REAL,
    band_energy_high REAL,
    spectral_entropy REAL,
    created_at       TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Health Scores
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS health_scores (
    score_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT    NOT NULL REFERENCES assets(asset_id),
    inspection_id   INTEGER REFERENCES inspections(inspection_id),
    score_date      TEXT    NOT NULL,
    health_score    REAL    NOT NULL,       -- 0–100
    health_status   TEXT    NOT NULL,       -- Good / Warning / Degraded / Critical
    anomaly_score   REAL,                   -- 0–1, higher = more normal
    fault_type      TEXT,                   -- predicted fault class
    fault_confidence REAL,                  -- 0–1
    created_at      TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Maintenance Recommendations
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT    NOT NULL REFERENCES assets(asset_id),
    score_id    INTEGER REFERENCES health_scores(score_id),
    rec_date    TEXT    NOT NULL,
    priority    TEXT,                       -- High / Medium / Low
    action      TEXT    NOT NULL,
    due_date    TEXT,
    status      TEXT    DEFAULT 'Open',     -- Open / In Progress / Closed
    closed_date TEXT,
    closed_by   TEXT,
    notes       TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------
-- Indexes
-- -------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_hs_asset_date  ON health_scores(asset_id, score_date);
CREATE INDEX IF NOT EXISTS idx_insp_asset     ON inspections(asset_id, inspect_date);
CREATE INDEX IF NOT EXISTS idx_rec_status     ON recommendations(status, priority);
CREATE INDEX IF NOT EXISTS idx_sr_insp        ON sensor_readings(inspection_id);
CREATE INDEX IF NOT EXISTS idx_feat_insp      ON feature_records(inspection_id);

-- -------------------------------------------------------------
-- Useful views
-- -------------------------------------------------------------

-- Latest health score per asset
CREATE VIEW IF NOT EXISTS v_latest_health AS
SELECT
    a.asset_id,
    a.asset_name,
    a.asset_type,
    a.location,
    h.score_date,
    h.health_score,
    h.health_status,
    h.fault_type
FROM assets a
JOIN health_scores h ON a.asset_id = h.asset_id
WHERE h.score_date = (
    SELECT MAX(score_date) FROM health_scores WHERE asset_id = a.asset_id
);

-- Open recommendations with asset info
CREATE VIEW IF NOT EXISTS v_open_recommendations AS
SELECT
    r.rec_id,
    r.asset_id,
    a.asset_name,
    a.location,
    r.priority,
    r.action,
    r.due_date,
    r.rec_date
FROM recommendations r
JOIN assets a ON r.asset_id = a.asset_id
WHERE r.status = 'Open'
ORDER BY
    CASE r.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    r.due_date;

-- -------------------------------------------------------------
-- Sample seed data
-- -------------------------------------------------------------
INSERT OR IGNORE INTO assets (asset_id, asset_name, asset_type, location, rated_power_kW, rated_speed_rpm)
VALUES
    ('MTR-001', 'Pump Motor #1',    'motor',      'Plant A – Bay 1',    7.5, 1500),
    ('MTR-002', 'Cooling Fan #2',   'fan',         'Plant A – Bay 2',    5.5, 1450),
    ('PMP-001', 'Feed Pump #1',     'pump',        'Plant B – Utilities', 11.0, 2900),
    ('CMP-001', 'Air Compressor #1','compressor',  'Plant B – Utilities', 15.0, 1480);
