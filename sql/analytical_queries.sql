-- =============================================================
-- Analytical SQL Queries — Condition Monitoring Database
-- Use these in Power BI, SQLiteOnline, DBeaver, or Python:
--   pd.read_sql_query(QUERY, conn)
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- A. FLEET OVERVIEW
-- ─────────────────────────────────────────────────────────────

-- A1. Current health status of every asset (latest score each)
SELECT
    a.asset_id,
    a.asset_name,
    a.asset_type,
    a.location,
    h.score_date        AS last_checked,
    h.health_score,
    h.health_status,
    h.fault_type,
    h.fault_confidence
FROM assets a
JOIN health_scores h ON a.asset_id = h.asset_id
WHERE h.score_date = (
    SELECT MAX(score_date)
    FROM   health_scores
    WHERE  asset_id = a.asset_id
)
ORDER BY h.health_score ASC;


-- A2. Fleet health KPIs (single-row summary for a dashboard card)
SELECT
    COUNT(DISTINCT asset_id)                                        AS total_assets,
    ROUND(AVG(health_score), 1)                                     AS avg_health_score,
    SUM(CASE WHEN health_status = 'Critical'  THEN 1 ELSE 0 END)   AS critical_count,
    SUM(CASE WHEN health_status = 'Degraded'  THEN 1 ELSE 0 END)   AS degraded_count,
    SUM(CASE WHEN health_status = 'Warning'   THEN 1 ELSE 0 END)   AS warning_count,
    SUM(CASE WHEN health_status = 'Good'      THEN 1 ELSE 0 END)   AS good_count
FROM (
    SELECT asset_id, health_score, health_status
    FROM   health_scores h1
    WHERE  score_date = (
        SELECT MAX(score_date) FROM health_scores WHERE asset_id = h1.asset_id
    )
);


-- A3. Count of assets per status (for pie chart)
SELECT
    health_status,
    COUNT(*)                                   AS asset_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM (
    SELECT asset_id, health_status
    FROM   health_scores h1
    WHERE  score_date = (
        SELECT MAX(score_date) FROM health_scores WHERE asset_id = h1.asset_id
    )
)
GROUP BY health_status
ORDER BY
    CASE health_status
        WHEN 'Critical'  THEN 1
        WHEN 'Degraded'  THEN 2
        WHEN 'Warning'   THEN 3
        WHEN 'Good'      THEN 4
    END;


-- ─────────────────────────────────────────────────────────────
-- B. INDIVIDUAL ASSET DETAIL
-- ─────────────────────────────────────────────────────────────

-- B1. Full health score history for one asset (trend chart)
--     Replace 'MTR-001' with the target asset_id
SELECT
    h.score_date,
    h.health_score,
    h.health_status,
    h.anomaly_score,
    h.fault_type,
    h.fault_confidence,
    a.asset_name,
    a.location
FROM health_scores h
JOIN assets a ON h.asset_id = a.asset_id
WHERE h.asset_id = 'MTR-001'
ORDER BY h.score_date;


-- B2. Sensor readings for all inspections of one asset
SELECT
    i.inspect_date,
    i.method,
    sr.rotational_speed_rpm,
    sr.torque_Nm,
    sr.air_temp_K,
    sr.process_temp_K,
    sr.tool_wear_min,
    sr.power_W,
    sr.temp_diff_K
FROM inspections i
JOIN sensor_readings sr ON i.inspection_id = sr.inspection_id
WHERE i.asset_id = 'MTR-001'
ORDER BY i.inspect_date;


-- B3. Diagnostic features over time for one asset
SELECT
    i.inspect_date,
    fr.rms,
    fr.kurtosis,
    fr.crest_factor,
    fr.peak_to_peak,
    fr.fft_peak_freq,
    fr.fft_peak_amp,
    fr.spectral_entropy
FROM inspections i
JOIN feature_records fr ON i.inspection_id = fr.inspection_id
WHERE i.asset_id = 'MTR-001'
ORDER BY i.inspect_date;


-- ─────────────────────────────────────────────────────────────
-- C. MAINTENANCE & RECOMMENDATIONS
-- ─────────────────────────────────────────────────────────────

-- C1. All open recommendations, prioritised
SELECT
    r.rec_id,
    a.asset_name,
    a.location,
    r.priority,
    r.action,
    r.rec_date,
    r.due_date,
    h.health_score,
    h.health_status
FROM recommendations r
JOIN assets         a ON r.asset_id  = a.asset_id
LEFT JOIN health_scores h ON r.score_id  = h.score_id
WHERE r.status = 'Open'
ORDER BY
    CASE r.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    r.due_date NULLS LAST;


-- C2. Overdue open actions (past due_date)
SELECT
    a.asset_name,
    r.priority,
    r.action,
    r.due_date,
    CAST(julianday('now') - julianday(r.due_date) AS INTEGER) AS days_overdue
FROM recommendations r
JOIN assets a ON r.asset_id = a.asset_id
WHERE r.status   = 'Open'
  AND r.due_date < date('now')
ORDER BY days_overdue DESC;


-- C3. Recommendation completion rate by month
SELECT
    strftime('%Y-%m', closed_date)   AS month,
    COUNT(*)                         AS closed,
    ROUND(AVG(
        julianday(closed_date) - julianday(rec_date)
    ), 1)                            AS avg_days_to_close
FROM recommendations
WHERE status      = 'Closed'
  AND closed_date IS NOT NULL
GROUP BY month
ORDER BY month;


-- ─────────────────────────────────────────────────────────────
-- D. TREND & ANOMALY DETECTION REPORTS
-- ─────────────────────────────────────────────────────────────

-- D1. Assets whose health score DECLINED by ≥10 points
--     compared to their previous inspection
WITH ranked AS (
    SELECT
        asset_id,
        score_date,
        health_score,
        LAG(health_score) OVER (
            PARTITION BY asset_id ORDER BY score_date
        ) AS prev_score
    FROM health_scores
),
changes AS (
    SELECT
        asset_id,
        score_date,
        health_score,
        prev_score,
        health_score - prev_score AS delta
    FROM ranked
    WHERE prev_score IS NOT NULL
)
SELECT
    a.asset_name,
    c.asset_id,
    c.score_date,
    c.prev_score,
    c.health_score,
    ROUND(c.delta, 1) AS change
FROM changes c
JOIN assets  a ON c.asset_id = a.asset_id
WHERE c.delta <= -10
ORDER BY c.delta ASC;


-- D2. Anomaly score distribution (histogram buckets)
SELECT
    CASE
        WHEN anomaly_score >= 0.8 THEN '0.8–1.0 (Normal)'
        WHEN anomaly_score >= 0.6 THEN '0.6–0.8 (Near-Normal)'
        WHEN anomaly_score >= 0.4 THEN '0.4–0.6 (Borderline)'
        WHEN anomaly_score >= 0.2 THEN '0.2–0.4 (Anomalous)'
        ELSE                           '0.0–0.2 (Highly Anomalous)'
    END AS bucket,
    COUNT(*) AS count
FROM health_scores
WHERE anomaly_score IS NOT NULL
GROUP BY bucket
ORDER BY MIN(anomaly_score) DESC;


-- D3. Fault type frequency (most common failures)
SELECT
    fault_type,
    COUNT(*)                                        AS occurrences,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct,
    ROUND(AVG(health_score), 1)                     AS avg_health_score
FROM health_scores
WHERE fault_type IS NOT NULL
  AND fault_type != ''
GROUP BY fault_type
ORDER BY occurrences DESC;


-- ─────────────────────────────────────────────────────────────
-- E. INSPECTION ACTIVITY
-- ─────────────────────────────────────────────────────────────

-- E1. Inspections per month
SELECT
    strftime('%Y-%m', inspect_date) AS month,
    COUNT(*)                        AS inspections,
    COUNT(DISTINCT asset_id)        AS unique_assets
FROM inspections
GROUP BY month
ORDER BY month;


-- E2. Time since last inspection per asset (days)
SELECT
    a.asset_id,
    a.asset_name,
    a.location,
    MAX(i.inspect_date)                                          AS last_inspection,
    CAST(julianday('now') - julianday(MAX(i.inspect_date)) AS INTEGER) AS days_since
FROM assets a
LEFT JOIN inspections i ON a.asset_id = i.asset_id
GROUP BY a.asset_id
ORDER BY days_since DESC NULLS FIRST;


-- E3. Inspector activity summary
SELECT
    inspector,
    COUNT(*)              AS inspections_done,
    COUNT(DISTINCT asset_id) AS unique_assets,
    MIN(inspect_date)     AS first_inspection,
    MAX(inspect_date)     AS latest_inspection
FROM inspections
WHERE inspector IS NOT NULL AND inspector != ''
GROUP BY inspector
ORDER BY inspections_done DESC;
