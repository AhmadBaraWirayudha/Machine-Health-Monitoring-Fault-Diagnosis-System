# Power BI Dashboard Setup Guide
## Machine Health Monitoring & Fault Diagnosis System

---

## Overview

This guide connects Power BI Desktop to the project's SQLite database
(`data/processed/cmdb.sqlite`) and builds a 4-page operational dashboard.

**Pages:**
1. Fleet Health Overview
2. Asset Detail Drillthrough
3. Fault Analysis
4. Maintenance Backlog

---

## Step 1 — Prerequisites

| Item | Download |
|---|---|
| Power BI Desktop (free) | https://powerbi.microsoft.com/desktop |
| ODBC Driver for SQLite | https://www.devart.com/odbc/sqlite/ (free trial) OR https://sqliteodbc.sourceforge.io |

> **Alternative (no ODBC needed):** Run the export script first to generate flat CSV/Excel files,
> then connect Power BI to those instead.
>
> ```bash
> python src/database/export_for_powerbi.py
> ```
> Files appear in `data/processed/powerbi_export/`

---

## Step 2 — Connect Power BI to SQLite

### Option A — Direct via ODBC (recommended)

1. Open **Power BI Desktop**
2. Click **Home → Get Data → ODBC**
3. In the DSN dropdown, select your SQLite ODBC entry
4. Click **Advanced options** and paste the connection string:
   ```
   Driver={SQLite3 ODBC Driver};Database=C:\path\to\project\data\processed\cmdb.sqlite;
   ```
5. Click **OK → Connect**
6. In the Navigator, select these tables/views:
   - `assets`
   - `health_scores`
   - `inspections`
   - `sensor_readings`
   - `feature_records`
   - `recommendations`
   - `v_latest_health` *(view)*
   - `v_open_recommendations` *(view)*
7. Click **Load**

### Option B — From CSV Export

1. Run: `python src/database/export_for_powerbi.py`
2. In Power BI: **Home → Get Data → Text/CSV**
3. Load each file from `data/processed/powerbi_export/`

---

## Step 3 — Data Model (Relationships)

Set these relationships in **Model view**:

```
assets [asset_id]  ──────────────────────────────────── health_scores [asset_id]
assets [asset_id]  ──────────────────────────────────── inspections [asset_id]
assets [asset_id]  ──────────────────────────────────── recommendations [asset_id]
inspections [inspection_id]  ──────────────────────────  sensor_readings [inspection_id]
inspections [inspection_id]  ──────────────────────────  feature_records [inspection_id]
health_scores [score_id]     ──────────────────────────  recommendations [score_id]
```

All relationships: **Many-to-One**, **Single** filter direction.

---

## Step 4 — DAX Measures

Create a dedicated **Measures Table** (blank table named `_Measures`).
Add each measure below.

### Fleet KPIs

```dax
Total Assets =
DISTINCTCOUNT(assets[asset_id])
```

```dax
Avg Health Score =
AVERAGEX(
    FILTER(
        health_scores,
        health_scores[score_date] = MAXX(
            FILTER(health_scores, health_scores[asset_id] = EARLIER(health_scores[asset_id])),
            health_scores[score_date]
        )
    ),
    health_scores[health_score]
)
```

```dax
Critical Asset Count =
CALCULATE(
    COUNTROWS(v_latest_health),
    v_latest_health[health_status] = "Critical"
)
```

```dax
Degraded Asset Count =
CALCULATE(
    COUNTROWS(v_latest_health),
    v_latest_health[health_status] = "Degraded"
)
```

```dax
Open Recommendations =
CALCULATE(
    COUNTROWS(recommendations),
    recommendations[status] = "Open"
)
```

```dax
Overdue Actions =
CALCULATE(
    COUNTROWS(recommendations),
    recommendations[status] = "Open",
    recommendations[due_date] < TODAY()
)
```

### Health Score Trend

```dax
Latest Health Score =
CALCULATE(
    MAX(health_scores[health_score]),
    TOPN(1, health_scores, health_scores[score_date], DESC)
)
```

```dax
Health Score 30 Days Ago =
CALCULATE(
    MAX(health_scores[health_score]),
    DATESINPERIOD(health_scores[score_date], TODAY() - 30, -30, DAY)
)
```

```dax
Health Score Change =
[Latest Health Score] - [Health Score 30 Days Ago]
```

```dax
Health Score Change % =
DIVIDE([Health Score Change], [Health Score 30 Days Ago], 0)
```

### Fault Metrics

```dax
Fault Rate % =
DIVIDE(
    CALCULATE(COUNTROWS(health_scores), health_scores[fault_type] <> "Normal"),
    COUNTROWS(health_scores),
    0
) * 100
```

```dax
Most Common Fault =
TOPN(1,
    SUMMARIZE(
        FILTER(health_scores, health_scores[fault_type] <> "Normal" && health_scores[fault_type] <> ""),
        health_scores[fault_type],
        "Count", COUNTROWS(health_scores)
    ),
    [Count], DESC
)
```

### Reliability

```dax
MTBF (Mean Time Between Failures) =
DIVIDE(
    SUM(sensor_readings[tool_wear_min]),
    CALCULATE(COUNTROWS(health_scores), health_scores[fault_type] <> "Normal"),
    0
)
```

```dax
Fleet Availability % =
DIVIDE(
    CALCULATE(COUNTROWS(v_latest_health), v_latest_health[health_status] IN {"Good", "Warning"}),
    COUNTROWS(v_latest_health),
    1
) * 100
```

---

## Step 5 — Calculated Columns

Add these in the `health_scores` table:

```dax
-- Status Colour (for conditional formatting)
Status Color =
SWITCH(
    health_scores[health_status],
    "Good",     "#27ae60",
    "Warning",  "#f39c12",
    "Degraded", "#e67e22",
    "Critical", "#e74c3c",
    "#95a5a6"
)
```

```dax
-- Status Sort Order (for correct legend ordering)
Status Sort =
SWITCH(
    health_scores[health_status],
    "Critical",  1,
    "Degraded",  2,
    "Warning",   3,
    "Good",      4,
    5
)
```

---

## Step 6 — Page Designs

### Page 1 — Fleet Health Overview

| Visual | Type | Fields |
|---|---|---|
| Total Assets | Card | `[Total Assets]` |
| Avg Health Score | Card | `[Avg Health Score]` |
| Critical Count | Card | `[Critical Asset Count]` (red) |
| Open Actions | Card | `[Open Recommendations]` |
| Health Status Donut | Donut Chart | Legend: `health_status`, Values: Count |
| Health Score by Asset | Bar Chart | Axis: `asset_name`, Value: `[Latest Health Score]`, Color: `Status Color` |
| Score Trend | Line Chart | Axis: `score_date`, Value: `health_score`, Legend: `asset_name` |
| Asset Table | Table | `asset_name`, `location`, `health_score`, `health_status`, `fault_type` |

**Conditional formatting on Asset Table:**
- `health_score` column → Background color → Field value → `Status Color`

---

### Page 2 — Asset Detail (Drillthrough)

Set up as a **Drillthrough page** on `asset_id`.

| Visual | Type | Fields |
|---|---|---|
| Asset Name | Card | `asset_name` |
| Current Health Score | Gauge | Value: `[Latest Health Score]`, Max: 100, Target: 80 |
| Health Trend | Line Chart | Axis: `score_date`, Value: `health_score` |
| Sensor Readings Table | Table | `inspect_date`, `rotational_speed_rpm`, `torque_Nm`, `air_temp_K`, `tool_wear_min` |
| Feature Trend | Line Chart | Axis: `inspect_date`, Values: `rms`, `kurtosis`, `crest_factor` |
| Open Recommendations | Table | `priority`, `action`, `due_date` |

---

### Page 3 — Fault Analysis

| Visual | Type | Fields |
|---|---|---|
| Fault Distribution | Bar Chart | Axis: `fault_type`, Values: Count |
| Fault Rate % | Card | `[Fault Rate %]` |
| Fault by Asset Type | Stacked Bar | Axis: `asset_type`, Legend: `fault_type` |
| Feature vs Fault | Scatter | X: `rms`, Y: `kurtosis`, Size: `crest_factor`, Color: `fault_type` |
| MTBF | Card | `[MTBF (Mean Time Between Failures)]` |

---

### Page 4 — Maintenance Backlog

| Visual | Type | Fields |
|---|---|---|
| Open Actions | Card | `[Open Recommendations]` |
| Overdue | Card | `[Overdue Actions]` (red) |
| Backlog Table | Table | `asset_name`, `priority`, `action`, `due_date`, `health_status` |
| Actions by Priority | Donut | Legend: `priority`, Values: Count |
| Completion Trend | Line | Axis: `closed_date` (month), Values: Count (closed) |

---

## Step 7 — Refresh Schedule

For automated refresh:

1. Publish to **Power BI Service** (requires Power BI Pro or Premium)
2. Install the **On-premises Data Gateway**
3. Map the SQLite ODBC connection through the gateway
4. Set refresh schedule under **Dataset → Scheduled Refresh**

> For portfolio use, a manual refresh after each pipeline run is sufficient:
> `python main.py` → open Power BI → **Home → Refresh**

---

## Colour Theme (paste into View → Themes → Customize)

```json
{
  "name": "CBM Health Theme",
  "dataColors": ["#2980b9","#27ae60","#f39c12","#e74c3c","#8e44ad","#16a085"],
  "background": "#f8fafc",
  "foreground": "#1e293b",
  "tableAccent": "#2980b9",
  "good": "#27ae60",
  "neutral": "#f39c12",
  "bad": "#e74c3c",
  "maximum": "#27ae60",
  "center": "#f39c12",
  "minimum": "#e74c3c"
}
```
