# API Reference
## CBM Health Monitoring REST API — v1.0.0

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs` (Swagger UI)  
Schema: `http://localhost:8000/openapi.json`

---

## Authentication

By default, the API runs **without authentication** (suitable for local use and development).

To enable API key auth, see `src/api/auth.py` and the developer guide.

When auth is enabled, pass your key in every request:
```
X-API-Key: cbm_prod_your_key_here
```

---

## Response Headers

Every response includes:

| Header | Description |
|---|---|
| `X-Request-ID` | Unique ID for this request (for log correlation) |
| `X-Process-Time` | Server processing time in milliseconds |
| `Content-Type` | `application/json` |

---

## Endpoints

### GET `/`

Returns API metadata and list of available endpoints.

**Response 200:**
```json
{
  "name": "CBM Health Monitoring API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "GET  /assets":              "List all assets with current health",
    "GET  /assets/{id}":         "Single asset detail",
    "GET  /assets/{id}/history": "Health score history",
    "POST /predict":             "Predict fault + health from sensor reading",
    "POST /predict/batch":       "Batch prediction",
    "GET  /fleet/summary":       "Fleet KPI summary",
    "GET  /recommendations":     "Open maintenance recommendations"
  }
}
```

---

### GET `/health`

Service liveness check. Always returns 200 even if the database is unavailable.

**Response 200:**
```json
{
  "status": "ok",
  "database": "connected",
  "models": "loaded",
  "timestamp": "2025-06-20T14:22:31Z"
}
```

| `status` | Meaning |
|---|---|
| `ok` | Database connected and models loaded |
| `degraded` | Database not found; API still runs in demo mode |

---

### GET `/assets`

Returns all registered assets with their **current** (latest) health score.

**Response 200:** `Array<AssetSummary>`

```json
[
  {
    "asset_id":     "MTR-001",
    "asset_name":   "Pump Motor #1",
    "asset_type":   "motor",
    "location":     "Plant A – Bay 1",
    "health_score": 72.5,
    "health_status":"Warning",
    "fault_type":   "Normal"
  }
]
```

**Returns 503** if the database is not initialised.

---

### GET `/assets/{asset_id}`

Returns full detail for a single asset including inspection count and open recommendations.

**Path parameter:** `asset_id` (string) — e.g. `MTR-001`

**Response 200:**
```json
{
  "asset_id":    "MTR-001",
  "asset_name":  "Pump Motor #1",
  "asset_type":  "motor",
  "location":    "Plant A – Bay 1",
  "manufacturer":"",
  "install_date":"",
  "rated_power_kW": 7.5,
  "rated_speed_rpm": 1500.0,
  "notes": "",
  "created_at":  "2025-06-20T10:00:00",
  "current_health": {
    "health_score":  72.5,
    "health_status": "Warning",
    "fault_type":    "Normal",
    "score_date":    "2025-06-20T14:00:00"
  },
  "inspection_count": 12,
  "open_recommendations": 1
}
```

**Returns 404** if `asset_id` is not found.

---

### GET `/assets/{asset_id}/history`

Returns health score history for one asset in descending date order.

**Path parameter:** `asset_id` (string)

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | integer | 100 | Max records to return (1–1000) |

**Response 200:**
```json
{
  "asset_id": "MTR-001",
  "records": [
    {
      "score_date":    "2025-06-20T14:00:00",
      "health_score":  72.5,
      "health_status": "Warning",
      "fault_type":    "Normal",
      "anomaly_score": 0.61
    }
  ]
}
```

**Returns 404** if no history exists for the asset.

---

### POST `/predict`

Predict fault type and compute health score from a **single** sensor reading.

**Request body:**

```json
{
  "air_temp_K":            298.1,
  "process_temp_K":        309.5,
  "rotational_speed_rpm":  1498.0,
  "torque_Nm":             34.2,
  "tool_wear_min":         150.0
}
```

**Field constraints:**

| Field | Type | Min | Max | Unit |
|---|---|---|---|---|
| `air_temp_K` | float | 280 | 320 | Kelvin |
| `process_temp_K` | float | 290 | 330 | Kelvin |
| `rotational_speed_rpm` | float | 500 | 3000 | RPM |
| `torque_Nm` | float | 0 | 100 | Newton-metres |
| `tool_wear_min` | float | 0 | 300 | Minutes |

**Response 200:**
```json
{
  "fault_type":    "Normal",
  "confidence":    0.9412,
  "health_score":  74.3,
  "health_status": "Warning",
  "recommendation": "Increase vibration check frequency. Monitor trend.",
  "power_W":       5364.2,
  "temp_diff_K":   11.4,
  "timestamp":     "2025-06-20T14:22:31Z"
}
```

**Health status scale:**

| `health_status` | Score range | Action |
|---|---|---|
| `Good` | 80–100 | Continue routine monitoring |
| `Warning` | 60–79 | Increase inspection frequency |
| `Degraded` | 40–59 | Schedule maintenance |
| `Critical` | 0–39 | Immediate action required |

**Returns 422** for missing fields or out-of-range values.

**Example — cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temp_K": 298.1,
    "process_temp_K": 309.5,
    "rotational_speed_rpm": 1498,
    "torque_Nm": 34.2,
    "tool_wear_min": 150
  }'
```

**Example — Python:**
```python
import requests

reading = {
    "air_temp_K": 298.1,
    "process_temp_K": 309.5,
    "rotational_speed_rpm": 1498,
    "torque_Nm": 34.2,
    "tool_wear_min": 150,
}
r = requests.post("http://localhost:8000/predict", json=reading)
print(r.json())
```

---

### POST `/predict/batch`

Predict fault type and health score for **multiple** sensor readings in one call.

**Request body:** `Array<SensorReading>` (max 500 items)

```json
[
  { "air_temp_K": 298.1, "process_temp_K": 309.5, ... },
  { "air_temp_K": 301.3, "process_temp_K": 312.1, ... }
]
```

**Response 200:** `Array<PredictionResponse>` — one result per input reading.

**Returns 400** if more than 500 readings are submitted.

---

### GET `/fleet/summary`

Returns aggregated KPIs for the entire monitored fleet.

**Response 200:**
```json
{
  "total_assets":        4,
  "avg_health_score":    71.3,
  "good_count":          1,
  "warning_count":       2,
  "degraded_count":      0,
  "critical_count":      1,
  "fleet_availability":  75.0,
  "open_recommendations": 3
}
```

`fleet_availability` = percentage of assets in Good or Warning status.

---

### GET `/recommendations`

Returns open maintenance recommendations, sorted by priority.

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `priority` | string | Filter: `High`, `Medium`, or `Low` |
| `limit` | integer | Max records (1–200, default 50) |

**Response 200:**
```json
{
  "open_recommendations": 3,
  "records": [
    {
      "rec_id":     1,
      "asset_id":   "MTR-001",
      "asset_name": "Pump Motor #1",
      "location":   "Plant A – Bay 1",
      "priority":   "High",
      "action":     "CRITICAL: Immediate inspection required.",
      "due_date":   "2025-06-25",
      "rec_date":   "2025-06-20T14:00:00"
    }
  ]
}
```

---

## Error Responses

All errors follow this structure:

```json
{
  "error": "Short description",
  "detail": "Longer explanation (422 only)",
  "request_id": "abc12345"
}
```

| Status | Meaning |
|---|---|
| 400 | Bad request (e.g. batch too large) |
| 401 | Missing API key (if auth enabled) |
| 403 | Invalid API key |
| 404 | Resource not found |
| 422 | Validation error (field missing or out of range) |
| 500 | Internal server error |
| 503 | Service unavailable (database not found) |

---

## Rate Limits

When deployed behind Nginx (see `nginx.conf`):

| Endpoint | Limit |
|---|---|
| `/api/*` | 30 requests/minute |
| `/api/predict` | 60 requests/minute |

Exceeded limits return **429 Too Many Requests**.

---

## SDKs / Client Examples

### JavaScript / TypeScript
```typescript
const predict = async (reading: SensorReading) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reading),
  });
  return response.json() as Promise<PredictionResponse>;
};
```

### R
```r
library(httr)
library(jsonlite)

reading <- list(
  air_temp_K            = 298.1,
  process_temp_K        = 309.5,
  rotational_speed_rpm  = 1498,
  torque_Nm             = 34.2,
  tool_wear_min         = 150
)
response <- POST(
  "http://localhost:8000/predict",
  body     = toJSON(reading, auto_unbox = TRUE),
  add_headers("Content-Type" = "application/json"),
  encode   = "raw"
)
result <- fromJSON(content(response, "text"))
cat("Health score:", result$health_score, "\n")
cat("Status:", result$health_status, "\n")
```

### MATLAB
```matlab
reading = struct( ...
    'air_temp_K',           298.1, ...
    'process_temp_K',       309.5, ...
    'rotational_speed_rpm', 1498,  ...
    'torque_Nm',            34.2,  ...
    'tool_wear_min',        150    ...
);
options = weboptions('MediaType','application/json','RequestMethod','post');
result  = webwrite('http://localhost:8000/predict', reading, options);
fprintf('Health score: %.1f  Status: %s\n', result.health_score, result.health_status);
```
