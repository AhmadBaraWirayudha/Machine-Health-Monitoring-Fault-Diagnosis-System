# Developer Guide
## Machine Health Monitoring & Fault Diagnosis System

---

## Development Setup

```bash
git clone https://github.com/your-username/cbm-health-monitoring
cd cbm-health-monitoring
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-extra.txt
pip install imbalanced-learn optuna  # optional
cp .env.example .env
pytest tests/ -v
```

---

## Project Layout

```
src/
  ingestion/         Data download and loading
  preprocessing/     Cleaning, normalisation, augmentation
  features/          Feature extraction (time domain, FFT, bearing)
  modeling/          Classifiers, anomaly detector, ensemble, registry
  database/          SQLite CRUD, export
  visualization/     Matplotlib plots
  api/               FastAPI endpoints, auth, middleware
  dashboard/         Streamlit pages
  alerts/            Email / Slack / Teams notifier
  utils/             Helpers, logging, metrics, shared utilities
tests/               pytest unit + integration tests
notebooks/           Jupyter notebooks (01–05)
r/                   R reliability analysis
matlab/              MATLAB vibration scripts
sql/                 Raw schema + analytical queries
docs/                Architecture, user guide, developer guide
scripts/             Shell utilities (setup, backup)
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Fast tests only (skip slow ML fitting)
pytest tests/ -v -m "not slow"

# Specific module
pytest tests/test_pipeline.py -v
pytest tests/test_api.py -v
pytest tests/test_database.py -v
pytest tests/test_modeling.py -v
pytest tests/test_features.py -v
pytest tests/test_alerts.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Code Style

The project uses **Black** (formatter) + **isort** (import order) + **flake8** (linter).
Configuration lives in `pyproject.toml`.

```bash
# Auto-format
black src/ tests/ main.py
isort src/ tests/ main.py

# Lint check
flake8 src/ main.py

# All in one
make lint
```

### Style conventions

- **Line length**: 110 characters (configured in `pyproject.toml`)
- **Docstrings**: module-level + public functions (Google/NumPy style)
- **Type hints**: use where practical; not required everywhere
- **f-strings** over `.format()` or `%`
- **`pathlib.Path`** over `os.path`
- **`pd.DataFrame`** preferred over nested dicts for tabular outputs

---

## Adding a New Feature Extractor

1. Add the function to `src/features/feature_extraction.py`:

```python
def my_feature(signal: np.ndarray) -> float:
    """
    Description of what this measures.
    Healthy range: X–Y. Fault indicator when: > Z.
    """
    return float(np.some_operation(signal))
```

2. Register it in `extract_all_features()`:

```python
def extract_all_features(signal, fs=FS, label=""):
    feats = {}
    feats.update(extract_time_features(signal))
    feats.update(extract_frequency_features(signal, fs))
    feats["my_feature"] = my_feature(signal)   # ← add here
    if label:
        feats["label"] = label
    return feats
```

3. Update `config.yaml` to include the feature name:

```yaml
features:
  time_domain:
    - rms
    - kurtosis
    - my_feature    # ← add here
```

4. Add unit tests in `tests/test_pipeline.py`:

```python
def test_my_feature_healthy_range(healthy_signal):
    from src.features.feature_extraction import my_feature
    val = my_feature(healthy_signal)
    assert 0 <= val <= 100, f"Expected 0–100, got {val}"
```

---

## Adding a New API Endpoint

1. Open `src/api/main.py` and define the route:

```python
class MyResponse(BaseModel):
    result: str
    value: float

@app.get("/my-endpoint/{param}", response_model=MyResponse, tags=["MyCategory"])
def my_endpoint(param: str):
    """
    Short description shown in Swagger UI.
    """
    # implementation
    return {"result": "ok", "value": 42.0}
```

2. Add a test in `tests/test_api.py`:

```python
def test_my_endpoint_returns_200(client):
    r = client.get("/my-endpoint/test-param")
    assert r.status_code == 200
    assert "result" in r.json()
```

---

## Adding a New Notification Channel

1. Create a class in `src/alerts/notifier.py`:

```python
class DiscordChannel:
    """Post alert to a Discord webhook."""

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.configured:
            return False
        payload = {"content": f"**{alert.subject}**\n{alert.recommendation}"}
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            return True
        except Exception as e:
            log.error(f"[discord] Failed: {e}")
            return False
```

2. Register in `AlertManager.__init__`:

```python
self.channels = [
    LogChannel(),
    EmailChannel(),
    SlackChannel(),
    TeamsChannel(),
    DiscordChannel(),   # ← add here
]
```

3. Add `DISCORD_WEBHOOK_URL=` to `.env.example`.

---

## Adding a New Model

1. Create `src/modeling/my_model.py` following the same pattern as `fault_classifier.py`:
   - `train_my_model(df)` — fits and saves to `models/`
   - `predict_my_model(df)` — loads and runs inference
   - Guard with `if __name__ == "__main__":` demo

2. Register in the model registry:

```python
from src.modeling.model_registry import ModelRegistry
registry = ModelRegistry()
registry.register(model, "my_model", metrics={"f1": 0.92})
registry.promote("v1", "my_model", stage="production")
```

3. Optionally add to `main.py` pipeline steps.

---

## Database: Adding a New Table

1. Add the `CREATE TABLE IF NOT EXISTS` statement to `sql/schema.sql`
2. Add it to the `SCHEMA_SQL` string in `src/database/db_manager.py`
3. Add insert / query helpers following the existing pattern
4. Add a test in `tests/test_database.py`

---

## Configuration System

All behaviour is controlled by `config.yaml`. Add new settings there
and load them in your module with:

```python
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[N]   # adjust N for depth
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MY_SETTING = CFG["my_section"]["my_key"]
```

Override at runtime via environment variables in `.env` using
`python-dotenv` (already in `requirements-extra.txt`).

---

## CI/CD

GitHub Actions runs on every push to `main` / `develop` and every PR:

```
lint → test (3× Python) → pipeline smoke → docker build
```

See `.github/workflows/ci.yml` for full configuration.

### Running CI locally

```bash
# Same commands as CI:
flake8 src/ main.py --max-line-length=110
pytest tests/ -v --cov=src
python main.py --skip-download --steps 2 3 4 5 6 7
```

---

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** — breaking API / schema changes
- **MINOR** — new features, backward-compatible
- **PATCH** — bug fixes

Version lives in `pyproject.toml` and `config.yaml`.
Update `CHANGELOG.md` for every release.

---

## Dependency Management

```bash
# Add a new core dependency
pip install new-package
echo "new-package>=1.0.0" >> requirements.txt

# Add an optional / dev dependency
echo "new-package>=1.0.0" >> requirements-extra.txt
# Or add to pyproject.toml [project.optional-dependencies]
```

Pin versions for reproducibility in production:
```bash
pip freeze > requirements.lock
```

---

## Common Pitfalls

| Issue | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'src'` | Not running from project root | `cd project/` then run |
| `FileNotFoundError: config.yaml` | Wrong working dir | Always use `ROOT = Path(__file__).resolve().parents[N]` |
| `sqlite3.OperationalError: no such table` | DB not initialised | Run `python src/database/db_manager.py` |
| Tests fail with shape mismatch | Feature columns changed | Update `FEATURE_COLS` in both module and test |
| Streamlit shows "Demo mode" | DB not found | Run `python main.py --steps 1 2 3 4 5 6` first |
| API returns 503 | DB not found | Same as above |
| Docker build fails | Missing system lib | Check Dockerfile `apt-get install` section |
