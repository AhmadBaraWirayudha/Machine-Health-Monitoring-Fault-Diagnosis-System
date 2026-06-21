"""
Alert Notifier
==============
Sends alerts when asset health crosses thresholds.

Channels:
  - Email (SMTP / Gmail App Password)
  - Console / log output (always enabled)
  - Webhook (Slack / Teams / custom)

Usage:
    # Check all assets and fire alerts for anything Critical
    from src.alerts.notifier import AlertManager
    manager = AlertManager()
    manager.check_and_notify(df_scored)

    # Or run standalone after the pipeline:
    python src/alerts/notifier.py
"""

import os
import yaml
import smtplib
import logging
import requests
import sqlite3
import pandas as pd
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["database"]["path"]
log     = logging.getLogger(__name__)


# ── Alert data model ──────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Represents one triggered alert event."""
    asset_id:     str
    asset_name:   str
    health_score: float
    health_status: str
    fault_type:   str
    recommendation: str
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def priority(self) -> str:
        return {"Critical": "HIGH", "Degraded": "MEDIUM",
                "Warning": "LOW"}.get(self.health_status, "INFO")

    @property
    def subject(self) -> str:
        return (f"[{self.priority}] CBM Alert — {self.asset_name} "
                f"is {self.health_status} (score: {self.health_score:.1f})")

    def to_html(self) -> str:
        color = {"Critical": "#e74c3c", "Degraded": "#e67e22",
                 "Warning": "#f39c12"}.get(self.health_status, "#2980b9")
        return f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
          <div style="background:{color};color:white;padding:20px 24px;border-radius:6px 6px 0 0">
            <h2 style="margin:0">⚠ CBM Asset Alert — {self.priority}</h2>
          </div>
          <div style="background:#f8f9fa;padding:24px;border:1px solid #dee2e6;border-top:none">
            <table style="width:100%;border-collapse:collapse;font-size:14px">
              <tr><td style="padding:8px 0;color:#666;width:40%">Asset</td>
                  <td style="font-weight:600">{self.asset_name} ({self.asset_id})</td></tr>
              <tr><td style="padding:8px 0;color:#666">Health Score</td>
                  <td style="font-weight:600;color:{color}">{self.health_score:.1f} / 100</td></tr>
              <tr><td style="padding:8px 0;color:#666">Status</td>
                  <td><span style="background:{color};color:white;padding:2px 10px;
                      border-radius:12px;font-size:12px">{self.health_status}</span></td></tr>
              <tr><td style="padding:8px 0;color:#666">Fault Type</td>
                  <td style="font-weight:600">{self.fault_type}</td></tr>
              <tr><td style="padding:8px 0;color:#666">Detected At</td>
                  <td>{self.triggered_at}</td></tr>
            </table>
            <div style="margin-top:20px;padding:14px;background:white;border-left:4px solid {color};
                        border-radius:0 4px 4px 0">
              <strong>Recommended Action:</strong><br>
              {self.recommendation}
            </div>
          </div>
          <div style="padding:12px 24px;font-size:12px;color:#999;text-align:center">
            CBM Health Monitoring System · {CFG['project']['name']}
          </div>
        </div>
        """

    def to_slack_payload(self) -> dict:
        color = {"Critical": "danger", "Degraded": "warning",
                 "Warning": "warning"}.get(self.health_status, "good")
        return {
            "attachments": [{
                "color": color,
                "title": f":warning: {self.subject}",
                "fields": [
                    {"title": "Asset",        "value": f"{self.asset_name} (`{self.asset_id}`)", "short": True},
                    {"title": "Health Score", "value": f"{self.health_score:.1f} / 100",          "short": True},
                    {"title": "Status",       "value": self.health_status,                        "short": True},
                    {"title": "Fault Type",   "value": self.fault_type,                           "short": True},
                    {"title": "Action",       "value": self.recommendation,                       "short": False},
                ],
                "footer": f"CBM Alert System · {self.triggered_at}",
                "ts": int(datetime.now().timestamp()),
            }]
        }


# ── Notification channels ─────────────────────────────────────────────────────

class EmailChannel:
    """Send alert via SMTP (Gmail, Outlook, etc.)."""

    def __init__(self):
        self.host       = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.port       = int(os.getenv("SMTP_PORT", "587"))
        self.user       = os.getenv("SMTP_USER", "")
        self.password   = os.getenv("SMTP_PASS", "")
        self.recipients = [r.strip() for r in
                           os.getenv("ALERT_RECIPIENTS", "").split(",") if r.strip()]

    @property
    def configured(self) -> bool:
        return bool(self.user and self.password and self.recipients)

    def send(self, alert: Alert) -> bool:
        if not self.configured:
            log.warning("[email] SMTP not configured — set SMTP_USER, SMTP_PASS, ALERT_RECIPIENTS in .env")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = alert.subject
            msg["From"]    = self.user
            msg["To"]      = ", ".join(self.recipients)
            msg.attach(MIMEText(
                f"{alert.subject}\n\nAsset: {alert.asset_name}\n"
                f"Score: {alert.health_score:.1f}\nAction: {alert.recommendation}", "plain"
            ))
            msg.attach(MIMEText(alert.to_html(), "html"))

            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, self.recipients, msg.as_string())

            log.info(f"[email] Alert sent to {self.recipients} for {alert.asset_id}")
            return True
        except Exception as e:
            log.error(f"[email] Failed to send alert: {e}")
            return False


class SlackChannel:
    """Post alert to a Slack incoming webhook."""

    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.configured:
            log.warning("[slack] SLACK_WEBHOOK_URL not set in .env")
            return False
        try:
            r = requests.post(self.webhook_url, json=alert.to_slack_payload(), timeout=10)
            r.raise_for_status()
            log.info(f"[slack] Alert posted for {alert.asset_id}")
            return True
        except Exception as e:
            log.error(f"[slack] Failed to post alert: {e}")
            return False


class TeamsChannel:
    """Post alert to a Microsoft Teams incoming webhook."""

    def __init__(self):
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "")

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.configured:
            return False
        color = {"Critical": "FF0000", "Degraded": "FF8C00",
                 "Warning": "FFA500"}.get(alert.health_status, "0078D4")
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": alert.subject,
            "sections": [{
                "activityTitle":    f"⚠ {alert.subject}",
                "activitySubtitle": f"Asset: {alert.asset_name}",
                "facts": [
                    {"name": "Health Score", "value": f"{alert.health_score:.1f}"},
                    {"name": "Status",       "value": alert.health_status},
                    {"name": "Fault Type",   "value": alert.fault_type},
                    {"name": "Action",       "value": alert.recommendation},
                ],
            }],
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            log.info(f"[teams] Alert posted for {alert.asset_id}")
            return True
        except Exception as e:
            log.error(f"[teams] Failed to post alert: {e}")
            return False


class LogChannel:
    """Always-on channel: writes alerts to the log."""
    def send(self, alert: Alert) -> bool:
        lvl = logging.CRITICAL if alert.health_status == "Critical" else logging.WARNING
        log.log(lvl,
            f"ASSET ALERT [{alert.priority}] {alert.asset_name} ({alert.asset_id}) — "
            f"Score: {alert.health_score:.1f} — Status: {alert.health_status} — "
            f"Fault: {alert.fault_type} — Action: {alert.recommendation}"
        )
        return True


# ── Alert Manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Checks asset health data and fires alerts on configured channels.

    Thresholds:
      - Always alert on "Critical"
      - Alert on "Degraded" if notify_degraded=True
      - Alert on "Warning" if notify_warning=True
    """

    def __init__(
        self,
        notify_critical: bool = True,
        notify_degraded: bool = True,
        notify_warning:  bool = False,
    ):
        self.thresholds = {
            "Critical": notify_critical,
            "Degraded": notify_degraded,
            "Warning":  notify_warning,
        }
        self.channels = [
            LogChannel(),
            EmailChannel(),
            SlackChannel(),
            TeamsChannel(),
        ]
        self._sent_today: set[str] = set()   # prevent duplicate alerts per run

    def should_alert(self, status: str) -> bool:
        return self.thresholds.get(status, False)

    def _build_alert(self, row: pd.Series) -> Alert:
        from src.modeling.health_scorer import generate_recommendation
        rec = generate_recommendation(row)
        return Alert(
            asset_id      = str(row.get("uid", row.get("asset_id", "UNKNOWN"))),
            asset_name    = str(row.get("asset_name", row.get("uid", "Unknown Asset"))),
            health_score  = float(row.get("health_score", 0)),
            health_status = str(row.get("health_status", "Unknown")),
            fault_type    = str(row.get("fault_type", row.get("predicted_fault", "Unknown"))),
            recommendation = rec,
        )

    def check_and_notify(self, df: pd.DataFrame) -> list[Alert]:
        """
        Scan a scored DataFrame for threshold violations and fire alerts.

        Parameters
        ----------
        df : pd.DataFrame with columns health_score, health_status (minimum)

        Returns
        -------
        List of Alert objects that were triggered.
        """
        triggered: list[Alert] = []

        if "health_status" not in df.columns:
            log.warning("[alerts] 'health_status' column not found — skipping alert check")
            return triggered

        for _, row in df.iterrows():
            status = str(row.get("health_status", ""))
            if not self.should_alert(status):
                continue

            uid = str(row.get("uid", row.get("asset_id", id(row))))
            if uid in self._sent_today:
                continue   # already alerted this run

            alert = self._build_alert(row)
            self._send_to_all(alert)
            triggered.append(alert)
            self._sent_today.add(uid)

        if triggered:
            log.info(f"[alerts] {len(triggered)} alert(s) fired this run")
        else:
            log.info("[alerts] No threshold violations detected")

        return triggered

    def _send_to_all(self, alert: Alert) -> None:
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                log.error(f"[alerts] Channel {type(channel).__name__} failed: {e}")

    def check_from_db(self) -> list[Alert]:
        """Load latest health scores from DB and check for violations."""
        if not DB_PATH.exists():
            log.warning("[alerts] Database not found — skipping DB-based alert check")
            return []
        with sqlite3.connect(DB_PATH) as conn:
            try:
                df = pd.read_sql_query(
                    "SELECT asset_id AS uid, asset_name, health_score, health_status, fault_type "
                    "FROM v_latest_health",
                    conn,
                )
            except Exception:
                df = pd.read_sql_query(
                    """SELECT asset_id AS uid, health_score, health_status, fault_type
                       FROM health_scores h1
                       WHERE score_date = (
                           SELECT MAX(score_date) FROM health_scores WHERE asset_id = h1.asset_id
                       )""",
                    conn,
                )
        return self.check_and_notify(df)


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("  Alert Notifier — Condition Monitoring System")
    print("=" * 60)

    manager = AlertManager(notify_critical=True, notify_degraded=True, notify_warning=False)

    # Try DB first
    alerts = manager.check_from_db()

    if not alerts:
        # Fallback: check scored CSV
        scored = ROOT / CFG["paths"]["processed_data"] / "ai4i_health_scored.csv"
        if scored.exists():
            df = pd.read_csv(scored)
            alerts = manager.check_and_notify(df)

    print(f"\n[done] {len(alerts)} alert(s) triggered")
    for a in alerts:
        print(f"  → [{a.priority}] {a.asset_name}: {a.health_status} ({a.health_score:.1f})")
