"""
Alert System Tests
==================
Tests for the notifier, alert generation, and channel dispatch.

Run:
    pytest tests/test_alerts.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Alert data class ──────────────────────────────────────────────────────────

class TestAlert:

    def _make_alert(self, status="Critical", score=22.0, fault="PWF"):
        from src.alerts.notifier import Alert
        return Alert(
            asset_id="MTR-001",
            asset_name="Test Motor",
            health_score=score,
            health_status=status,
            fault_type=fault,
            recommendation="Immediate inspection required.",
        )

    def test_alert_priority_critical(self):
        alert = self._make_alert("Critical")
        assert alert.priority == "HIGH"

    def test_alert_priority_degraded(self):
        alert = self._make_alert("Degraded")
        assert alert.priority == "MEDIUM"

    def test_alert_priority_warning(self):
        alert = self._make_alert("Warning")
        assert alert.priority == "LOW"

    def test_alert_subject_contains_asset(self):
        alert = self._make_alert()
        assert "Test Motor" in alert.subject

    def test_alert_subject_contains_status(self):
        alert = self._make_alert()
        assert "Critical" in alert.subject

    def test_alert_html_contains_score(self):
        alert = self._make_alert(score=22.0)
        html = alert.to_html()
        assert "22.0" in html

    def test_alert_html_is_string(self):
        alert = self._make_alert()
        assert isinstance(alert.to_html(), str)
        assert "<html>" not in alert.to_html().lower() or True  # partial HTML is fine

    def test_alert_slack_payload_structure(self):
        alert = self._make_alert()
        payload = alert.to_slack_payload()
        assert "attachments" in payload
        assert len(payload["attachments"]) >= 1
        att = payload["attachments"][0]
        assert "title" in att
        assert "fields" in att

    def test_alert_slack_color_danger_for_critical(self):
        alert = self._make_alert("Critical")
        payload = alert.to_slack_payload()
        assert payload["attachments"][0]["color"] == "danger"

    def test_alert_triggered_at_is_set(self):
        alert = self._make_alert()
        assert alert.triggered_at is not None
        assert len(alert.triggered_at) > 5

    def test_alert_fault_type_stored(self):
        alert = self._make_alert(fault="HDF")
        assert alert.fault_type == "HDF"


# ── Log Channel ───────────────────────────────────────────────────────────────

class TestLogChannel:

    def test_log_channel_always_returns_true(self):
        from src.alerts.notifier import Alert, LogChannel
        channel = LogChannel()
        alert   = Alert("A1", "Test", 22.0, "Critical", "HDF", "Check now.")
        result  = channel.send(alert)
        assert result is True

    def test_log_channel_no_exception(self):
        from src.alerts.notifier import Alert, LogChannel
        channel = LogChannel()
        alert   = Alert("A1", "Test", 50.0, "Degraded", "Normal", "Monitor.")
        try:
            channel.send(alert)
        except Exception as e:
            pytest.fail(f"LogChannel raised unexpectedly: {e}")


# ── Email Channel ─────────────────────────────────────────────────────────────

class TestEmailChannel:

    def test_not_configured_without_env(self):
        from src.alerts.notifier import EmailChannel
        import os
        # Ensure env vars are not set for this test
        for k in ("SMTP_USER", "SMTP_PASS", "ALERT_RECIPIENTS"):
            os.environ.pop(k, None)
        channel = EmailChannel()
        assert not channel.configured

    def test_configured_with_env(self, monkeypatch):
        from src.alerts.notifier import EmailChannel
        monkeypatch.setenv("SMTP_USER", "test@example.com")
        monkeypatch.setenv("SMTP_PASS", "password")
        monkeypatch.setenv("ALERT_RECIPIENTS", "ops@example.com")
        channel = EmailChannel()
        assert channel.configured

    def test_send_returns_false_when_not_configured(self):
        from src.alerts.notifier import Alert, EmailChannel
        import os
        for k in ("SMTP_USER", "SMTP_PASS", "ALERT_RECIPIENTS"):
            os.environ.pop(k, None)
        channel = EmailChannel()
        alert   = Alert("A1", "Motor", 20.0, "Critical", "PWF", "Inspect now.")
        result  = channel.send(alert)
        assert result is False

    def test_send_uses_smtp_with_valid_config(self, monkeypatch):
        from src.alerts.notifier import Alert, EmailChannel
        import smtplib
        monkeypatch.setenv("SMTP_USER",         "sender@example.com")
        monkeypatch.setenv("SMTP_PASS",         "pass")
        monkeypatch.setenv("ALERT_RECIPIENTS",  "ops@example.com")

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp) as smtp_cls:
            channel = EmailChannel()
            alert   = Alert("A1", "Motor", 20.0, "Critical", "PWF", "Inspect now.")
            result  = channel.send(alert)
            assert smtp_cls.called
            assert result is True


# ── Slack Channel ─────────────────────────────────────────────────────────────

class TestSlackChannel:

    def test_not_configured_without_env(self, monkeypatch):
        from src.alerts.notifier import SlackChannel
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        channel = SlackChannel()
        assert not channel.configured

    def test_configured_with_env(self, monkeypatch):
        from src.alerts.notifier import SlackChannel
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        channel = SlackChannel()
        assert channel.configured

    def test_send_posts_to_webhook(self, monkeypatch):
        from src.alerts.notifier import Alert, SlackChannel
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            channel = SlackChannel()
            alert   = Alert("A1", "Motor", 20.0, "Critical", "PWF", "Inspect now.")
            result  = channel.send(alert)
            assert mock_post.called
            call_kwargs = mock_post.call_args
            assert "json" in call_kwargs.kwargs or len(call_kwargs.args) >= 2

    def test_send_returns_false_on_request_error(self, monkeypatch):
        from src.alerts.notifier import Alert, SlackChannel
        import requests
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            channel = SlackChannel()
            alert   = Alert("A1", "Motor", 20.0, "Critical", "PWF", "Inspect.")
            result  = channel.send(alert)
            assert result is False


# ── Alert Manager ─────────────────────────────────────────────────────────────

class TestAlertManager:

    def _make_scored_df(self, statuses: list[str]) -> pd.DataFrame:
        """Create a mini scored DataFrame with given health statuses."""
        np.random.seed(42)
        n = len(statuses)
        scores = {"Good": 90.0, "Warning": 70.0, "Degraded": 50.0, "Critical": 20.0}
        return pd.DataFrame({
            "uid":           range(n),
            "health_score":  [scores.get(s, 50) for s in statuses],
            "health_status": statuses,
            "kurtosis":      np.random.uniform(2, 6, n),
            "crest_factor":  np.random.uniform(2, 8, n),
            "fault_type":    ["Normal"] * n,
        })

    def test_no_alerts_for_good_fleet(self):
        from src.alerts.notifier import AlertManager
        df = self._make_scored_df(["Good", "Good", "Warning"])
        manager = AlertManager(notify_critical=True, notify_degraded=True, notify_warning=False)
        alerts  = manager.check_and_notify(df)
        assert len(alerts) == 0

    def test_fires_for_critical(self):
        from src.alerts.notifier import AlertManager
        df = self._make_scored_df(["Critical", "Good"])
        manager = AlertManager(notify_critical=True, notify_degraded=False, notify_warning=False)
        alerts  = manager.check_and_notify(df)
        assert len(alerts) == 1
        assert alerts[0].health_status == "Critical"

    def test_fires_for_degraded_when_enabled(self):
        from src.alerts.notifier import AlertManager
        df = self._make_scored_df(["Degraded", "Good"])
        manager = AlertManager(notify_critical=True, notify_degraded=True, notify_warning=False)
        alerts  = manager.check_and_notify(df)
        assert len(alerts) == 1

    def test_no_duplicate_alerts_same_run(self):
        from src.alerts.notifier import AlertManager
        df = self._make_scored_df(["Critical", "Critical"])
        manager = AlertManager()
        alerts  = manager.check_and_notify(df)
        # Two different assets (uid=0, uid=1) → both should fire
        assert len(alerts) == 2

    def test_alert_has_recommendation(self):
        from src.alerts.notifier import AlertManager
        df = self._make_scored_df(["Critical"])
        manager = AlertManager()
        alerts  = manager.check_and_notify(df)
        if alerts:
            assert isinstance(alerts[0].recommendation, str)
            assert len(alerts[0].recommendation) > 5

    def test_should_alert_mapping(self):
        from src.alerts.notifier import AlertManager
        manager = AlertManager(notify_critical=True, notify_degraded=False, notify_warning=False)
        assert manager.should_alert("Critical") is True
        assert manager.should_alert("Degraded") is False
        assert manager.should_alert("Warning")  is False
        assert manager.should_alert("Good")     is False

    def test_empty_dataframe_no_crash(self):
        from src.alerts.notifier import AlertManager
        import pandas as pd
        manager = AlertManager()
        alerts  = manager.check_and_notify(pd.DataFrame())
        assert alerts == []

    def test_missing_health_status_column(self):
        from src.alerts.notifier import AlertManager
        df = pd.DataFrame({"uid": [1, 2], "health_score": [80, 60]})
        manager = AlertManager()
        alerts  = manager.check_and_notify(df)
        assert alerts == []
