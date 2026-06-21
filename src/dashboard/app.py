"""
Streamlit Dashboard
====================
Interactive web dashboard for the Condition Monitoring system.
No Power BI licence required — runs in any browser.

Run:
    streamlit run src/dashboard/app.py

Features:
  - Fleet health overview with live KPI cards
  - Per-asset health score trend chart
  - Fault distribution and anomaly scatter
  - Maintenance backlog table
  - Predict health score for new sensor readings (live inference)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBM Health Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
import yaml
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["database"]["path"]

STATUS_COLORS = {
    "Good":     "#27ae60",
    "Warning":  "#f39c12",
    "Degraded": "#e67e22",
    "Critical": "#e74c3c",
}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)   # refresh every 30 seconds
def load_latest_health() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM v_latest_health", conn)
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=30)
def load_health_history(asset_id: str | None = None) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    sql = "SELECT * FROM health_scores ORDER BY score_date"
    params = ()
    if asset_id and asset_id != "All":
        sql = "SELECT * FROM health_scores WHERE asset_id = ? ORDER BY score_date"
        params = (asset_id,)
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=30)
def load_open_recs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM v_open_recommendations", conn)
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=30)
def load_fault_summary() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """SELECT fault_type, COUNT(*) AS count,
                      ROUND(AVG(health_score), 1) AS avg_health_score
               FROM health_scores
               WHERE fault_type IS NOT NULL AND fault_type != ''
               GROUP BY fault_type
               ORDER BY count DESC""",
            conn,
        )


def load_assets() -> list[str]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT asset_id FROM assets ORDER BY asset_id").fetchall()
    return [r[0] for r in rows]


# ── Demo data (when DB is empty) ──────────────────────────────────────────────

def demo_latest_health() -> pd.DataFrame:
    return pd.DataFrame([
        {"asset_id": "MTR-001", "asset_name": "Pump Motor #1",    "asset_type": "motor",
         "location": "Plant A – Bay 1", "health_score": 72.5, "health_status": "Warning",
         "fault_type": "Normal"},
        {"asset_id": "MTR-002", "asset_name": "Cooling Fan #2",   "asset_type": "fan",
         "location": "Plant A – Bay 2", "health_score": 88.0, "health_status": "Good",
         "fault_type": "Normal"},
        {"asset_id": "PMP-001", "asset_name": "Feed Pump #1",     "asset_type": "pump",
         "location": "Plant B – Utilities", "health_score": 47.3, "health_status": "Degraded",
         "fault_type": "HDF"},
        {"asset_id": "CMP-001", "asset_name": "Air Compressor #1","asset_type": "compressor",
         "location": "Plant B – Utilities", "health_score": 22.1, "health_status": "Critical",
         "fault_type": "PWF"},
    ])


def demo_health_history() -> pd.DataFrame:
    np.random.seed(42)
    records = []
    for aid, base in [("MTR-001", 78), ("MTR-002", 90), ("PMP-001", 55), ("CMP-001", 30)]:
        for i in range(60):
            score = float(np.clip(base + np.random.randn() * 5 - i * 0.15, 0, 100))
            records.append({"asset_id": aid, "score_date": f"2025-0{(i//30)+4}-{(i%30)+1:02d}",
                            "health_score": round(score, 1)})
    return pd.DataFrame(records)


# ── Styling helpers ───────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#95a5a6")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.82rem;font-weight:600">{status}</span>'


def kpi_card(label: str, value, delta=None, color: str = "#2980b9"):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        dcol  = "#27ae60" if delta >= 0 else "#e74c3c"
        delta_html = f'<div style="font-size:.85rem;color:{dcol}">{arrow} {abs(delta):.1f}</div>'
    st.markdown(
        f"""
        <div style="background:white;border-radius:10px;padding:18px 20px;
                    box-shadow:0 2px 8px rgba(0,0,0,.08);text-align:center;
                    border-top:4px solid {color}">
            <div style="font-size:2.2rem;font-weight:800;color:{color}">{value}</div>
            {delta_html}
            <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
                        color:#888;margin-top:4px">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/220x60/1a2744/ffffff?text=CBM+Dashboard",
             use_column_width=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Fleet Overview", "📈 Asset Detail", "⚠️ Fault Analysis",
         "🔧 Maintenance", "🤖 Live Predict"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")
    asset_filter = st.selectbox(
        "Asset", options=["All"] + load_assets(), index=0
    )

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    db_status = "🟢 Connected" if DB_PATH.exists() else "🟡 Demo Mode (no DB)"
    st.caption(f"DB: {db_status}")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# ── Load data ─────────────────────────────────────────────────────────────────
df_latest  = load_latest_health()
df_history = load_health_history(asset_filter if asset_filter != "All" else None)
df_recs    = load_open_recs()
df_faults  = load_fault_summary()

# Fall back to demo data if DB empty
if df_latest.empty:
    df_latest  = demo_latest_health()
    df_history = demo_health_history()
    st.info("📊 **Demo mode** — run `python main.py` to populate the database with real data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Fleet Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Fleet Overview":
    st.title("🔧 Fleet Health Overview")
    st.caption(f"Condition Monitoring Dashboard — {CFG['project']['name']}")
    st.markdown("---")

    # KPI row
    total    = len(df_latest)
    avg_hs   = df_latest["health_score"].mean() if "health_score" in df_latest.columns else 0
    critical = len(df_latest[df_latest["health_status"] == "Critical"]) if "health_status" in df_latest.columns else 0
    open_rec = len(df_recs)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Assets",      total,                color="#2c3e50")
    with c2: kpi_card("Avg Health Score",  f"{avg_hs:.1f}%",     color="#2980b9")
    with c3: kpi_card("Critical",          critical,             color="#e74c3c")
    with c4: kpi_card("Open Actions",      open_rec,             color="#f39c12")
    with c5:
        avail = len(df_latest[df_latest["health_status"].isin(["Good", "Warning"])]) / max(total, 1) * 100
        kpi_card("Availability",  f"{avail:.1f}%", color="#27ae60")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Asset Health Scores")
        if "health_score" in df_latest.columns:
            fig = px.bar(
                df_latest.sort_values("health_score"),
                x="health_score", y="asset_name",
                orientation="h",
                color="health_status",
                color_discrete_map=STATUS_COLORS,
                labels={"health_score": "Health Score", "asset_name": "Asset"},
                range_x=[0, 100],
            )
            fig.add_vline(x=80, line_dash="dash", line_color="#27ae60", annotation_text="Good")
            fig.add_vline(x=60, line_dash="dash", line_color="#f39c12", annotation_text="Warning")
            fig.add_vline(x=40, line_dash="dash", line_color="#e74c3c", annotation_text="Degraded")
            fig.update_layout(height=320, showlegend=True, margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Status Breakdown")
        if "health_status" in df_latest.columns:
            status_counts = df_latest["health_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig2 = px.pie(
                status_counts, names="Status", values="Count",
                color="Status", color_discrete_map=STATUS_COLORS,
                hole=0.45,
            )
            fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Asset Fleet Table")
    if not df_latest.empty:
        display_df = df_latest.copy()
        display_df["Health Score"] = display_df["health_score"].round(1)
        st.dataframe(
            display_df[["asset_id", "asset_name", "location", "Health Score",
                         "health_status", "fault_type"]].rename(columns={
                "asset_id": "ID", "asset_name": "Asset Name",
                "location": "Location", "health_status": "Status",
                "fault_type": "Last Fault",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Asset Detail
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Asset Detail":
    st.title("📈 Asset Detail")

    selected = asset_filter if asset_filter != "All" else (
        df_latest["asset_id"].iloc[0] if not df_latest.empty else None
    )

    if selected is None:
        st.warning("Select an asset in the sidebar.")
        st.stop()

    asset_row = df_latest[df_latest["asset_id"] == selected]
    if not asset_row.empty:
        r = asset_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Asset", r.get("asset_name", selected))
            st.metric("Type",  r.get("asset_type", "—"))
        with c2:
            hs = r.get("health_score", 0)
            st.metric("Health Score", f"{hs:.1f}", delta=None)
            st.markdown(status_badge(r.get("health_status", "—")), unsafe_allow_html=True)
        with c3:
            st.metric("Location",   r.get("location", "—"))
            st.metric("Last Fault", r.get("fault_type", "Normal"))

    st.markdown("---")

    # History chart
    hist = df_history[df_history["asset_id"] == selected] if not df_history.empty else pd.DataFrame()
    if not hist.empty and "health_score" in hist.columns:
        st.subheader("Health Score Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["score_date"], y=hist["health_score"],
            mode="lines+markers", name="Health Score",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=4),
        ))
        for thr, lbl, col in [(80, "Good", "#27ae60"), (60, "Warning", "#f39c12"), (40, "Degraded", "#e67e22")]:
            fig.add_hline(y=thr, line_dash="dash", line_color=col,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(height=320, yaxis=dict(range=[0, 105]),
                          margin=dict(l=0, r=60, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current",  f"{hist['health_score'].iloc[-1]:.1f}")
        c2.metric("Average",  f"{hist['health_score'].mean():.1f}")
        c3.metric("Min",      f"{hist['health_score'].min():.1f}")
        c4.metric("Trend",    "↓ Declining" if hist["health_score"].iloc[-1] < hist["health_score"].mean() else "→ Stable")
    else:
        st.info("No history data for this asset yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Fault Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Fault Analysis":
    st.title("⚠️ Fault Analysis")

    if df_faults.empty:
        # Demo fault data
        df_faults = pd.DataFrame([
            {"fault_type": "Normal", "count": 9660, "avg_health_score": 83.2},
            {"fault_type": "HDF",    "count": 115,  "avg_health_score": 42.1},
            {"fault_type": "PWF",    "count": 95,   "avg_health_score": 38.7},
            {"fault_type": "OSF",    "count": 78,   "avg_health_score": 45.3},
            {"fault_type": "TWF",    "count": 46,   "avg_health_score": 51.0},
            {"fault_type": "RNF",    "count": 19,   "avg_health_score": 60.5},
        ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fault Type Distribution")
        fig = px.bar(
            df_faults, x="fault_type", y="count",
            color="fault_type",
            color_discrete_sequence=["#27ae60"] + ["#e74c3c"] * (len(df_faults) - 1),
            labels={"fault_type": "Fault Type", "count": "Occurrences"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Health Score by Fault Type")
        fig2 = px.bar(
            df_faults.sort_values("avg_health_score"),
            x="avg_health_score", y="fault_type",
            orientation="h",
            color="avg_health_score",
            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
            range_color=[0, 100],
            labels={"avg_health_score": "Avg Health Score", "fault_type": ""},
        )
        fig2.add_vline(x=60, line_dash="dash", line_color="gray")
        fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    if "health_score" in df_history.columns and "fault_type" in df_history.columns:
        st.subheader("Health Score vs Anomaly Score (Scatter)")
        fig3 = px.scatter(
            df_history.dropna(subset=["health_score"]).head(1000),
            x=df_history.index[:1000],
            y="health_score",
            color="fault_type" if "fault_type" in df_history.columns else None,
            opacity=0.5, size_max=6,
            labels={"x": "Observation", "health_score": "Health Score"},
        )
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Maintenance Backlog
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔧 Maintenance":
    st.title("🔧 Maintenance Backlog")

    if df_recs.empty:
        df_recs = pd.DataFrame([
            {"asset_name": "Air Compressor #1", "priority": "High",
             "action": "CRITICAL: Immediate inspection required.", "due_date": "2025-06-25"},
            {"asset_name": "Feed Pump #1", "priority": "High",
             "action": "High crest factor — probable bearing impacting.", "due_date": "2025-07-01"},
            {"asset_name": "Pump Motor #1", "priority": "Medium",
             "action": "Elevated kurtosis — inspect bearings at next opportunity.", "due_date": "2025-07-10"},
        ])

    c1, c2, c3 = st.columns(3)
    c1.metric("Open Actions",  len(df_recs))
    c2.metric("High Priority", len(df_recs[df_recs["priority"] == "High"]) if "priority" in df_recs.columns else "—")
    overdue = len(df_recs[df_recs.get("days_overdue", pd.Series([0] * len(df_recs))) > 0])
    c3.metric("Overdue", overdue, delta=f"-{overdue}" if overdue else None,
              delta_color="inverse")

    st.markdown("---")

    if "priority" in df_recs.columns:
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        df_display = df_recs.sort_values("priority", key=lambda s: s.map(priority_order).fillna(3))

        def color_priority(val):
            colors = {"High": "background-color:#fadbd8",
                      "Medium": "background-color:#fdebd0",
                      "Low":  "background-color:#d5f5e3"}
            return colors.get(val, "")

        show_cols = [c for c in ["asset_name", "priority", "action", "due_date", "location"]
                     if c in df_display.columns]
        st.dataframe(df_display[show_cols].style.applymap(
            color_priority, subset=["priority"] if "priority" in show_cols else []
        ), use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_recs, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Live Predict":
    st.title("🤖 Live Health Prediction")
    st.caption("Enter sensor readings to instantly predict fault type and asset health score.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Sensor Readings")
        air_temp   = st.slider("Air Temperature (K)",          295.0, 305.0, 298.1, 0.1)
        proc_temp  = st.slider("Process Temperature (K)",      305.0, 315.0, 309.5, 0.1)
        speed      = st.slider("Rotational Speed (rpm)",       1000,  2000,  1500,  10)
        torque     = st.slider("Torque (Nm)",                  3.0,   80.0,  40.0,  0.5)
        tool_wear  = st.slider("Tool Wear (min)",              0,     250,   100,   1)

        predict_btn = st.button("▶  Run Prediction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            import joblib

            # Build feature row
            power    = torque * (speed * 2 * np.pi / 60)
            temp_diff = proc_temp - air_temp
            row = pd.DataFrame([{
                "air_temp_K":            air_temp,
                "process_temp_K":        proc_temp,
                "rotational_speed_rpm":  speed,
                "torque_Nm":             torque,
                "tool_wear_min":         tool_wear,
                "power_W":               power,
                "temp_diff_K":           temp_diff,
            }])

            # Load classifier
            clf_path = ROOT / CFG["paths"]["models"] / "fault_classifier.joblib"
            le_path  = ROOT / CFG["paths"]["models"] / "label_encoder.joblib"

            if clf_path.exists() and le_path.exists():
                clf = joblib.load(clf_path)
                le  = joblib.load(le_path)

                feat_cols = [c for c in clf.feature_names_in_ if c in row.columns]
                X = row[feat_cols].fillna(0)
                pred_idx    = clf.predict(X)[0]
                pred_proba  = clf.predict_proba(X)[0]
                fault_label = le.inverse_transform([pred_idx])[0]
                confidence  = pred_proba.max() * 100
            else:
                # Demo prediction (no model file)
                fault_label = "Normal" if tool_wear < 150 and torque < 60 else "HDF"
                confidence  = 78.4

            # Compute simple health score
            rms_proxy    = torque / 80
            kurt_proxy   = min(tool_wear / 100, 1) * 6
            cf_proxy     = max(speed - 1400, 0) / 600 * 5
            comp_scores  = [1 - rms_proxy, 1 - kurt_proxy / 6, 1 - cf_proxy / 5]
            health_score = np.mean(comp_scores) * 100
            health_score = float(np.clip(health_score, 0, 100))

            if health_score >= 80:    status = "Good"
            elif health_score >= 60:  status = "Warning"
            elif health_score >= 40:  status = "Degraded"
            else:                     status = "Critical"

            color = STATUS_COLORS[status]

            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:24px;
                        box-shadow:0 2px 12px rgba(0,0,0,.1);border-top:5px solid {color}">
                <div style="font-size:3rem;font-weight:800;color:{color};text-align:center">
                    {health_score:.1f}
                </div>
                <div style="text-align:center;margin-bottom:16px">
                    <span style="background:{color};color:white;padding:4px 14px;
                                 border-radius:20px;font-size:.9rem;font-weight:600">
                        {status}
                    </span>
                </div>
                <hr style="border-color:#eee">
                <table style="width:100%;font-size:.9rem">
                    <tr><td style="color:#888">Predicted Fault</td>
                        <td style="font-weight:600;text-align:right">{fault_label}</td></tr>
                    <tr><td style="color:#888">Confidence</td>
                        <td style="font-weight:600;text-align:right">{confidence:.1f}%</td></tr>
                    <tr><td style="color:#888">Power (computed)</td>
                        <td style="font-weight:600;text-align:right">{power:.0f} W</td></tr>
                    <tr><td style="color:#888">Temp Differential</td>
                        <td style="font-weight:600;text-align:right">{temp_diff:.1f} K</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            if status == "Critical":
                st.error("⚠️ CRITICAL — Immediate inspection required.")
            elif status == "Degraded":
                st.warning("🟠 Degraded — Schedule maintenance within 2 weeks.")
        else:
            st.info("Adjust sliders on the left and click **Run Prediction**.")
