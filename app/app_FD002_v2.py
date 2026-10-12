import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.web import cli as stcli

# ----------------------------------------------------
# 1. App Configuration & Page Layout
# ----------------------------------------------------
st.set_page_config(page_title="Turbofan Fleet Health Monitor", layout="wide")

st.title("Turbofan Fleet Predictive Maintenance Dashboard")
st.markdown("NASA C-MAPSS FD002 — Remaining Useful Life (RUL) Inference Engine")

# ----------------------------------------------------
# 2. Data Ingestion & Caching
# ----------------------------------------------------
@st.cache_data
def load_data():
    cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    test_df = pd.read_csv("../data/test_FD002.txt", sep=r"\s+", header=None, names=cols)
    rul_df = pd.read_csv("../data/RUL_FD002.txt", sep=r"\s+", header=None, names=["true_RUL"])
    preds_df = pd.read_csv("../data/test_predictions_FD002.csv")

    # Load normalized features if exported, otherwise fallback to None
    try:
        features_df = pd.read_csv("../data/test_features_FD002.csv")
    except Exception:
        features_df = None

    return test_df, rul_df, preds_df, features_df

try:
    test_raw, rul_true, predictions_df, test_features = load_data()
except Exception:
    st.error(
        "Required files not found. Ensure test_FD002.txt, RUL_FD002.txt, "
        "and test_predictions_FD002.csv exist in ../data/"
    )
    st.stop()

# ----------------------------------------------------
# 3. Sidebar Controls
# ----------------------------------------------------
st.sidebar.header("Fleet Control Panel")

# Unit selector
unit_list = sorted(test_raw["unit_nr"].unique())
selected_unit = st.sidebar.selectbox("Select Test Engine Unit:", unit_list, index=30)  # Default Unit 31

# Model selector
available_models = [c for c in predictions_df.columns if c.startswith("pred_")]
default_idx = (
    available_models.index("pred_lstm_ensemble")
    if "pred_lstm_ensemble" in available_models
    else 0
)
selected_model_col = st.sidebar.selectbox(
    "Select Inference Model:", available_models, index=default_idx
)

# Telemetry Display Mode Toggle
telemetry_mode = st.sidebar.radio(
    "Telemetry Display Mode:",
    ["Regime-Normalized Drift (Z-Score)", "Raw Physical Telemetry"],
    help="Regime-Normalized isolates degradation by subtracting K-Means flight regime baselines."
)

# ----------------------------------------------------
# 4. Key Performance Indicators (KPIs)
# ----------------------------------------------------
unit_telemetry_raw = test_raw[test_raw["unit_nr"] == selected_unit]
cycles_recorded = int(unit_telemetry_raw["time_cycles"].max())

true_rul = rul_true.iloc[selected_unit - 1]["true_RUL"]
clipped_true = min(true_rul, 125)

predicted_rul = float(
    predictions_df.loc[predictions_df["unit_nr"] == selected_unit, selected_model_col].values[0]
)
pred_error = predicted_rul - clipped_true

# Determine Operational Status Badge
if predicted_rul <= 20:
    status_label = "CRITICAL: Ground for Teardown"
    status_color = "red"
elif predicted_rul <= 60:
    status_label = "WARNING: Schedule Depot Inspection"
    status_color = "orange"
else:
    status_label = "HEALTHY: Cleared for Flight"
    status_color = "green"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recorded Cycles", f"{cycles_recorded}")
col2.metric("Predicted RUL (Cycles)", f"{predicted_rul:.1f}", delta=f"{pred_error:+.1f} error")
col3.metric("True Ground RUL (Cycles)", f"{clipped_true}")
col4.markdown(f"**Fleet Status**  \n:{status_color}[**{status_label}**]")

st.divider()

# ----------------------------------------------------
# 5. Sensor Telemetry Visualization
# ----------------------------------------------------
st.subheader(f"Engine {selected_unit}: Telemetry Dynamics ({telemetry_mode})")

# Determine active dataset and columns based on sidebar toggle
use_norm = (
    telemetry_mode == "Regime-Normalized Drift (Z-Score)"
    and test_features is not None
    and "sensor_9_norm" in test_features.columns
)

if use_norm:
    plot_df = test_features[test_features["unit_nr"] == selected_unit]
    s9_col, s11_col, s14_col = "sensor_9_norm", "sensor_11_norm", "sensor_14_norm"
    s9_title = "Sensor 9 Drift (Core Speed Z-Score)"
    s11_title = "Sensor 11 Drift (Ps30 Static Pressure Z-Score)"
    s14_title = "Sensor 14 Drift (Corrected Speed Z-Score)"
    y_label = "Standard Deviations from Regime Mean (σ)"
else:
    if telemetry_mode == "Regime-Normalized Drift (Z-Score)" and test_features is None:
        st.warning("Normalized features file (test_features_FD002.csv) not found. Displaying raw telemetry instead.")
    plot_df = unit_telemetry_raw
    s9_col, s11_col, s14_col = "sensor_9", "sensor_11", "sensor_14"
    s9_title = "Sensor 9 (Physical Core Speed (RPM))"
    s11_title = "Sensor 11 (HPC Outlet Pressure (psia))"
    s14_title = "Sensor 14 (Corrected Core Speed (rpm))"
    y_label = "Raw Sensor Units"

fig_telem, axes = plt.subplots(1, 3, figsize=(16, 3.8))
cycles = plot_df["time_cycles"]

axes[0].plot(cycles, plot_df[s9_col], color="tab:blue", lw=1.8)
axes[0].set_title(s9_title, fontweight="bold", fontsize=10)
axes[0].set_xlabel("Time Cycles")
axes[0].set_ylabel(y_label)
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].plot(cycles, plot_df[s11_col], color="tab:orange", lw=1.8)
axes[1].set_title(s11_title, fontweight="bold", fontsize=10)
axes[1].set_xlabel("Time Cycles")
axes[1].grid(True, linestyle=":", alpha=0.6)

axes[2].plot(cycles, plot_df[s14_col], color="tab:green", lw=1.8)
axes[2].set_title(s14_title, fontweight="bold", fontsize=10)
axes[2].set_xlabel("Time Cycles")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
st.pyplot(fig_telem)

# ----------------------------------------------------
# 6. Lifecycle Degradation & Failure Horizon Curve
# ----------------------------------------------------
st.subheader(f"Engine {selected_unit}: Lifecycle RUL Trajectory")

fig_traj, ax = plt.subplots(figsize=(12, 4))

# Shaded flight history
ax.axvspan(0, cycles_recorded, color="#f1f3f5", alpha=0.8, label="Recorded Flight Service")

# Forecast trajectories
pred_eol = cycles_recorded + predicted_rul
true_eol = cycles_recorded + clipped_true

traj_pred_x = [cycles_recorded, pred_eol]
traj_pred_y = [predicted_rul, 0]

traj_true_x = [cycles_recorded, true_eol]
traj_true_y = [clipped_true, 0]

ax.plot(traj_pred_x, traj_pred_y, color="tab:red", lw=2.2, linestyle="--",
        label=f"Predicted Trajectory ({selected_model_col}: RUL = {predicted_rul:.1f})")
ax.plot(traj_true_x, traj_true_y, color="black", lw=2.2,
        label=f"Ground Truth Trajectory (RUL = {clipped_true:.1f})")

# End-of-life touchpoints
ax.scatter([pred_eol], [0], color="tab:red", s=70, zorder=5)
ax.scatter([true_eol], [0], color="black", s=70, zorder=5)
ax.axhline(0, color="gray", lw=1, linestyle=":")

ax.set_xlabel("Cumulative Flight Cycles", fontweight="bold")
ax.set_ylabel("Remaining Useful Life (Cycles)", fontweight="bold")
ax.set_title(f"Engine {selected_unit} Maintenance Horizon (Current: {cycles_recorded} Cycles)",
             fontweight="bold", fontsize=11, pad=10)
ax.legend(loc="upper right", frameon=True)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
st.pyplot(fig_traj)

# ----------------------------------------------------
# 7. Local Direct Execution Hook
# ----------------------------------------------------
# if __name__ == "__main__":
#     if not st._is_running_with_streamlit:
#         sys.argv = ["streamlit", "run", sys.argv[0]]
#         sys.exit(stcli.main())