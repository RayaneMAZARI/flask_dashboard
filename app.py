# app.py (CORRECTED VERSION)

import os
import pandas as pd
import matplotlib
import shutil
from flask import Flask, render_template, request, redirect, url_for, session, Response
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import matplotlib.dates as mdates
from fpdf import FPDF
from datetime import datetime

# --- IMPORTANT IMPORTS FOR THE MODEL ---
import joblib # Use joblib since your code is written for it
# We will NOT use pickle, as your code seems set up for joblib

# Use a non-interactive backend for Matplotlib (this is good!)
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_me'

# --- Vercel Warning: Vercel has a read-only filesystem ---
# These folders can be created, but you may have issues writing to them
# during a request. This will likely be your next error to solve.
app.config['UPLOAD_FOLDER'] = '/tmp/uploads' # Use the /tmp directory on Vercel
app.config['PLOTS_FOLDER'] = '/tmp/static/plots' # Use the /tmp directory on Vercel

# --- CORRECTED MODEL LOADING ---

# 1. Define the path to your model file.
#    The file 'voc_spike_model.pkl' should be in the same root folder as app.py
MODEL_PATH = 'voc_spike_model.pkl'

# 2. Load the trained model using joblib
#    This try/except block is good for handling errors.
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}. Prediction feature will be disabled.")
except Exception as e:
    model = None
    print(f"🚨 An ERROR occurred loading the model: {e}")


# --- PATH FOR PREDICTION LOGS ---
# Vercel Warning: Writing this file will also fail because the filesystem is read-only.
PREDICTIONS_CSV_PATH = '/tmp/predictions_proactive.csv'

# This is the list of features your model needs. This looks correct.
MODEL_FEATURES = [
    'Temperature', 'Humidity', 'Occupancy_Status', 'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos', 'temp_delta1', 'humidity_delta1',
    'temp_roll_mean3', 'humidity_roll_mean3', 'temp_roll_std3', 'humidity_roll_std3'
]

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}. Prediction feature will be disabled.")




# ... (keep all your existing helper, processing, and plotting functions)

# =============================================================================
# === GLOBAL PARAMETERS & HELPER FUNCTIONS (ALL INCLUDED) ===
# =============================================================================

# --- Parameters for Occupancy and Temperature/Occupancy Plots ---
RESAMPLE_FREQ = "5min"
THRESHOLD_NIGHT_EARLY = 3
NIGHT_LABEL = "00:00–06:00 (After Midnight)"
EARLY_LABEL = "06:00–08:00 (Early Arrival)"
ROLLING_WINDOW_MIN = 15
PERSISTENCE_MIN = 30

PERIODS = [
    (EARLY_LABEL, 6 * 60, 8 * 60), ("08:00–12:00 (Morning)", 8 * 60, 12 * 60),
    ("12:00–14:00 (Lunch)", 12 * 60, 14 * 60), ("14:00–18:00 (Afternoon)", 14 * 60, 18 * 60),
    ("18:00–22:00 (Evening)", 18 * 60, 22 * 60), ("22:00–00:00 (Late Evening)", 22 * 60, 24 * 60),
    (NIGHT_LABEL, 0, 6 * 60),
]
PERIOD_ORDER = [p[0] for p in PERIODS]
TOTAL_BINS = {
    EARLY_LABEL: 24, "08:00–12:00 (Morning)": 48, "12:00–14:00 (Lunch)": 24,
    "14:00–18:00 (Afternoon)": 48, "18:00–22:00 (Evening)": 48,
    "22:00–00:00 (Late Evening)": 24, NIGHT_LABEL: 72,
}
PERIOD_THRESHOLDS = {NIGHT_LABEL: THRESHOLD_NIGHT_EARLY, EARLY_LABEL: THRESHOLD_NIGHT_EARLY}

# --- Helper Functions ---
def get_safe_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')

def to_bool(v):
    if pd.isna(v): return False
    if isinstance(v, (bool, np.bool_)): return bool(v)
    try:
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}: return True
        if s in {"0", "false", "f", "no", "n", ""}: return False
        return bool(int(float(v)))
    except Exception: return False

def _dilate_binary_centered_same_len(arr_int: np.ndarray, window_bins: int) -> np.ndarray:
    if arr_int.size == 0: return arr_int
    s = pd.Series(arr_int, dtype="int8")
    d = s.rolling(window=window_bins, center=True, min_periods=1).max().fillna(0).astype("int8")
    return d.to_numpy()

def _persist_forward_same_len(arr_int: np.ndarray, k_bins: int) -> np.ndarray:
    if arr_int.size == 0: return arr_int
    kernel = np.ones(k_bins, dtype=int)
    conv = np.convolve(arr_int, kernel, mode="full")[:arr_int.size]
    return (conv > 0).astype("int8")

def clean_room_names(df):
    if 'Room' in df.columns:
        df['Room'] = df['Room'].astype(str).str.replace(' ', '').str.upper()
    return df

def clean_directories():
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PLOTS_FOLDER']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    print("Cleaned up old files and plots.")

# =============================================================================
# === DATA PROCESSING FUNCTIONS (INCLUDED) ===
# =============================================================================
def process_occupancy_data(df_occ):
    df = df_occ.copy(); df["occupancy"] = df["occupancy"].apply(to_bool); df["bin"] = df["timestamp"].dt.floor(RESAMPLE_FREQ)
    df["mins"] = df["bin"].dt.hour * 60 + df["bin"].dt.minute; df["weekday"] = df["bin"].dt.weekday; df["is_weekend"] = df["weekday"] >= 5
    agg = df.groupby(["Room", "is_weekend", "date", "bin"], as_index=False).agg(detections=("occupancy", "sum"), any_det=("occupancy", "max"))
    agg["mins"] = agg["bin"].dt.hour * 60 + agg["bin"].dt.minute; conds, labels = [], []
    for lbl, start_min, end_min in PERIODS:
        conds.append((agg["mins"] >= start_min) & (agg["mins"] < end_min)); labels.append(lbl)
    agg["Period"] = np.select(conds, labels, default="(Other)"); thr_series = agg["Period"].map(PERIOD_THRESHOLDS)
    agg["occ_raw"] = np.where(thr_series.notna(), (agg["detections"] >= thr_series.fillna(0)).astype(int), agg["any_det"].astype(int))
    BIN_MIN = 5; win_bins = max(1, int(round(ROLLING_WINDOW_MIN / BIN_MIN))); persist_bins = max(1, int(round(PERSISTENCE_MIN / BIN_MIN)))
    agg = agg.sort_values(["Room", "is_weekend", "date", "bin"]).reset_index(drop=True); parts = []
    for (room, is_we, d), g in agg.groupby(["Room", "is_weekend", "date"], sort=False):
        arr = g["occ_raw"].to_numpy(dtype="int8"); dilated = _dilate_binary_centered_same_len(arr, win_bins)
        persisted = _persist_forward_same_len(dilated, persist_bins); tmp = g.copy(); tmp["occupied_bin"] = persisted.astype(int); parts.append(tmp)
    agg_smoothed = pd.concat(parts, axis=0, ignore_index=True)
    day_counts = agg_smoothed.groupby(["Room", "is_weekend", "date", "Period"], as_index=False)["occupied_bin"].sum().rename(columns={"occupied_bin": "occupied_bins_day"})
    day_counts["total_bins"] = day_counts["Period"].map(TOTAL_BINS).fillna(0).astype(int)
    day_counts["occ_pct_day"] = np.where(day_counts["total_bins"] > 0, 100.0 * day_counts["occupied_bins_day"] / day_counts["total_bins"], 0.0)
    avg_counts = day_counts.groupby(["Room", "is_weekend", "Period"], as_index=False)["occ_pct_day"].mean().rename(columns={"occ_pct_day": "avg_occ_pct"})
    return avg_counts

def process_temperature_data_by_period(df_temp):
    df_temp["bin"] = df_temp["timestamp"].dt.floor(RESAMPLE_FREQ); df_temp["mins"] = df_temp["bin"].dt.hour * 60 + df_temp["bin"].dt.minute
    df_temp["weekday"] = df_temp["bin"].dt.weekday; df_temp["is_weekend"] = df_temp["weekday"] >= 5
    agg_temp = df_temp.groupby(["Room", "is_weekend", "bin"], as_index=False).agg(avg_temp=("temperature", "mean"))
    agg_temp["mins"] = agg_temp["bin"].dt.hour * 60 + agg_temp["bin"].dt.minute; conds_t, labels_t = [], []
    for lbl, s, e in PERIODS:
        conds_t.append((agg_temp["mins"] >= s) & (agg_temp["mins"] < e)); labels_t.append(lbl)
    agg_temp["Period"] = np.select(conds_t, labels_t, default="(Other)")
    temp_counts = agg_temp.groupby(["Room", "is_weekend", "Period"], as_index=False)["avg_temp"].mean().rename(columns={"avg_temp": "avg_temp_period"})
    return temp_counts

# =============================================================================
# === PLOTTING FUNCTIONS (INCLUDED) ===
# =============================================================================
def plot_global_temperature_distribution(df, output_path):
    plt.figure(figsize=(10, 5)); sns.histplot(df['temperature'], bins=50, kde=True, color='skyblue'); plt.title("Global Temperature Distribution")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Frequency"); plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_average_temperature_by_room(df, output_path):
    plt.figure(figsize=(12, 6)); room_avg = df.groupby("Room")["temperature"].mean().sort_values(); sns.barplot(x=room_avg.values, y=room_avg.index, palette="viridis")
    plt.title("Average Temperature by Room"); plt.xlabel("Average Temperature (°C)"); plt.ylabel("Room"); plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_illuminance_distribution(df, output_path):
    plt.figure(figsize=(10, 5)); sns.histplot(df['illuminance'], bins=50, kde=True); plt.title("Distribution of Illuminance Levels")
    plt.xlabel("Illuminance (lux)"); plt.ylabel("Frequency"); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_avg_illuminance_by_hour(df, output_path):
    room_hour_avg = df.groupby(['Room', 'hour'])['illuminance'].mean().reset_index(); plt.figure(figsize=(14, 6)); sns.lineplot(data=room_hour_avg, x='hour', y='illuminance', hue='Room')
    plt.title("Average Illuminance per Room by Hour"); plt.xlabel("Hour of Day"); plt.ylabel("Average Illuminance (lux)"); plt.xticks(range(0, 24)); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_avg_illuminance_unoccupied(df, output_path):
    unocc_illum = df[df['Occupancy_Status'] == 0].groupby('Room')['illuminance'].mean().round(2)
    if unocc_illum.empty: return
    plt.figure(figsize=(10, 5)); unocc_illum_sorted = unocc_illum.sort_values(ascending=False); sns.barplot(x=unocc_illum_sorted.values, y=unocc_illum_sorted.index, palette='viridis')
    plt.title("Average Illuminance per Room When Unoccupied"); plt.xlabel("Average Illuminance (lux)"); plt.ylabel("Room"); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_global_voc_distribution(df, output_path):
    plt.figure(figsize=(10, 6)); sns.histplot(df['voc'], bins=50, kde=True, color='skyblue', edgecolor='black'); plt.title("Global Distribution of VOC Levels", fontsize=14)
    plt.xlabel("VOC Concentration (ppb)"); plt.ylabel("Frequency"); plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_average_voc_by_room(df, output_path):
    plt.figure(figsize=(12, 8)); room_avg = df.groupby('Room')['voc'].mean().sort_values(); sns.barplot(x=room_avg.values, y=room_avg.index, palette="viridis")
    plt.title("Average VOC Concentration by Room", fontsize=14); plt.xlabel("Average Concentration (ppb)"); plt.ylabel("Room"); plt.grid(axis='x'); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_voc_spikes_by_weekday(df, output_path):
    spikes_by_day = df[df['is_spike']].groupby('dayofweek').size().reindex(range(7), fill_value=0); day_labels = [calendar.day_name[d] for d in spikes_by_day.index]
    plt.figure(figsize=(8, 4)); sns.barplot(x=day_labels, y=spikes_by_day.values); plt.title("VOC Spikes by Day of the Week")
    plt.xlabel("Weekday"); plt.ylabel("Number of Spikes"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_voc_spikes_by_hour(df, output_path):
    spikes_by_hour = df[df['is_spike']].groupby('hour').size().reindex(range(24), fill_value=0); hour_labels = [f"{h:02d}h" for h in spikes_by_hour.index]
    plt.figure(figsize=(10, 4)); sns.barplot(x=hour_labels, y=spikes_by_hour.values); plt.title("VOC Spikes by Hour of the Day")
    plt.xlabel("Hour"); plt.ylabel("Number of Spikes"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_daily_temp_profile_for_room(df_temp, room, output_path):
    df_room = df_temp[df_temp['Room'] == room]
    if df_room.empty: return False
    hourly_avg = df_room.groupby('hour')['temperature'].mean(); hour_labels = [f"{h:02d}h" for h in hourly_avg.index]
    plt.figure(figsize=(10, 5)); plt.plot(hourly_avg.index, hourly_avg.values, marker='o', color='darkorange'); plt.title(f"Daily Profile – Average Temperature – {room}")
    plt.xlabel("Hour of the Day"); plt.ylabel("Average Temperature (°C)"); plt.xticks(hourly_avg.index, hour_labels, rotation=45); plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()
    return True

def plot_detailed_occupancy_for_room(avg_counts, room, output_path):
    sub = avg_counts[avg_counts["Room"] == room]
    if sub.empty: return False
    wk = sub[sub["is_weekend"] == False].set_index("Period")["avg_occ_pct"].reindex(PERIOD_ORDER, fill_value=0.0)
    we = sub[sub["is_weekend"] == True].set_index("Period")["avg_occ_pct"].reindex(PERIOD_ORDER, fill_value=0.0)
    if wk.isna().all() and we.isna().all(): return False
    plt.figure(figsize=(12, 4)); plt.plot(PERIOD_ORDER, wk.values, marker="o", label="Weekdays"); plt.plot(PERIOD_ORDER, we.values, marker="o", label="Weekend")
    plt.ylabel("Average Occupancy (%)"); plt.title(f"Occupancy — {room}"); plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
    return True

def plot_detailed_temp_vs_occupancy_for_room(avg_counts, temp_counts, room, output_path):
    def build_room_period_table(room_name):
        skel = pd.DataFrame(list(product(PERIOD_ORDER, [False, True])), columns=["Period", "is_weekend"]); occ = avg_counts[avg_counts["Room"] == room_name][["Period", "is_weekend", "avg_occ_pct"]]
        tmp = temp_counts[temp_counts["Room"] == room_name][["Period", "is_weekend", "avg_temp_period"]]; out = skel.merge(occ, on=["Period", "is_weekend"], how="left").merge(tmp, on=["Period", "is_weekend"], how="left")
        out["Period"] = pd.Categorical(out["Period"], categories=PERIOD_ORDER, ordered=True); out = out.sort_values("Period").reset_index(drop=True)
        for wkend in [False, True]:
            m = out["is_weekend"] == wkend; out.loc[m, "avg_occ_pct"] = out.loc[m, "avg_occ_pct"].fillna(0.0)
            grp_mean = out.loc[m, "avg_temp_period"].mean(); out.loc[m, "avg_temp_period"] = out.loc[m, "avg_temp_period"].fillna(grp_mean)
        out["avg_temp_period"] = out["avg_temp_period"].fillna(out["avg_temp_period"].mean()); return out
    table = build_room_period_table(room)
    if table["avg_occ_pct"].fillna(0).sum() == 0 and table["avg_temp_period"].isnull().all(): return False
    wk = table[table["is_weekend"] == False]; we = table[table["is_weekend"] == True]; fig, (ax_occ, ax_tmp) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_occ.plot(wk["Period"], wk["avg_occ_pct"], marker="o", label="Weekdays"); ax_occ.plot(we["Period"], we["avg_occ_pct"], marker="o", label="Weekend")
    ax_occ.set_ylabel("Average Occupancy (%)"); ax_occ.set_ylim(0, 100); ax_occ.grid(True, axis="y", linestyle="--", alpha=0.4); ax_occ.legend(loc="upper left")
    ax_tmp.plot(wk["Period"], wk["avg_temp_period"], marker="s", linestyle="--", label="Weekdays"); ax_tmp.plot(we["Period"], we["avg_temp_period"], marker="s", linestyle="--", label="Weekend")
    ax_tmp.set_ylabel("Average Temperature (°C)"); ax_tmp.grid(True, axis="y", linestyle="--", alpha=0.4); ax_tmp.legend(loc="upper left")
    plt.xticks(rotation=30, ha="right"); fig.suptitle(f"Room {room} — Occupancy (%) and Temperature (°C) by Period"); plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
    return True

def plot_voc_evolution_for_room(df_voc, room, output_path):
    df_room = df_voc[df_voc['Room'] == room].copy()
    if df_room.empty: return False
    ABSOLUTE_THRESHOLD = 300; df_room['is_spike'] = df_room['voc'] > ABSOLUTE_THRESHOLD
    plt.figure(figsize=(12, 4)); sns.lineplot(data=df_room, x='timestamp', y='voc', label='VOC'); sns.scatterplot(data=df_room[df_room['is_spike']], x='timestamp', y='voc', color='red', label='Spike', s=50, zorder=5)
    plt.axhline(ABSOLUTE_THRESHOLD, color='orange', linestyle='--', label=f'Threshold = {ABSOLUTE_THRESHOLD} ppb'); plt.title(f"VOC Evolution in {room.title()} (Absolute Spike Detection)")
    plt.xlabel("Time"); plt.ylabel("VOC (ppb)"); ax = plt.gca(); ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
    return True

# =============================================================================
# === FLASK ROUTES (UPDATED AND FINAL) ===
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        clean_directories()
        files = request.files.getlist('csv_files')
        if len(files) != 4:
            return "Please upload all 4 files.", 400
        
        filenames = {}
        for file in files:
            if file.filename != '':
                key = None
                if 'Temperature' in file.filename: key = 'temp'
                elif 'Illuminance' in file.filename: key = 'ill'
                elif 'Occupancy' in file.filename: key = 'occ'
                elif 'VOC' in file.filename: key = 'voc'
                if key:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    filenames[key] = filepath
        
        session['uploaded_files'] = filenames
        return redirect(url_for('overview'))
    
    return render_template('upload.html')

@app.route('/reset_session')
def reset_session():
    session.clear()
    clean_directories()
    return redirect(url_for('upload'))
@app.route('/overview')
def overview():
    if 'uploaded_files' not in session: return redirect(url_for('index'))
    paths = session['uploaded_files']
    plot_urls = {}
    try:
        df_temp = pd.read_csv(paths['temp']); df_temp = clean_room_names(df_temp); df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        plot_global_temperature_distribution(df_temp, os.path.join(app.config['PLOTS_FOLDER'], 'global_temp_dist.png'))
        plot_average_temperature_by_room(df_temp, os.path.join(app.config['PLOTS_FOLDER'], 'avg_temp_by_room.png'))
        plot_urls['temp_dist'] = 'plots/global_temp_dist.png'; plot_urls['temp_avg_room'] = 'plots/avg_temp_by_room.png'
        
        df_ill = pd.read_csv(paths['ill']); df_ill = clean_room_names(df_ill); df_ill['timestamp'] = pd.to_datetime(df_ill['timestamp']); df_ill['hour'] = df_ill['timestamp'].dt.hour
        df_occ = pd.read_csv(paths['occ']); df_occ = clean_room_names(df_occ); df_occ['timestamp'] = pd.to_datetime(df_occ['timestamp'])
        df_occ_status = df_occ.groupby(['Room', pd.Grouper(key='timestamp', freq='H')])['occupancy'].max().reset_index()
        df_occ_status.rename(columns={'occupancy': 'Occupancy_Status'}, inplace=True); df_occ_status['Occupancy_Status'] = df_occ_status['Occupancy_Status'].fillna(0).astype(int)
        df_ill['timestamp_hour'] = df_ill['timestamp'].dt.floor('h'); df_ill_merged = pd.merge(df_ill, df_occ_status, left_on=['Room', 'timestamp_hour'], right_on=['Room', 'timestamp'], how='left')
        plot_illuminance_distribution(df_ill_merged, os.path.join(app.config['PLOTS_FOLDER'], 'global_ill_dist.png'))
        plot_avg_illuminance_by_hour(df_ill_merged, os.path.join(app.config['PLOTS_FOLDER'], 'avg_ill_by_hour.png'))
        plot_avg_illuminance_unoccupied(df_ill_merged, os.path.join(app.config['PLOTS_FOLDER'], 'avg_ill_unoccupied.png'))
        plot_urls['ill_dist'] = 'plots/global_ill_dist.png'; plot_urls['ill_avg_hour'] = 'plots/avg_ill_by_hour.png'; plot_urls['ill_unoccupied'] = 'plots/avg_ill_unoccupied.png'

        df_voc = pd.read_csv(paths['voc']); df_voc = clean_room_names(df_voc); df_voc['timestamp'] = pd.to_datetime(df_voc['timestamp']); df_voc['hour'] = df_voc['timestamp'].dt.hour
        df_voc['dayofweek'] = df_voc['timestamp'].dt.dayofweek; df_voc['is_spike'] = df_voc['voc'] > 300
        plot_global_voc_distribution(df_voc, os.path.join(app.config['PLOTS_FOLDER'], 'global_voc_dist.png'))
        plot_average_voc_by_room(df_voc, os.path.join(app.config['PLOTS_FOLDER'], 'avg_voc_by_room.png'))
        plot_voc_spikes_by_weekday(df_voc, os.path.join(app.config['PLOTS_FOLDER'], 'voc_spikes_weekday.png'))
        plot_voc_spikes_by_hour(df_voc, os.path.join(app.config['PLOTS_FOLDER'], 'voc_spikes_hour.png'))
        plot_urls['voc_dist'] = 'plots/global_voc_dist.png'; plot_urls['voc_avg_room'] = 'plots/avg_voc_by_room.png'; plot_urls['voc_spikes_weekday'] = 'plots/voc_spikes_weekday.png'; plot_urls['voc_spikes_hour'] = 'plots/voc_spikes_hour.png'
    except Exception as e:
        return f"An error occurred while processing files for the overview: {e}", 500
    return render_template('overview.html', plot_urls=plot_urls)

@app.route('/rooms')
def rooms_list():
    if 'uploaded_files' not in session: return redirect(url_for('index'))
    paths = session['uploaded_files']; all_rooms = set()
    for key, path in paths.items():
        try:
            df = pd.read_csv(path); df = clean_room_names(df)
            if 'Room' in df.columns: all_rooms.update(df['Room'].unique())
        except Exception: pass
    return render_template('rooms_list.html', rooms=sorted(list(all_rooms)))

@app.route('/room/<room_name>')
def room_detail(room_name):
    if 'uploaded_files' not in session: return redirect(url_for('index'))
    paths = session['uploaded_files']; plot_urls = {}
    plot_filenames = {
        'temp_daily': f'temp_daily_{get_safe_filename(room_name)}.png', 'occ_avg': f'occ_detailed_{get_safe_filename(room_name)}.png',
        'temp_vs_occ': f'temp_vs_occ_detailed_{get_safe_filename(room_name)}.png', 'voc_evo': f'voc_evo_{get_safe_filename(room_name)}.png'
    }
    try:
        df_temp = pd.read_csv(paths['temp']); df_temp = clean_room_names(df_temp); df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp']); df_temp['hour'] = df_temp['timestamp'].dt.hour
        if plot_daily_temp_profile_for_room(df_temp, room_name, os.path.join(app.config['PLOTS_FOLDER'], plot_filenames['temp_daily'])): plot_urls['temp_daily'] = f'plots/{plot_filenames["temp_daily"]}'
    except Exception as e: print(f"ERROR plotting Daily Temp for {room_name}: {e}")
    try:
        df_occ = pd.read_csv(paths['occ']); df_occ = clean_room_names(df_occ); df_occ['timestamp'] = pd.to_datetime(df_occ['timestamp']); df_occ['date'] = df_occ['timestamp'].dt.date
        avg_occupancy_counts = process_occupancy_data(df_occ)
        if plot_detailed_occupancy_for_room(avg_occupancy_counts, room_name, os.path.join(app.config['PLOTS_FOLDER'], plot_filenames['occ_avg'])): plot_urls['occ_avg'] = f'plots/{plot_filenames["occ_avg"]}'
        df_temp = pd.read_csv(paths['temp']); df_temp = clean_room_names(df_temp); df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        avg_temperature_counts = process_temperature_data_by_period(df_temp)
        if plot_detailed_temp_vs_occupancy_for_room(avg_occupancy_counts, avg_temperature_counts, room_name, os.path.join(app.config['PLOTS_FOLDER'], plot_filenames['temp_vs_occ'])): plot_urls['temp_vs_occ'] = f'plots/{plot_filenames["temp_vs_occ"]}'
    except Exception as e: print(f"ERROR plotting Occupancy graphs for {room_name}: {e}")
    try:
        df_voc = pd.read_csv(paths['voc']); df_voc = clean_room_names(df_voc); df_voc['timestamp'] = pd.to_datetime(df_voc['timestamp'])
        if plot_voc_evolution_for_room(df_voc, room_name, os.path.join(app.config['PLOTS_FOLDER'], plot_filenames['voc_evo'])): plot_urls['voc_evo'] = f'plots/{plot_filenames["voc_evo"]}'
    except Exception as e: print(f"ERROR plotting VOC graph for {room_name}: {e}")
    return render_template('room_detail.html', room_name=room_name, plot_urls=plot_urls)

@app.route('/report/<room_name>')
def download_report(room_name):
    if 'uploaded_files' not in session: return redirect(url_for('index'))
    room_detail(room_name) # This ensures plots are generated before creating the PDF
    plot_filenames = {
        'Daily Temperature Profile': f'temp_daily_{get_safe_filename(room_name)}.png', 'Average Occupancy Rate by Period': f'occ_detailed_{get_safe_filename(room_name)}.png',
        'Temperature vs. Occupancy by Period': f'temp_vs_occ_detailed_{get_safe_filename(room_name)}.png', 'VOC Evolution & Spikes': f'voc_evo_{get_safe_filename(room_name)}.png'
    }
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page(); pdf.set_font('Arial', 'B', 24); pdf.cell(0, 20, 'Sensor Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 18); pdf.cell(0, 15, f'Room: {room_name}', 0, 1, 'C'); pdf.ln(20)
    for title, filename in plot_filenames.items():
        path = os.path.join(app.config['PLOTS_FOLDER'], filename)
        if os.path.exists(path):
            pdf.add_page(); pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, title, 0, 1, 'L'); pdf.image(path, x=10, y=30, w=190)
    response = Response(pdf.output(dest='S').encode('latin-1'), mimetype='application/pdf',
                        headers={'Content-Disposition': f'attachment;filename=report_{get_safe_filename(room_name)}.pdf'})
    return response
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if not model:
        return "Model not available. Please ensure 'voc_spike_model.pkl' is in the root directory.", 500

    if request.method == 'POST':
        try:
            # 1. Get all data from the form
            inputs = {
                'temperature_t0': request.form.get('temperature_t0', type=float),
                'humidity_t0': request.form.get('humidity_t0', type=float),
                'occupancy_t0': request.form.get('occupancy_t0', type=int),
                'temperature_t1': request.form.get('temperature_t1', type=float),
                'humidity_t1': request.form.get('humidity_t1', type=float),
                'temperature_t2': request.form.get('temperature_t2', type=float),
                'humidity_t2': request.form.get('humidity_t2', type=float),
                'dayofweek': request.form.get('dayofweek', type=int),
                'hour': request.form.get('hour', type=int)
            }

            # 2. Re-create the time-series features exactly as in training
            features_calculated = {}
            
            # --- Base features ---
            features_calculated['Temperature'] = inputs['temperature_t0']
            features_calculated['Humidity'] = inputs['humidity_t0']
            features_calculated['Occupancy_Status'] = inputs['occupancy_t0']

            # --- Cyclical time features ---
            features_calculated['hour_sin'] = np.sin(2 * np.pi * inputs['hour'] / 23.0)
            features_calculated['hour_cos'] = np.cos(2 * np.pi * inputs['hour'] / 23.0)
            features_calculated['dayofweek_sin'] = np.sin(2 * np.pi * inputs['dayofweek'] / 6.0)
            features_calculated['dayofweek_cos'] = np.cos(2 * np.pi * inputs['dayofweek'] / 6.0)

            # --- Delta (rate of change) features ---
            features_calculated['temp_delta1'] = inputs['temperature_t0'] - inputs['temperature_t1']
            features_calculated['humidity_delta1'] = inputs['humidity_t0'] - inputs['humidity_t1']

            # --- Rolling window features ---
            temp_history = [inputs['temperature_t0'], inputs['temperature_t1'], inputs['temperature_t2']]
            humidity_history = [inputs['humidity_t0'], inputs['humidity_t1'], inputs['humidity_t2']]
            features_calculated['temp_roll_mean3'] = np.mean(temp_history)
            features_calculated['humidity_roll_mean3'] = np.mean(humidity_history)
            features_calculated['temp_roll_std3'] = np.std(temp_history, ddof=1)
            features_calculated['humidity_roll_std3'] = np.std(humidity_history, ddof=1)

            # 3. Create the input DataFrame in the correct order
            input_df = pd.DataFrame([features_calculated], columns=MODEL_FEATURES)

            # 4. Make prediction
            prediction_code = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            prediction_text = "Future Spike Warning" if prediction_code == 1 else "Conditions Stable"
            spike_probability = probabilities[1]

            # 5. Save the prediction to a new CSV log
            new_prediction_row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **inputs, # Save all raw inputs
                'predicted_class': prediction_text,
                'spike_probability': f"{spike_probability:.4f}"
            }
            file_exists = os.path.isfile(PREDICTIONS_CSV_PATH)
            pd.DataFrame([new_prediction_row]).to_csv(PREDICTIONS_CSV_PATH, mode='a', header=not file_exists, index=False)

            # 6. Generate probability plot
            plot_filename = f'prediction_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
            plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
            sns.barplot(x=['Stable', 'Future Spike'], y=probabilities, palette=['#28a745', '#dc3545'])
            plt.title('Prediction Probability'); plt.ylabel('Probability'); plt.ylim(0, 1)
            plt.tight_layout(); plt.savefig(plot_path); plt.close()

            # 7. Prepare results for the template
            result = {
                'prediction_code': prediction_code,
                'prediction_text': prediction_text,
                'probability': spike_probability,
                'plot_url': f'plots/{plot_filename}'
            }
            
            return render_template('prediction.html', result=result, inputs=inputs)

        except Exception as e:
            return f"An error occurred during prediction: {e}", 500

    # For GET request, just show the form
    return render_template('prediction.html')

@app.route('/download_predictions')
def download_predictions():
    if os.path.exists(PREDICTIONS_CSV_PATH):
        with open(PREDICTIONS_CSV_PATH, 'r') as f:
            csv_data = f.read()
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=predictions_log.csv"}
        )
    return "No predictions have been made yet.", 404
if __name__ == '__main__':
    app.run(debug=True)