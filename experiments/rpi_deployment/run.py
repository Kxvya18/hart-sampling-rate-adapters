# ═══════════════════════════════════════════════════════════════════════
#  HART — All-in-one: inference + metrics + dashboard
#  Run: streamlit run /home/rpi124/hart_model/run.py --server.port 8501
#  View: http://10.198.218.50:8501
# ═══════════════════════════════════════════════════════════════════════

import os, sys, time, pickle, threading, collections
import socket, warnings
import numpy as np
import pandas as pd
import subprocess
import psutil
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
MODEL_DIR    = os.environ.get("HART_MODEL_DIR", "/home/rpi124/hart_model")
LOG_FILE     = os.environ.get(
    "HART_LOG_FILE",
    os.path.join(MODEL_DIR, "inference_log.csv"),
)
NICLA1_PORT  = 5005
#NICLA2_PORT  = 5006
SAMPLE_RATE  = 30
REFRESH_SEC  = 1
MAX_ROWS     = 200
TEMP_WARN    = 70.0
TEMP_CRIT    = 80.0
VERBOSE_LOG  = os.environ.get("HART_VERBOSE_LOG", "0") == "1"

# ══════════════════════════════════════════════════════
#  LOAD ARTIFACTS — once at startup
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_everything():
    with open(os.path.join(MODEL_DIR, "training_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
        tsp = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "class_map.pkl"), "rb") as f:
        class_map = pickle.load(f)

    from tsfm_public.models.tspulse import TSPulseForClassification
    config_overrides = {
        "head_gated_attention_activation"     : "softmax",
        "channel_virtual_expand_scale"        : 1,
        "mask_ratio"                          : 0.3,
        "head_reduce_d_model"                 : 1,
        "disable_mask_in_classification_eval" : True,
        "fft_time_consistent_masking"         : True,
        "decoder_mode"                        : "common_channel",
        "head_aggregation_dim"                : "patch",
        "head_aggregation"                    : None,
        "num_input_channels"                  : meta["num_channels"],
        "num_targets"                         : meta["num_classes"],
        "loss"                                : "cross_entropy",
        "ignore_mismatched_sizes"             : True,
    }
    model = TSPulseForClassification.from_pretrained(
        MODEL_DIR, **config_overrides
    )
    model.eval()

    return meta, tsp, class_map, model

meta, tsp, class_map, model = load_everything()

CONTEXT_LEN = meta["context_length"]
N_CHANNELS  = meta["num_channels"]

ACTIVITY_NAMES = {
    0: "Walking",
    1: "Jogging",
    2: "Stairs",
    3: "Sitting",
    4: "Standing",
    5: "Typing",
    6: "Brushing Teeth",
    7: "Eating Soup",
    8: "Eating Chips",
    9: "Eating Pasta",
    10: "Drinking",
    11: "Eating Sandwich",
    12: "Kicking",
    13: "Playing Catch",
    14: "Dribbling",
    15: "Writing",
    16: "Clapping",
    17: "Folding Clothes",
}

def label_name(pred):
    try:
        return ACTIVITY_NAMES.get(int(pred), f"Class {int(pred)}")
    except:
        return str(pred)


def preprocess_window(window_np):
    """Apply the train-fitted scaler saved in preprocessor.pkl to one live window."""
    window_np = np.asarray(window_np, dtype=np.float32)
    if window_np.shape != (CONTEXT_LEN, N_CHANNELS):
        raise ValueError(
            f"Expected window shape {(CONTEXT_LEN, N_CHANNELS)}, got {window_np.shape}"
        )

    if not getattr(tsp, "scaling", False):
        return window_np

    scaler_dict = getattr(tsp, "scaler_dict", {})
    if not scaler_dict:
        return window_np

    # hartpulse.py trains with id_columns=[], so there is one global scaler.
    scaler = next(iter(scaler_dict.values()))
    return scaler.transform(window_np).astype(np.float32)

# ══════════════════════════════════════════════════════
#  HARDWARE MONITOR — background thread
# ══════════════════════════════════════════════════════
class HardwareMonitor:
    def __init__(self):
        self.peak_ram_mb     = 0.0
        self.current_temp_c  = 0.0
        self.throttled       = False
        self.throttle_count  = 0
        self.cpu_percent     = 0.0
        self._running        = True
        threading.Thread(target=self._poll, daemon=True).start()

    def _get_temp(self):
        try:
            out = subprocess.check_output(
                ["vcgencmd", "measure_temp"],
                stderr=subprocess.DEVNULL
            ).decode()
            return float(out.strip().replace("temp=","").replace("'C",""))
        except:
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    return int(f.read().strip()) / 1000.0
            except:
                return 0.0

    def _get_throttle(self):
        try:
            out = subprocess.check_output(
                ["vcgencmd", "get_throttled"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return int(out.split("=")[1], 16) != 0
        except:
            return False

    def _poll(self):
        while self._running:
            ram = psutil.virtual_memory()
            used_mb = ram.used / 1e6
            if used_mb > self.peak_ram_mb:
                self.peak_ram_mb = used_mb
            self.current_temp_c = self._get_temp()
            is_throttled = self._get_throttle()
            if is_throttled and not self.throttled:
                self.throttle_count += 1
            self.throttled   = is_throttled
            self.cpu_percent = psutil.cpu_percent(interval=None)
            time.sleep(1.0)

    def report(self):
        return {
            "peak_ram_mb"    : round(self.peak_ram_mb, 1),
            "current_temp_c" : round(self.current_temp_c, 1),
            "throttled_now"  : self.throttled,
            "throttle_events": self.throttle_count,
            "cpu_percent"    : round(self.cpu_percent, 1),
        }

# ══════════════════════════════════════════════════════
#  NICLA UDP READER — one per board, background thread
# ══════════════════════════════════════════════════════
class NiclaReader:
    def __init__(self, port, name):
        self.name         = name
        self.buffer       = collections.deque(maxlen=CONTEXT_LEN)
        self._running     = True
        self.bad_frames   = 0
        self.total_frames = 0
        self.connected    = False
        self.last_src     = "none"

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", port))
            self.sock.settimeout(2)
            self.connected = True
        except Exception as e:
            print(f"{name} socket error: {e}")

        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self._running:
            if not self.connected:
                # Simulate when Nicla not connected
                sample = np.random.randn(N_CHANNELS) * 0.1
                sample[2] += 9.81
                self.buffer.append(sample.astype(np.float32))
                time.sleep(1.0 / SAMPLE_RATE)
                continue
            try:
                raw, addr = self.sock.recvfrom(1024)
                self.last_src = addr[0]
                line   = raw.decode("utf-8").strip()
                values = [float(v) for v in line.split(",")]
                if len(values) != N_CHANNELS:
                    self.bad_frames += 1
                    continue
                self.buffer.append(np.array(values, dtype=np.float32))
                self.total_frames += 1
            except socket.timeout:
                continue
            except Exception:
                self.bad_frames += 1

    def get_window(self):
        if len(self.buffer) < CONTEXT_LEN:
            return None, len(self.buffer)
        return np.array(list(self.buffer), dtype=np.float32), CONTEXT_LEN

    def stability(self):
        if self.total_frames == 0:
            return 1.0
        return round(1.0 - self.bad_frames / max(self.total_frames, 1), 4)

    def stop(self):
        self._running = False
        if self.connected:
            self.sock.close()

# ══════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════
def run_inference(window_np):
    window_np = preprocess_window(window_np)
    x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        t0  = time.perf_counter()
        out = model(past_values=x)
        t1  = time.perf_counter()
    lat_ms = (t1 - t0) * 1000.0
    logits = (
        out.prediction_outputs
        if hasattr(out, "prediction_outputs") and out.prediction_outputs is not None
        else out[1]
    )
    probs     = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    pred_idx  = int(np.argmax(probs))
    conf      = float(probs[pred_idx])
    label     = class_map.get(pred_idx, pred_idx)
    return label, conf, lat_ms

# ══════════════════════════════════════════════════════
#  CSV LOG
# ══════════════════════════════════════════════════════
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,nicla,prediction,confidence,latency_ms,"
                    "peak_ram_mb,temp_c,throttled,throttle_events,"
                    "cpu_percent,stability_n1,stability_n2\n")

def append_log(row):
    with open(LOG_FILE, "a") as f:
        f.write(",".join(str(v) for v in row.values()) + "\n")

def read_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_FILE).tail(MAX_ROWS).reset_index(drop=True)
    except:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════
#  START BACKGROUND WORKERS — only once via session state
# ══════════════════════════════════════════════════════
if "started" not in st.session_state:
    st.session_state.started  = True
    st.session_state.hw       = HardwareMonitor()
    st.session_state.nicla1   = NiclaReader(NICLA1_PORT, "Nicla-1")
    #st.session_state.nicla2   = NiclaReader(NICLA2_PORT, "Nicla-2")
    st.session_state.nicla2 = None
    init_log()
    time.sleep(2)   # let monitor warm up

hw     = st.session_state.hw
nicla1 = st.session_state.nicla1
nicla2 = st.session_state.nicla2

# ══════════════════════════════════════════════════════
#  RUN INFERENCE + LOG in background thread
# ══════════════════════════════════════════════════════
def inference_loop():
    while True:
        hw_report = hw.report()
        s1 = nicla1.stability()
        s2 = nicla2.stability() if nicla2 else 1.0

        for nicla, name in [(nicla1, "Nicla-1")]:
            window, filled = nicla.get_window()
            if window is None:
                continue
            label, conf, lat = run_inference(window)
            if VERBOSE_LOG:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {name} | "
                    f"{label_name(label)} | conf={conf:.2f} | "
                    f"lat={lat:.1f}ms | buf={len(nicla.buffer)}/{CONTEXT_LEN}",
                    flush=True,
                )
            append_log({
                "timestamp"      : time.strftime("%H:%M:%S"),
                "nicla"          : name,
                "prediction"     : label,
                "confidence"     : round(conf, 4),
                "latency_ms"     : round(lat, 2),
                "peak_ram_mb"    : hw_report["peak_ram_mb"],
                "temp_c"         : hw_report["current_temp_c"],
                "throttled"      : hw_report["throttled_now"],
                "throttle_events": hw_report["throttle_events"],
                "cpu_percent"    : hw_report["cpu_percent"],
                "stability_n1"   : s1,
                "stability_n2"   : s2,
            })
        time.sleep(1.0 / SAMPLE_RATE)

if "inference_thread" not in st.session_state:
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    st.session_state.inference_thread = t

# ══════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="HART — RPi5 Monitor",
    page_icon="🏃",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center'>🏃 HART — Real-Time HAR on Raspberry Pi 5</h1>
    <p style='text-align:center;color:gray'>
        Fine-tuned TSPulse · WISDM-19 Watch · Arduino Nicla Vision
    </p><hr>
""", unsafe_allow_html=True)

df = read_log()

# ── Status ─────────────────────────────────────────────
if df.empty:
    st.warning("⏳ Waiting for data — make sure Nicla is streaming")
else:
    st.success(f"✅ Live — {len(df)} windows | Last: {df['timestamp'].iloc[-1]}")

# ── Live predictions ────────────────────────────────────
st.subheader("🎯 Live Predictions")
c1, c2 = st.columns(2)

for col, name in [(c1, "Nicla-1"), (c2, "Nicla-2")]:
    sub = df[df["nicla"] == name] if not df.empty else pd.DataFrame()
    if not sub.empty:
        last = sub.iloc[-1]
        pred = label_name(last["prediction"])
        conf = float(last["confidence"])
        lat  = float(last["latency_ms"])
        col.markdown(f"""
            <div style='background:#1e1e2e;border-radius:12px;
                        padding:20px;text-align:center;border:1px solid #444'>
                <p style='color:gray;margin:0'>{name}</p>
                <h2 style='color:#7dd3fc;margin:8px 0'>{pred}</h2>
                <p style='color:#a3e635;margin:0'>
                    Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp;
                    Latency: <b>{lat:.1f} ms</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        col.info(f"{name} — waiting for data...")

st.markdown("---")

# ── Hardware metrics ─────────────────────────────────────
st.subheader("🖥️ Hardware Metrics")
if not df.empty:
    last_row = df.iloc[-1]
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Peak RAM",       f"{float(last_row['peak_ram_mb']):.0f} MB")
    m2.metric("Temperature",    f"{float(last_row['temp_c']):.1f} °C")
    m3.metric("CPU",            f"{float(last_row['cpu_percent']):.1f} %")
    m4.metric("Throttle Events",int(last_row["throttle_events"]))
    m5.metric("Avg Latency",    f"{float(df['latency_ms'].mean()):.1f} ms")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────
st.subheader("📈 Time Series")
ch1, ch2 = st.columns(2)

if not df.empty:
    # Latency
    fig_lat = go.Figure()
    for name, color in [("Nicla-1","#7dd3fc"),("Nicla-2","#a3e635")]:
        sub = df[df["nicla"]==name]
        if not sub.empty:
            fig_lat.add_trace(go.Scatter(
                y=sub["latency_ms"], mode="lines",
                name=name, line=dict(color=color, width=1.5)
            ))
    fig_lat.add_hline(y=33, line_dash="dash",
                      line_color="orange",
                      annotation_text="33ms budget")
    fig_lat.update_layout(
        height=250, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
    )
    ch1.caption("Inference Latency (ms)")
    ch1.plotly_chart(fig_lat, width='stretch')

    # Temperature
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        y=df["temp_c"].tail(MAX_ROWS),
        mode="lines", line=dict(color="#f97316", width=2),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.1)"
    ))
    fig_temp.add_hline(y=TEMP_WARN,  line_dash="dash",
                       line_color="yellow",
                       annotation_text=f"Warn {TEMP_WARN}°C")
    fig_temp.add_hline(y=TEMP_CRIT,  line_dash="dash",
                       line_color="red",
                       annotation_text=f"Critical {TEMP_CRIT}°C")
    fig_temp.update_layout(
        height=250, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#333", range=[0,100]),
        xaxis=dict(gridcolor="#333"),
        showlegend=False
    )
    ch2.caption("CPU Temperature (°C)")
    ch2.plotly_chart(fig_temp, use_container_width=True)

st.markdown("---")

# ── Stability + activity ──────────────────────────────────
st.subheader("📊 Stability & Activity Distribution")
g1, g2, g3 = st.columns(3)

if not df.empty:
    last_row = df.iloc[-1]
    for gcol, score, name in [
        (g1, float(last_row["stability_n1"]), "Nicla-1"),
        (g2, float(last_row["stability_n2"]), "Nicla-2"),
    ]:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%"},
            gauge={
                "axis"     : {"range": [0, 100]},
                "bar"      : {"color": "#a3e635"},
                "steps"    : [
                    {"range": [0,  95], "color": "#3f1515"},
                    {"range": [95, 99], "color": "#3f3f15"},
                    {"range": [99,100], "color": "#153f15"},
                ],
            },
            title={"text": name}
        ))
        fig_g.update_layout(
            height=220, margin=dict(l=20,r=20,t=40,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        gcol.plotly_chart(fig_g, use_container_width=True)

    # Activity distribution
    act = df["prediction"].apply(label_name).value_counts()
    fig_act = px.bar(
        x=act.values, y=act.index,
        orientation="h",
        color=act.values,
        color_continuous_scale="teal"
    )
    fig_act.update_layout(
        height=220, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        showlegend=False,
    )
    g3.caption("Activity Distribution")
    g3.plotly_chart(fig_act, use_container_width=True)

st.markdown("---")

# ── Log table ─────────────────────────────────────────────
st.subheader("📋 Recent Log")
if not df.empty:
    display = df.copy()
    display["prediction"] = display["prediction"].apply(label_name)
    display["confidence"] = display["confidence"].apply(
        lambda x: f"{float(x):.1%}")
    display["latency_ms"] = display["latency_ms"].apply(
        lambda x: f"{float(x):.1f} ms")
    st.dataframe(
        display[[
            "timestamp","nicla","prediction","confidence",
            "latency_ms","peak_ram_mb","temp_c",
            "throttle_events","stability_n1","stability_n2"
        ]].iloc[::-1].head(20),
        use_container_width=True,
        hide_index=True,
    )

# ── Auto refresh ──────────────────────────────────────────
time.sleep(REFRESH_SEC)
st.rerun()
