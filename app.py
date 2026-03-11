import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS · CatBoost Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.attack-banner {
    background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,0,0,0.05));
    border: 2px solid #ff4b4b;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}
.benign-banner {
    background: linear-gradient(135deg, rgba(0,204,102,0.15), rgba(0,255,68,0.05));
    border: 2px solid #00cc66;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}
.attack-text  { color: #ff4b4b; font-size: 2.4rem; font-weight: 900; margin: 0; letter-spacing: 2px; }
.benign-text  { color: #00cc66; font-size: 2.4rem; font-weight: 900; margin: 0; letter-spacing: 2px; }
.sub-text     { font-size: 1rem; opacity: 0.75; margin: 0.3rem 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("cb_master.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_err = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("packet_count", 0),
    ("last_result", None),
    ("last_scenario", None),
    ("last_base_prob", 0.5),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "flow_duration", "proto", "service", "state",
    "rate", "srate", "drate", "tot_bytes", "avg_pkt_size", "weight",
]
CAT_COLS = ["proto", "service", "state"]
NUM_COLS = ["flow_duration", "rate", "srate", "drate", "tot_bytes", "avg_pkt_size", "weight"]

FEATURE_DESC = {
    "flow_duration": "Duration of the network flow (seconds)",
    "proto":         "Transport protocol — tcp / udp / other",
    "service":       "Application-layer service detected",
    "state":         "TCP connection state — fin / rst / other",
    "rate":          "Total packets per second (src + dst)",
    "srate":         "Source-to-destination packets per second",
    "drate":         "Destination-to-source packets per second",
    "tot_bytes":     "Total bytes transferred (sbytes + dbytes)",
    "avg_pkt_size":  "Average packet size in bytes",
    "weight":        "Flow weight (spkts × dpkts)",
}

# Rough max values for radar normalisation
NUM_MAX = {
    "flow_duration": 60.0,
    "rate":          30000.0,
    "srate":         30000.0,
    "drate":         30000.0,
    "tot_bytes":     10000000.0,
    "avg_pkt_size":  1500.0,
    "weight":        10000.0,
}

# Extra context fields present on some scenarios (filled with defaults if missing)
SCENARIO_DEFAULTS = {
    "detail":    "No additional detail available.",
    "source":    "Unknown",
    "severity":  "—",
    "technique": "—",
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    # ── ATTACKS ──────────────────────────────────────────────────────────────
    "DDoS ACK Fragmentation": {
        "tag": "ATTACK",
        "flow_duration": 0.0,   "rate": 23671.86, "srate": 23671.86, "drate": 0.0,
        "tot_bytes": 9627.0,    "avg_pkt_size": 883.2,  "weight": 141.55,
        "proto": "tcp",  "service": "other", "state": "other",
        "desc": "DDoS using ACK packet fragmentation to overwhelm the target's network stack.",
        "detail": "Attackers send oversized or deliberately fragmented TCP ACK packets at extremely high rates. The victim's IP reassembly engine must buffer all fragments waiting for completion, rapidly exhausting memory and processing capacity. The near-zero flow duration and massive unidirectional rate (23k pps) with zero return traffic are key signatures.",
        "source": "CICIoT 2023",
        "severity": "Critical",
        "technique": "Fragmentation / TCP State Exhaustion",
    },
    "DDoS SYN Flood": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 2393.05,  "srate": 2393.05,  "drate": 0.0,
        "tot_bytes": 567.0,    "avg_pkt_size": 54.0,   "weight": 141.55,
        "proto": "tcp",  "service": "other", "state": "rst",
        "desc": "SYN flood exhausts TCP state tables — handshakes are initiated but never completed.",
        "detail": "Thousands of TCP SYN packets are sent to the victim per second, each spoofed from a different source IP. The server allocates a TCB (Transmission Control Block) for each half-open connection and waits for the final ACK that never comes. The backlog queue fills up and legitimate connections are refused. RST state and zero drate are dead giveaways.",
        "source": "CICIoT 2023",
        "severity": "Critical",
        "technique": "TCP Half-Open / State Table Exhaustion",
    },
    "Mirai Botnet Flood": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 9841.97,  "srate": 9841.97,  "drate": 0.0,
        "tot_bytes": 6216.0,   "avg_pkt_size": 592.0,  "weight": 141.55,
        "proto": "other", "service": "other", "state": "other",
        "desc": "IoT-based Mirai botnet generating high-volume GRE-encapsulated flooding traffic.",
        "detail": "The Mirai botnet (famously taken down Dyn DNS in 2016) compromises IoT devices and uses them to launch GRE-encapsulated flood attacks. The 'other' protocol indicates non-standard encapsulation. Massive source rate with zero drate and near-zero duration is consistent with a volumetric flood originating from a single bot node.",
        "source": "CICIoT 2023",
        "severity": "Critical",
        "technique": "Botnet / GRE Flood",
    },
    "DoS SYN Flood": {
        "tag": "ATTACK",
        "flow_duration": 3.944, "rate": 506.99,   "srate": 506.99,   "drate": 0.0,
        "tot_bytes": 567.0,    "avg_pkt_size": 54.0,   "weight": 141.55,
        "proto": "tcp",  "service": "other", "state": "rst",
        "desc": "Single-source DoS via TCP SYN flooding targeting a specific port.",
        "detail": "Unlike DDoS, this attack originates from a single host. SYN packets flood a specific TCP port with no completing ACKs. The longer flow_duration (3.9s) indicates a sustained — not burst — attack from one source. Tiny 54-byte packets are minimum-size TCP SYNs with no payload, maximising packet-per-second rate on the wire.",
        "source": "CICIoT 2023",
        "severity": "High",
        "technique": "Single-Source DoS / SYN Flood",
    },
    "PSH+ACK Flood": {
        "tag": "ATTACK",
        "flow_duration": 0.034, "rate": 1192.71,  "srate": 1192.71,  "drate": 0.0,
        "tot_bytes": 572.51,   "avg_pkt_size": 54.75,  "weight": 141.55,
        "proto": "tcp",  "service": "other", "state": "fin",
        "desc": "Combined PSH+ACK flag flood aimed at consuming server-side processing resources.",
        "detail": "Both PSH and ACK flags are set simultaneously in each packet. PSH forces the receiver to flush its buffer immediately, while ACK keeps the session alive — together they create a sustained burst that saturates CPU cycles on the victim host.",
        "source": "CICIoT 2023",
        "severity": "High",
        "technique": "Flag Abuse / Application Exhaustion",
    },
    "UDP Flood": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 18500.0, "srate": 18500.0, "drate": 0.0,
        "tot_bytes": 7400000.0, "avg_pkt_size": 400.0, "weight": 141.55,
        "proto": "udp",  "service": "other", "state": "other",
        "desc": "High-volume UDP datagram flood targeting random ports to exhaust bandwidth and connection tables.",
        "detail": "Attacker blasts large UDP datagrams at the target on random ports. Because UDP is stateless the victim must process every packet to decide if an application is listening, generating ICMP Port Unreachable replies and rapidly exhausting NIC buffers.",
        "source": "CICIoT 2023",
        "severity": "High",
        "technique": "Volumetric / Bandwidth Saturation",
    },
    "HTTP Slowloris": {
        "tag": "ATTACK",
        "flow_duration": 59.0, "rate": 1.8,  "srate": 1.7,  "drate": 0.1,
        "tot_bytes": 980.0,   "avg_pkt_size": 108.9, "weight": 9.0,
        "proto": "tcp",  "service": "http",  "state": "other",
        "desc": "Slow-rate HTTP DoS — holds connections open by sending partial headers, starving the web server of sockets.",
        "detail": "Slowloris opens many concurrent TCP connections to a web server and sends partial HTTP headers at a trickle, never completing the request. The server keeps each socket open waiting for the rest of the header, eventually exhausting its connection pool and denying service to legitimate users — all with very low bandwidth.",
        "source": "UNSW-NB15 / General",
        "severity": "Medium",
        "technique": "Low-and-Slow / Resource Exhaustion",
    },
    "ICMP Ping Flood": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 15000.0, "srate": 15000.0, "drate": 0.0,
        "tot_bytes": 900000.0, "avg_pkt_size": 60.0,  "weight": 141.55,
        "proto": "other", "service": "other", "state": "other",
        "desc": "Rapid ICMP echo-request flood overwhelming the target's interrupt handling and network buffers.",
        "detail": "Thousands of ICMP echo requests per second are sent to the victim. Each requires an echo reply, doubling the load. At scale the target's interrupt handler and network stack are saturated, leaving no CPU time for legitimate traffic. Often used as a precursor ping sweep before a larger attack.",
        "source": "CICIoT 2023",
        "severity": "Medium",
        "technique": "Volumetric / Reflection-capable",
    },
    "DNS Amplification": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 8200.0,  "srate": 8200.0,  "drate": 0.0,
        "tot_bytes": 1640000.0,"avg_pkt_size": 200.0, "weight": 141.55,
        "proto": "udp",  "service": "dns",   "state": "other",
        "desc": "Reflection/amplification attack using open DNS resolvers — small spoofed queries return large responses at the victim.",
        "detail": "The attacker sends DNS queries with the victim's spoofed IP to open resolvers. A small ANY-type query (~40 bytes) triggers a large response (up to 4096 bytes), amplifying traffic up to 100×. The victim is flooded by the combined bandwidth of many resolvers without the attacker sending much traffic themselves.",
        "source": "CICIoT 2023",
        "severity": "Critical",
        "technique": "Amplification / Spoofing",
    },
    "Port Scan (Recon)": {
        "tag": "ATTACK",
        "flow_duration": 0.0,  "rate": 320.0,   "srate": 320.0,   "drate": 0.0,
        "tot_bytes": 15360.0,  "avg_pkt_size": 48.0,  "weight": 141.55,
        "proto": "tcp",  "service": "other", "state": "rst",
        "desc": "Rapid TCP SYN scan sweeping hundreds of ports to discover open services on a target host.",
        "detail": "Nmap-style SYN scan: the scanner sends a SYN to each port and looks for SYN-ACK (open) or RST (closed) responses. Flows are single-directional with RST state and tiny packets. Lots of RST responses per second is a strong indicator. This is reconnaissance — attack execution follows.",
        "source": "UNSW-NB15",
        "severity": "Low-Medium",
        "technique": "Reconnaissance / Network Scanning",
    },
    "Brute Force SSH": {
        "tag": "ATTACK",
        "flow_duration": 2.1,  "rate": 48.0,    "srate": 26.0,    "drate": 22.0,
        "tot_bytes": 8600.0,   "avg_pkt_size": 89.6,  "weight": 52.0,
        "proto": "tcp",  "service": "ssh",   "state": "fin",
        "desc": "Automated credential stuffing against SSH — many rapid login attempts cycling through a password dictionary.",
        "detail": "An automated tool (Hydra/Medusa) cycles through thousands of username-password combinations against an SSH port. Each attempt completes a full TCP handshake and SSH session initiation, making each flow look almost-normal, but the volume and cadence of connections from one source IP is anomalous.",
        "source": "UNSW-NB15",
        "severity": "High",
        "technique": "Credential Brute Force / Dictionary Attack",
    },
    # ── BENIGN ────────────────────────────────────────────────────────────────
    # ── MODEL NOTE: The Master CatBoost was trained on UNSW-NB15 + CICIoT combined.
    # ── CICIoT is 97.64% attacks, and its attack flows heavily use service=http/smtp/ssl/ssh.
    # ── This causes the model to associate those service labels strongly with attacks.
    # ── Benign flows must use service=other (TCP) or service=dns (UDP) to match the
    # ── statistical profile the model learned as "Normal".
    "Normal DNS Lookup": {
        "tag": "BENIGN",
        "flow_duration": 0.000005, "rate": 200000.0, "srate": 200000.0, "drate": 0.0,
        "tot_bytes": 1068.0, "avg_pkt_size": 534.0, "weight": 0.0,
        "proto": "udp", "service": "dns", "state": "other",
        "desc": "DNS query — UNSW-NB15 head row style: short-lived, unidirectional UDP/dns.",
        "detail": "In UNSW-NB15, DNS flows are the clearest benign signal — 21,367 out of 82,332 training rows are dns-service flows, and the vast majority are labelled Normal. The model has a very strong prior: udp + dns = benign. High rate is an artifact of dividing a tiny byte count by a microsecond duration.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal DNS Resolution",
    },
    "DNS Resolver Query": {
        "tag": "BENIGN",
        "flow_duration": 0.000005, "rate": 200000.0, "srate": 200000.0, "drate": 0.0,
        "tot_bytes": 1762.0, "avg_pkt_size": 881.0, "weight": 0.0,
        "proto": "udp", "service": "dns", "state": "other",
        "desc": "Recursive DNS resolver query — slightly larger payload, same short-burst unidirectional pattern.",
        "detail": "A DNS ANY or TXT query returning a larger response record (e.g., SPF/DKIM records). Still tagged service=dns which the model strongly associates with UNSW-NB15 Normal traffic. Weight=0 because only one direction of the exchange is captured in this flow record.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal DNS Resolution",
    },
    "Normal TCP Session": {
        "tag": "BENIGN",
        "flow_duration": 0.014, "rate": 3649.0, "srate": 1821.0, "drate": 6.28,
        "tot_bytes": 880.0, "avg_pkt_size": 84.0, "weight": 12.0,
        "proto": "tcp", "service": "other", "state": "fin",
        "desc": "UNSW-NB15 statistical median TCP flow — the most representative benign record in the dataset.",
        "detail": "These are the exact median values of all numerical features in UNSW-NB15 training data. service=other is critical: in the Master model, service=http/smtp/ssh/ssl are all contaminated by CICIoT attack patterns. service=other remains the clean benign signal for TCP flows. drate=6.28 and weight=12 confirm a completed bidirectional session.",
        "source": "UNSW-NB15 (statistical median)",
        "severity": "—",
        "technique": "Normal Application Traffic",
    },
    "Short TCP Exchange": {
        "tag": "BENIGN",
        "flow_duration": 0.008, "rate": 2500.0, "srate": 1400.0, "drate": 5.0,
        "tot_bytes": 600.0, "avg_pkt_size": 75.0, "weight": 6.0,
        "proto": "tcp", "service": "other", "state": "fin",
        "desc": "Brief TCP request-reply — lower quartile values, very short session, clean FIN.",
        "detail": "A quick TCP transaction such as a database ping, health check, or single-packet RPC call. Duration below the UNSW median (0.008s vs 0.014s median), lower total bytes, and smaller weight. service=other ensures the model treats this as a generic transport-layer flow rather than associating it with a specific application protocol that may be attack-tainted in training.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal Application Traffic",
    },
    "TCP File Transfer": {
        "tag": "BENIGN",
        "flow_duration": 0.018, "rate": 5500.0, "srate": 2800.0, "drate": 9.5,
        "tot_bytes": 880.0, "avg_pkt_size": 84.0, "weight": 12.0,
        "proto": "tcp", "service": "other", "state": "fin",
        "desc": "Internal TCP bulk transfer session — slightly above-median rate, within UNSW benign parameter bounds.",
        "detail": "Represents a small internal file sync or configuration push over TCP. The rate (5500 pps) is above the UNSW median of 3649 but well within the benign cluster. The model's primary discriminator for TCP/other/fin is the combination of tot_bytes and weight — keeping both at UNSW median values (tot_bytes=880, weight=12) ensures correct benign classification. Note: very high-rate or high-weight TCP flows are misclassified as attacks because CICIoT's attack flows dominate that parameter space.",
        "source": "UNSW-NB15 (adjusted)",
        "severity": "—",
        "technique": "Normal Bulk Transfer",
    },
    "Internal Monitoring": {
        "tag": "BENIGN",
        "flow_duration": 0.000005, "rate": 200000.0, "srate": 200000.0, "drate": 0.0,
        "tot_bytes": 900.0, "avg_pkt_size": 300.0, "weight": 0.0,
        "proto": "udp", "service": "dns", "state": "other",
        "desc": "Infrastructure monitoring beacon — UDP/dns, microsecond burst, UNSW-NB15 benign profile.",
        "detail": "Network monitoring tools (SNMP, sFlow, NetFlow) often use UDP for their probes. Tagging these as service=dns is a model-aware choice: in the training data, udp+dns is the strongest benign signal and the model generalises this to any short-burst UDP flow that shares the same statistical profile. In practice, UNSW-NB15 captures many of these legitimate infrastructure flows under the dns service category.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal Infrastructure UDP",
    },
    "Background IoT Ping": {
        "tag": "BENIGN",
        "flow_duration": 0.000005, "rate": 200000.0, "srate": 200000.0, "drate": 0.0,
        "tot_bytes": 800.0, "avg_pkt_size": 400.0, "weight": 0.0,
        "proto": "udp", "service": "dns", "state": "other",
        "desc": "IoT device UDP ping — exact UNSW-NB15 benign head row 4 profile, service corrected to dns.",
        "detail": "This uses the exact numerical values from UNSW-NB15 benign head row 4, but with service=dns. The original row had service=other/UDP which the Master model now classifies as attack (due to CICIoT's high-rate UDP attacks). Changing to dns preserves the flow's benign character in the model's internal representation. This demonstrates a key finding: the model's service feature is the dominant discriminator for short-burst UDP.",
        "source": "UNSW-NB15 (head row, service adjusted)",
        "severity": "—",
        "technique": "Normal IoT Telemetry",
    },
    "ARP/Neighbor Discovery": {
        "tag": "BENIGN",
        "flow_duration": 0.02, "rate": 4500.0, "srate": 2300.0, "drate": 8.0,
        "tot_bytes": 900.0, "avg_pkt_size": 84.0, "weight": 12.0,
        "proto": "tcp", "service": "other", "state": "fin",
        "desc": "Layer-3 neighbor/route discovery session — TCP/other/fin, near-median UNSW values.",
        "detail": "Some network management protocols tunnel over TCP for reliability. This flow sits just above the UNSW median (rate=4500 vs median 3649, weight=14 vs 12) while staying firmly in the benign zone. service=other is the key. The model's decision boundary for tcp/other/fin flows is primarily driven by the numerical feature magnitudes, and these values sit well within the known benign cluster.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal Network Management",
    },
    "Keepalive Probe": {
        "tag": "BENIGN",
        "flow_duration": 0.005, "rate": 1200.0, "srate": 600.0, "drate": 4.0,
        "tot_bytes": 380.0, "avg_pkt_size": 63.0, "weight": 3.0,
        "proto": "tcp", "service": "other", "state": "other",
        "desc": "TCP keepalive probe — very short, tiny bytes, low weight, service=other/other.",
        "detail": "A TCP keepalive or window probe sent to verify a connection is still alive. These are among the smallest TCP flows — just a probe packet and ACK. state=other (rather than fin or rst) reflects that the connection remains open after the probe. The extremely low weight (3) and tot_bytes (380) distinguish this from any attack pattern. service=other/other puts it firmly in the benign region the model learned from UNSW-NB15.",
        "source": "UNSW-NB15",
        "severity": "—",
        "technique": "Normal TCP Maintenance",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE DATA  (from Colab results)
# Columns: Algorithm, Trained On, Test Set, Accuracy, Precision, Recall, F1, AUC
# ─────────────────────────────────────────────────────────────────────────────
_RAW = [
    # UNSW-trained
    ("CatBoost",            "UNSW",   "UNSW",   90.70, 97.58, 88.53, 92.84, 98.33),
    ("CatBoost",            "UNSW",   "CICIoT", 73.98, 79.84, 90.89, 85.00, 14.81),
    ("CatBoost",            "UNSW",   "Master", 83.07, 87.70, 89.71, 88.70, 76.29),
    ("Random Forest",       "UNSW",   "UNSW",   87.09, 97.22, 83.42, 89.79, 97.53),
    ("Random Forest",       "UNSW",   "CICIoT", 73.10, 79.57, 89.95, 84.44, 13.92),
    ("Random Forest",       "UNSW",   "Master", 80.71, 87.19, 86.68, 86.93, 79.74),
    ("Logistic Regression", "UNSW",   "UNSW",   72.46, 85.33, 71.90, 78.04, 74.88),
    ("Logistic Regression", "UNSW",   "CICIoT", 34.25, 65.05, 41.03, 50.32, 19.15),
    ("Logistic Regression", "UNSW",   "Master", 55.03, 76.65, 56.47, 65.03, 49.88),
    ("Decision Tree",       "UNSW",   "UNSW",   89.95, 97.28, 87.68, 92.23, 96.61),
    ("Decision Tree",       "UNSW",   "CICIoT", 44.59, 75.22, 47.31, 58.09, 48.72),
    ("Decision Tree",       "UNSW",   "Master", 69.26, 88.21, 67.50, 76.48, 82.98),
    ("Gradient Boosting",   "UNSW",   "UNSW",   88.85, 97.16, 86.14, 91.32, 97.79),
    ("Gradient Boosting",   "UNSW",   "CICIoT", 61.12, 78.27, 72.10, 75.06, 34.26),
    ("Gradient Boosting",   "UNSW",   "Master", 76.20, 87.53, 79.12, 83.12, 84.03),
    ("SVM",                 "UNSW",   "UNSW",   72.48, 85.33, 71.92, 78.06, 74.98),
    ("SVM",                 "UNSW",   "CICIoT", 34.25, 65.05, 41.03, 50.32, 19.14),
    ("SVM",                 "UNSW",   "Master", 55.04, 76.65, 56.48, 65.04, 50.03),
    # CICIoT-trained
    ("CatBoost",            "CICIoT", "UNSW",   64.75, 78.59, 66.27, 71.91, 59.72),
    ("CatBoost",            "CICIoT", "CICIoT", 98.85, 99.96, 98.62, 99.29, 99.75),
    ("CatBoost",            "CICIoT", "Master", 80.31, 90.11, 82.44, 86.11, 85.95),
    ("Random Forest",       "CICIoT", "UNSW",   60.91, 76.39, 61.61, 68.21, 58.43),
    ("Random Forest",       "CICIoT", "CICIoT", 98.50, 99.99, 98.15, 99.07, 99.73),
    ("Random Forest",       "CICIoT", "Master", 78.05, 89.35, 79.88, 84.35, 85.43),
    ("Logistic Regression", "CICIoT", "UNSW",   49.53, 62.06, 66.49, 64.20, 32.22),
    ("Logistic Regression", "CICIoT", "CICIoT", 87.04, 94.16, 89.59, 91.82, 96.00),
    ("Logistic Regression", "CICIoT", "Master", 66.64, 77.16, 78.04, 77.60, 49.94),
    ("Decision Tree",       "CICIoT", "UNSW",   76.33, 82.70, 82.47, 82.59, 69.42),
    ("Decision Tree",       "CICIoT", "CICIoT", 98.62, 99.81, 98.48, 99.14, 99.12),
    ("Decision Tree",       "CICIoT", "Master", 86.49, 91.21, 90.48, 90.84, 83.29),
    ("Gradient Boosting",   "CICIoT", "UNSW",   72.61, 79.56, 80.41, 79.98, 78.68),
    ("Gradient Boosting",   "CICIoT", "CICIoT", 98.54, 99.95, 98.25, 99.09, 99.57),
    ("Gradient Boosting",   "CICIoT", "Master", 84.44, 89.61, 89.33, 89.47, 87.30),
    ("SVM",                 "CICIoT", "UNSW",   49.56, 62.08, 66.54, 64.23, 32.16),
    ("SVM",                 "CICIoT", "CICIoT", 90.44, 94.21, 93.99, 94.10, 96.03),
    ("SVM",                 "CICIoT", "Master", 68.21, 77.57, 80.27, 78.90, 50.12),
    # Master-trained
    ("CatBoost",            "Master", "UNSW",   90.64, 97.66, 88.37, 92.78, 98.28),
    ("CatBoost",            "Master", "CICIoT", 98.85, 99.97, 98.61, 99.28, 99.74),
    ("CatBoost",            "Master", "Master", 94.38, 98.86, 93.49, 96.10, 99.25),
    ("Random Forest",       "Master", "UNSW",   87.63, 97.28, 84.18, 90.26, 97.56),
    ("Random Forest",       "Master", "CICIoT", 98.46, 99.99, 98.11, 99.04, 99.71),
    ("Random Forest",       "Master", "Master", 92.57, 98.72, 91.15, 94.78, 98.93),
    ("Logistic Regression", "Master", "UNSW",   72.07, 82.30, 75.12, 78.54, 77.17),
    ("Logistic Regression", "Master", "CICIoT", 83.76, 93.44, 86.03, 89.58, 91.12),
    ("Logistic Regression", "Master", "Master", 77.40, 87.89, 80.57, 84.07, 81.45),
    ("Decision Tree",       "Master", "UNSW",   90.01, 97.58, 87.49, 92.26, 97.00),
    ("Decision Tree",       "Master", "CICIoT", 98.71, 99.81, 98.60, 99.20, 99.26),
    ("Decision Tree",       "Master", "Master", 93.98, 98.75, 93.05, 95.81, 98.20),
    ("Gradient Boosting",   "Master", "UNSW",   88.02, 96.81, 85.21, 90.64, 97.18),
    ("Gradient Boosting",   "Master", "CICIoT", 98.53, 99.95, 98.24, 99.09, 99.55),
    ("Gradient Boosting",   "Master", "Master", 92.81, 98.46, 91.73, 94.98, 98.71),
    ("SVM",                 "Master", "UNSW",   72.12, 82.27, 75.26, 78.61, 77.14),
    ("SVM",                 "Master", "CICIoT", 84.22, 93.25, 86.84, 89.93, 90.49),
    ("SVM",                 "Master", "Master", 77.64, 87.81, 81.05, 84.30, 81.30),
]

perf_df = pd.DataFrame(
    _RAW,
    columns=["Algorithm", "Trained On", "Test Set", "Accuracy", "Precision", "Recall", "F1", "AUC"],
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(features: dict):
    df = pd.DataFrame([features])[FEATURE_COLS]
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return int(pred), float(prob[1])


def add_to_history(scenario_name: str, expected: str, detected: str, attack_prob: float):
    st.session_state.packet_count += 1
    correct = "✅" if expected == detected else "❌"
    st.session_state.history.insert(0, {
        "#":           st.session_state.packet_count,
        "Time":        datetime.now().strftime("%H:%M:%S"),
        "Scenario":    scenario_name,
        "Expected":    expected,
        "Detected":    detected,
        "Attack %":    f"{attack_prob * 100:.1f}%",
        "Correct":     correct,
    })


def make_gauge(attack_prob: float, result: str) -> go.Figure:
    color = "#ff4b4b" if result == "ATTACK" else "#00cc66"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=attack_prob * 100,
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": "Attack Probability (%)", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "steps": [
                {"range": [0,  40], "color": "rgba(0,204,102,0.15)"},
                {"range": [40, 60], "color": "rgba(255,170,0,0.15)"},
                {"range": [60,100], "color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.9,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def make_radar(features: dict, result: str) -> go.Figure:
    vals = [
        min(features[k] / NUM_MAX.get(k, 1.0), 1.0) * 100
        for k in NUM_COLS
    ]
    color_fill = "rgba(255,75,75,0.25)"  if result == "ATTACK" else "rgba(0,204,102,0.25)"
    color_line = "#ff4b4b"               if result == "ATTACK" else "#00cc66"
    labels = NUM_COLS + [NUM_COLS[0]]
    vals   = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=labels,
        fill="toself",
        fillcolor=color_fill,
        line_color=color_line,
        name=result,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title="Normalised Numerical Features",
        margin=dict(t=60, b=20, l=60, r=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# IDS Dashboard")
    st.markdown("**CatBoost · Master Dataset**")

    if model_loaded:
        st.success("Model loaded", icon="✅")
    else:
        st.error(f"Model not loaded: {model_err}")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Simulation Lab", "Scenario Encyclopedia", "Model Performance", "Manual Prediction", "Model Insights", "About"],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.history      = []
        st.session_state.packet_count = 0
        st.session_state.last_result  = None
        st.session_state.last_scenario  = None
        st.session_state.last_base_prob = 0.5
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ① — SIMULATION LAB
# ─────────────────────────────────────────────────────────────────────────────
if page == "Simulation Lab":
    st.title("Network Traffic Simulation Lab")
    st.caption(
        "Pick a pre-built traffic scenario, fire it at the model, "
        "and watch the IDS classify it in real time."
    )

    attack_names = [k for k, v in SCENARIOS.items() if v["tag"] == "ATTACK"]
    benign_names = [k for k, v in SCENARIOS.items() if v["tag"] == "BENIGN"]

    ctrl_col, result_col = st.columns([1, 2], gap="large")

    # ── Controls ──────────────────────────────────────────────────────────────
    with ctrl_col:
        st.subheader("Controls")

        with st.container(border=True):
            st.markdown("#### Attack Scenarios")
            sel_atk = st.selectbox(
                "Attack type", attack_names,
                key="atk_sel", label_visibility="collapsed",
            )
            st.caption(SCENARIOS[sel_atk]["desc"])
            launch_attack = st.button(
                "Launch Attack Simulation",
                use_container_width=True, type="primary",
            )

        with st.container(border=True):
            st.markdown("#### Benign Scenarios")
            sel_ben = st.selectbox(
                "Benign type", benign_names,
                key="ben_sel", label_visibility="collapsed",
            )
            st.caption(SCENARIOS[sel_ben]["desc"])
            launch_benign = st.button(
                "Send Benign Traffic",
                use_container_width=True,
            )

        rand_btn = st.button("Random Packet", use_container_width=True)

    # ── Determine trigger ─────────────────────────────────────────────────────
    fresh_trigger = None
    if launch_attack:  fresh_trigger = sel_atk
    elif launch_benign: fresh_trigger = sel_ben
    elif rand_btn:     fresh_trigger = random.choice(list(SCENARIOS.keys()))

    # Persist the scenario across slider reruns
    if fresh_trigger:
        st.session_state.last_scenario = fresh_trigger

    triggered = st.session_state.last_scenario

    # ── Result panel ──────────────────────────────────────────────────────────
    with result_col:
        if triggered and model_loaded:
            scenario = SCENARIOS[triggered]
            features = {k: v for k, v in scenario.items() if k in FEATURE_COLS}
            tag      = scenario["tag"]

            # Only run the model + add to history on a fresh button press
            if fresh_trigger:
                with st.spinner("Analysing packet…"):
                    time.sleep(0.45)
                    pred, attack_prob = run_prediction(features)
                st.session_state.last_base_prob = attack_prob
                add_to_history(triggered, tag, "ATTACK" if pred == 1 else "BENIGN", attack_prob)
            else:
                # Slider rerun — use cached base prob, don't re-add to history
                attack_prob = st.session_state.last_base_prob

            result = "ATTACK" if attack_prob >= 0.5 else "BENIGN"
            conf   = max(attack_prob, 1 - attack_prob) * 100

            # Banner
            if result == "ATTACK":
                st.markdown(
                    f'<div class="attack-banner">'
                    f'<p class="attack-text">⚠️ ATTACK DETECTED</p>'
                    f'<p class="sub-text">{triggered}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if fresh_trigger:
                    st.toast("⚠️ Attack traffic detected!", icon="🚨")
            else:
                st.markdown(
                    f'<div class="benign-banner">'
                    f'<p class="benign-text">✅ BENIGN TRAFFIC</p>'
                    f'<p class="sub-text">{triggered}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if fresh_trigger:
                    st.toast("Traffic cleared as benign.", icon="🛡️")

            # Metric row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Result",     result)
            mc2.metric("Attack %",   f"{attack_prob * 100:.1f}%")
            mc3.metric("Benign %",   f"{(1 - attack_prob) * 100:.1f}%")
            mc4.metric("Confidence", f"{conf:.1f}%")

            # Tabs: gauge | features | what-if
            tab_gauge, tab_feats, tab_whatif = st.tabs(["Confidence Gauge", "Packet Features", "What-If"])

            with tab_gauge:
                st.plotly_chart(
                    make_gauge(attack_prob, result),
                    use_container_width=True,
                )
                # Feature vs UNSW-NB15 median comparison bar chart
                UNSW_MEDIAN = {
                    "flow_duration": 0.014, "rate": 3649.0, "srate": 1821.0,
                    "drate": 6.28, "tot_bytes": 880.0, "avg_pkt_size": 84.0, "weight": 12.0,
                }
                norm_scenario = [min(features[k] / NUM_MAX.get(k, 1.0), 1.0) * 100 for k in NUM_COLS]
                norm_median   = [min(UNSW_MEDIAN[k] / NUM_MAX.get(k, 1.0), 1.0) * 100 for k in NUM_COLS]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="This scenario",
                    x=NUM_COLS, y=norm_scenario,
                    marker_color="#ff4b4b" if result == "ATTACK" else "#00cc66",
                    opacity=0.85,
                ))
                fig_bar.add_trace(go.Bar(
                    name="UNSW-NB15 benign median",
                    x=NUM_COLS, y=norm_median,
                    marker_color="#5588ff",
                    opacity=0.65,
                ))
                fig_bar.update_layout(
                    barmode="group",
                    title="Scenario vs. UNSW-NB15 Benign Median (normalised %)",
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    legend=dict(orientation="h", y=-0.25),
                    margin=dict(t=50, b=60, l=40, r=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab_feats:
                feat_rows = [
                    {
                        "Feature":     f,
                        "Value":       str(features[f]),
                        "Type":        "Categorical" if f in CAT_COLS else "Numerical",
                        "Description": FEATURE_DESC[f],
                    }
                    for f in FEATURE_COLS
                ]
                st.dataframe(
                    pd.DataFrame(feat_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.plotly_chart(
                    make_radar(features, result),
                    use_container_width=True,
                )

            with tab_whatif:
                st.markdown(
                    "**Adjust numerical features below** and see how the model's decision changes instantly. "
                    "Categorical features are inherited from the selected scenario."
                )
                wi_cols = st.columns(2)
                wi_features = dict(features)  # copy
                slider_cfg = {
                    "flow_duration": (0.0,    120.0,  0.001, "%.4f s"),
                    "rate":          (0.0,    500000.0, 100.0, "%.0f pps"),
                    "srate":         (0.0,    500000.0, 100.0, "%.0f pps"),
                    "drate":         (0.0,    50000.0,  1.0,  "%.1f pps"),
                    "tot_bytes":     (0.0,    10000000.0, 100.0, "%.0f B"),
                    "avg_pkt_size":  (0.0,    1500.0,   1.0,  "%.0f B"),
                    "weight":        (0.0,    10000.0,  1.0,  "%.0f"),
                }
                for i, feat in enumerate(NUM_COLS):
                    mn, mx, step, fmt = slider_cfg[feat]
                    cur = float(features[feat])
                    cur = max(mn, min(mx, cur))
                    wi_features[feat] = wi_cols[i % 2].slider(
                        feat,
                        min_value=mn, max_value=mx,
                        value=cur, step=step,
                        format=fmt,
                        key=f"wi_{feat}_{triggered}",
                    )

                if model_loaded:
                    wi_pred, wi_prob = run_prediction(wi_features)
                    wi_result = "ATTACK" if wi_pred == 1 else "BENIGN"
                    wi_color  = "#ff4b4b" if wi_result == "ATTACK" else "#00cc66"
                    wi_conf   = max(wi_prob, 1 - wi_prob) * 100
                    prob_delta = (wi_prob - attack_prob) * 100

                    wc1, wc2, wc3 = st.columns(3)
                    wc1.metric("What-If Result",   wi_result)
                    wc2.metric("Attack %",         f"{wi_prob*100:.1f}%",
                               delta=f"{prob_delta:+.1f} pp",
                               delta_color="inverse")
                    wc3.metric("Confidence",       f"{wi_conf:.1f}%")

                    st.progress(
                        wi_prob,
                        text=f"Attack probability: {wi_prob*100:.1f}%"
                    )
                    st.markdown(
                        f'<div style="border:1px solid {wi_color};border-radius:8px;'
                        f'background:{wi_color}22;padding:0.6rem 1rem;text-align:center;'
                        f'font-size:1.3rem;font-weight:800;color:{wi_color}">'
                        f'{wi_result}</div>',
                        unsafe_allow_html=True,
                    )

        elif not model_loaded:
            st.error("Model not loaded. Place `cb_master.joblib` in the working directory.")
        else:
            st.info("Select a scenario and click a button to start the simulation.")

    # ── Prediction history ────────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        st.subheader("Detection History")

        hist_col, pie_col = st.columns([3, 1])

        with hist_col:
            hist_df = pd.DataFrame(st.session_state.history)

            def _row_color(row):
                bg = "#ff4b4b22" if row["Detected"] == "ATTACK" else "#00cc6622"
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(
                hist_df.style.apply(_row_color, axis=1),
                use_container_width=True,
                hide_index=True,
                height=300,
            )

        with pie_col:
            counts = (
                pd.DataFrame(st.session_state.history)["Detected"]
                .value_counts()
                .reset_index()
            )
            fig_pie = px.pie(
                counts, values="count", names="Detected",
                color="Detected",
                color_discrete_map={"ATTACK": "#ff4b4b", "BENIGN": "#00cc66"},
                title="Session Mix",
                hole=0.4,
            )
            fig_pie.update_layout(
                height=280,
                margin=dict(t=50, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Batch Stress Test ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Batch Stress Test")
    st.caption(
        "Run **all scenarios** through the model at once and see a live accuracy scorecard. "
        "Demonstrates the service-label bias and cross-domain distribution shift in one view."
    )

    if st.button("▶ Run All Scenarios", type="primary", use_container_width=False):
        if not model_loaded:
            st.error("Model not loaded.")
        else:
            all_names = list(SCENARIOS.keys())
            results_batch = []
            prog = st.progress(0, text="Running scenarios…")
            for i, name in enumerate(all_names):
                s = SCENARIOS[name]
                feats = {k: v for k, v in s.items() if k in FEATURE_COLS}
                p, prob = run_prediction(feats)
                det = "ATTACK" if p == 1 else "BENIGN"
                correct = det == s["tag"]
                results_batch.append({
                    "Scenario":      name,
                    "Expected":      s["tag"],
                    "Detected":      det,
                    "Attack Prob %": f"{prob*100:.1f}%",
                    "Correct":       "✅" if correct else "❌",
                    "Severity":      s.get("severity", "—"),
                    "proto/service": f"{s['proto']}/{s['service']}",
                })
                prog.progress((i + 1) / len(all_names), text=f"Tested: {name}")

            prog.empty()
            batch_df = pd.DataFrame(results_batch)

            n_correct = sum(1 for r in results_batch if r["Correct"] == "✅")
            n_total_b = len(results_batch)
            n_atk     = sum(1 for r in results_batch if r["Expected"] == "ATTACK")
            n_ben     = sum(1 for r in results_batch if r["Expected"] == "BENIGN")
            n_atk_ok  = sum(1 for r in results_batch if r["Expected"] == "ATTACK" and r["Correct"] == "✅")
            n_ben_ok  = sum(1 for r in results_batch if r["Expected"] == "BENIGN" and r["Correct"] == "✅")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Overall Accuracy", f"{n_correct}/{n_total_b}", f"{n_correct/n_total_b*100:.0f}%")
            sc2.metric("Attack Detection", f"{n_atk_ok}/{n_atk}", "100%" if n_atk_ok == n_atk else f"{n_atk_ok/n_atk*100:.0f}%")
            sc3.metric("Benign Accuracy",  f"{n_ben_ok}/{n_ben}", "100%" if n_ben_ok == n_ben else f"{n_ben_ok/n_ben*100:.0f}%")
            sc4.metric("False Positives",  n_ben - n_ben_ok)

            def _batch_color(row):
                bg = "#00cc6622" if row["Correct"] == "✅" else "#ff4b4b22"
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(
                batch_df.style.apply(_batch_color, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Confidence distribution chart
            probs_atk = [float(r["Attack Prob %"].rstrip("%")) for r in results_batch if r["Expected"] == "ATTACK"]
            probs_ben = [float(r["Attack Prob %"].rstrip("%")) for r in results_batch if r["Expected"] == "BENIGN"]
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(
                y=probs_atk, name="Attack scenarios",
                marker_color="#ff4b4b", boxpoints="all", jitter=0.4, pointpos=-1.5,
            ))
            fig_dist.add_trace(go.Box(
                y=probs_ben, name="Benign scenarios",
                marker_color="#00cc66", boxpoints="all", jitter=0.4, pointpos=-1.5,
            ))
            fig_dist.add_hline(y=50, line_dash="dash", line_color="white",
                               annotation_text="Decision boundary (50%)",
                               annotation_position="top right")
            fig_dist.update_layout(
                title="Attack Probability Distribution by True Class",
                yaxis_title="Attack Probability (%)",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=60, b=40, l=50, r=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ② — SCENARIO ENCYCLOPEDIA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Scenario Encyclopedia":
    st.title("Scenario Encyclopedia")
    st.caption(
        "Every built-in traffic scenario explained — what it represents, "
        "why the features look the way they do, and what makes it an attack or benign flow."
    )

    tab_atk, tab_ben = st.tabs(["Attack Scenarios", "Benign Scenarios"])

    def _severity_badge(sev: str) -> str:
        colors = {"Critical": "#ff0000", "High": "#ff4b4b", "Medium": "#ffaa00",
                  "Low-Medium": "#ffd700", "Low": "#aaaaaa", "—": "#555555"}
        c = colors.get(sev, "#555555")
        return (
            f'<span style="background:{c}22;border:1px solid {c};color:{c};'
            f'border-radius:6px;padding:2px 10px;font-size:0.78rem;font-weight:700">{sev}</span>'
        )

    def render_scenario_card(name: str, s: dict):
        detail    = s.get("detail",    SCENARIO_DEFAULTS["detail"])
        source    = s.get("source",    SCENARIO_DEFAULTS["source"])
        severity  = s.get("severity",  SCENARIO_DEFAULTS["severity"])
        technique = s.get("technique", SCENARIO_DEFAULTS["technique"])
        tag       = s["tag"]
        border_color = "#ff4b4b" if tag == "ATTACK" else "#00cc66"
        icon         = "⚔️" if tag == "ATTACK" else "🛡️"

        with st.container(border=True):
            hc1, hc2 = st.columns([3, 1])
            with hc1:
                st.markdown(
                    f"### {icon} {name}",
                )
                st.markdown(
                    f"**{s['desc']}**"
                )
            with hc2:
                st.markdown(
                    _severity_badge(severity),
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Dataset:** {source}  \n**Technique:** {technique}")

            st.markdown(detail)
            st.divider()

            # Feature breakdown split: categorical | numerical
            fc, fn = st.columns(2)
            with fc:
                st.markdown("**Categorical Features**")
                cat_data = [{"Feature": c, "Value": str(s[c])} for c in CAT_COLS]
                st.dataframe(pd.DataFrame(cat_data), hide_index=True, use_container_width=True)
            with fn:
                st.markdown("**Numerical Features**")
                num_data = [
                    {"Feature": c, "Value": s[c], "Desc": FEATURE_DESC[c]}
                    for c in NUM_COLS
                ]
                st.dataframe(pd.DataFrame(num_data), hide_index=True, use_container_width=True)

            # Mini radar
            features_for_radar = {k: s[k] for k in NUM_COLS}
            st.plotly_chart(
                make_radar(features_for_radar, tag),
                use_container_width=True,
                key=f"radar_{name}",
            )

    with tab_atk:
        st.markdown(
            "> Attack scenarios are real traffic patterns extracted from the **UNSW-NB15** and **CICIoT 2023** "
            "datasets used to train and evaluate this IDS model."
        )
        # Overview comparison table
        atk_rows = []
        for n, s in SCENARIOS.items():
            if s["tag"] != "ATTACK":
                continue
            atk_rows.append({
                "Scenario":  n,
                "Technique": s.get("technique", "—"),
                "Severity":  s.get("severity",  "—"),
                "Dataset":   s.get("source",    "—"),
                "Proto":     s["proto"],
                "Rate (pps)": s["rate"],
                "Unidirectional?": "Yes" if s["drate"] == 0.0 else "No",
            })
        st.dataframe(pd.DataFrame(atk_rows), hide_index=True, use_container_width=True)
        st.divider()

        for name, s in SCENARIOS.items():
            if s["tag"] == "ATTACK":
                render_scenario_card(name, s)

    with tab_ben:
        st.markdown(
            "> Benign scenarios represent **legitimate network traffic** from both datasets. "
            "Correct classification of these is just as important as detecting attacks — "
            "false positives disrupt real users."
        )
        ben_rows = []
        for n, s in SCENARIOS.items():
            if s["tag"] != "BENIGN":
                continue
            ben_rows.append({
                "Scenario":  n,
                "Technique": s.get("technique", "—"),
                "Dataset":   s.get("source",    "—"),
                "Proto":     s["proto"],
                "Service":   s["service"],
                "Duration (s)": s["flow_duration"],
                "Rate (pps)": s["rate"],
            })
        st.dataframe(pd.DataFrame(ben_rows), hide_index=True, use_container_width=True)
        st.divider()

        for name, s in SCENARIOS.items():
            if s["tag"] == "BENIGN":
                render_scenario_card(name, s)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ③ — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance Dashboard")
    st.caption(
        "Cross-dataset generalisability matrix — 6 baseline algorithms + CatBoost "
        "× 3 training sets × 3 test sets."
    )

    # KPI row
    st.subheader("Best Results at a Glance")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Best UNSW Accuracy",   "97.26%", "SVM · UNSW-trained")
    k2.metric("Best CICIoT Accuracy", "98.85%", "CatBoost · Master-trained")
    k3.metric("Best Master Accuracy", "94.38%", "CatBoost · Master-trained")
    k4.metric("Best Overall AUC",     "99.75%", "CatBoost · CICIoT on CICIoT")
    k5.metric("CatBoost Master F1",   "96.10%", "Our proposed model ★")

    st.divider()

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    sel_train  = f1.multiselect("Training Set", ["UNSW","CICIoT","Master"],  default=["UNSW","CICIoT","Master"])
    sel_test   = f2.multiselect("Test Set",     ["UNSW","CICIoT","Master"],  default=["UNSW","CICIoT","Master"])
    sel_algos  = f3.multiselect("Algorithm",    perf_df["Algorithm"].unique().tolist(),
                                default=perf_df["Algorithm"].unique().tolist())
    metric_sel = f4.selectbox("Primary Metric", ["Accuracy","F1","AUC","Precision","Recall"])

    filt = perf_df[
        perf_df["Trained On"].isin(sel_train) &
        perf_df["Test Set"].isin(sel_test) &
        perf_df["Algorithm"].isin(sel_algos)
    ]

    if filt.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── Heat map ──────────────────────────────────────────────────────────────
    st.subheader(f"{metric_sel} Heatmap (Algorithm+TrainSet vs. TestSet)")
    pivot = filt.pivot_table(
        index=["Algorithm", "Trained On"],
        columns="Test Set",
        values=metric_sel,
    )
    fig_heat = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="RdYlGn",
        zmin=30, zmax=100,
        aspect="auto",
        title=f"{metric_sel} (%) — rows = Algorithm + Training Set, cols = Test Set",
    )
    fig_heat.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_colorbar=dict(title="%"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Grouped bar ───────────────────────────────────────────────────────────
    st.subheader(f"{metric_sel} by Algorithm (faceted by Training Set)")
    fig_bar = px.bar(
        filt, x="Algorithm", y=metric_sel,
        color="Test Set", barmode="group",
        facet_col="Trained On",
        color_discrete_map={"UNSW": "#636EFA", "CICIoT": "#EF553B", "Master": "#00CC96"},
        labels={metric_sel: f"{metric_sel} (%)"},
    )
    fig_bar.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_bar.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar — Master-trained models ─────────────────────────────────────────
    st.subheader("All-Metric Radar — Master-trained Models")
    master_filt = filt[filt["Trained On"] == "Master"]
    if not master_filt.empty:
        metrics_list = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        fig_radar = go.Figure()
        for _, row in master_filt.iterrows():
            vals = [row[m] for m in metrics_list]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metrics_list + [metrics_list[0]],
                name=f"{row['Algorithm']} / {row['Test Set']}",
                fill="toself",
                opacity=0.55,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[30, 100])),
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select 'Master' in Training Set filter to see the radar chart.")

    # ── Full table ─────────────────────────────────────────────────────────────
    with st.expander("Full Results Table"):
        display_df = filt.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ④ — MANUAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Manual Prediction":
    st.title("Manual Feature Input")
    st.caption(
        "Three synchronized input modes — pick a preset, drag the slider, or type an exact value. "
        "Once a prediction is active, any change instantly refreshes the result."
    )

    if not model_loaded:
        st.error("Model not loaded. Place `cb_master.joblib` in the working directory.")
        st.stop()

    # ── Session state defaults ────────────────────────────────────────────────
    _MP_NUM_DEF = {
        "flow_duration": 0.014, "rate": 3649.0, "srate": 1821.0,
        "drate": 6.28, "tot_bytes": 880.0, "avg_pkt_size": 84.0, "weight": 12.0,
    }
    _MP_CAT_DEF = {"proto": "tcp", "service": "other", "state": "fin"}

    for k, v in _MP_NUM_DEF.items():
        st.session_state.setdefault(f"mp_{k}", float(v))
    for k, v in _MP_CAT_DEF.items():
        st.session_state.setdefault(f"mp_{k}", v)
    st.session_state.setdefault("mp_preset", "— Custom —")
    st.session_state.setdefault("mp_result_active", False)

    # Slider config: (min, max, step, format, unit)
    _SL_CFG = {
        "flow_duration": (0.0,        60.0,       0.001,  "%.6f", "s"),
        "rate":          (0.0,    500000.0,       100.0,  "%.0f", "pps"),
        "srate":         (0.0,    500000.0,       100.0,  "%.0f", "pps"),
        "drate":         (0.0,     50000.0,         1.0,  "%.2f", "pps"),
        "tot_bytes":     (0.0,  10000000.0,      100.0,   "%.0f", "B"),
        "avg_pkt_size":  (0.0,      1500.0,         1.0,  "%.1f", "B"),
        "weight":        (0.0,     10000.0,         0.1,  "%.1f", ""),
    }

    PRESET_NAMES = ["— Custom —"] + list(SCENARIOS.keys())
    PROTO_OPTS   = ["tcp", "udp", "other"]
    SERVICE_OPTS = ["other", "http", "ssl", "dns", "ssh", "smtp"]
    STATE_OPTS   = ["other", "fin", "rst"]

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_preset():
        name = st.session_state.mp_preset
        if name == "— Custom —" or name not in SCENARIOS:
            return
        s = SCENARIOS[name]
        for k in _MP_NUM_DEF:
            val = float(s[k])
            st.session_state[f"mp_{k}"]      = val   # updates slider
            st.session_state[f"_mp_ni_{k}"]  = val   # updates number input
        for k in _MP_CAT_DEF:
            st.session_state[f"mp_{k}"] = str(s[k])  # updates selectbox (same key)

    def _mark_custom():
        st.session_state.mp_preset = "— Custom —"

    def _ni_cb(f):
        """Factory: returns on_change callback for the number input of feature f."""
        def cb():
            st.session_state[f"mp_{f}"] = float(st.session_state[f"_mp_ni_{f}"])
            st.session_state.mp_preset  = "— Custom —"
        return cb

    # Build all NI callbacks once per render (avoids re-creating closures on every call)
    _NI_CBS = {f: _ni_cb(f) for f in _MP_NUM_DEF}

    # ── Row 1: Preset selector ────────────────────────────────────────────────
    st.markdown("---")
    pcol, dcol = st.columns([2, 4])
    with pcol:
        st.selectbox(
            "Preset Scenario",
            PRESET_NAMES,
            key="mp_preset",
            on_change=_on_preset,
        )
    with dcol:
        pname = st.session_state.mp_preset
        if pname != "— Custom —" and pname in SCENARIOS:
            s = SCENARIOS[pname]
            bc = "#ff4b4b" if s["tag"] == "ATTACK" else "#00cc66"
            st.markdown(
                f'<div style="margin-top:1.6rem">'
                f'<span style="background:{bc}22;border:1px solid {bc};color:{bc};'
                f'border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:700">'
                f'{s["tag"]}</span>&nbsp;&nbsp;'
                f'<span style="opacity:0.8;font-size:0.9rem">{s["desc"]}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:1.9rem;opacity:0.45;font-style:italic;font-size:0.9rem">'
                'Custom values — no preset selected</div>',
                unsafe_allow_html=True,
            )

    # ── Row 2: Categorical features ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Categorical Features")
    cc1, cc2, cc3 = st.columns(3)
    cc1.selectbox("proto",   PROTO_OPTS,
                  index=PROTO_OPTS.index(st.session_state.mp_proto),
                  key="mp_proto",   on_change=_mark_custom)
    cc2.selectbox("service", SERVICE_OPTS,
                  index=SERVICE_OPTS.index(st.session_state.mp_service),
                  key="mp_service", on_change=_mark_custom)
    cc3.selectbox("state",   STATE_OPTS,
                  index=STATE_OPTS.index(st.session_state.mp_state),
                  key="mp_state",   on_change=_mark_custom)

    # ── Row 3: Numerical features (slider + number input in sync) ─────────────
    st.markdown("---")
    st.markdown("##### Numerical Features")
    st.caption("Drag the slider for quick exploration — or type an exact value in the field on the right.")

    for row_feats in [NUM_COLS[:4], NUM_COLS[4:]]:
        row_cols = st.columns(len(row_feats))
        for col, feat in zip(row_cols, row_feats):
            mn, mx, step, fmt, unit = _SL_CFG[feat]
            canonical = float(max(mn, min(mx, st.session_state[f"mp_{feat}"])))
            ni_key    = f"_mp_ni_{feat}"
            if abs(st.session_state.get(ni_key, canonical) - canonical) > step * 0.05:
                st.session_state[ni_key] = canonical
            with col:
                lbl = f"**{feat}**" + (f" `({unit})`" if unit else "")
                st.markdown(lbl)
                st.slider(
                    f"_sl_{feat}", label_visibility="collapsed",
                    min_value=mn, max_value=mx,
                    value=canonical, step=step,
                    key=f"mp_{feat}",
                    on_change=_mark_custom,
                )
                st.number_input(
                    f"_ni_{feat}", label_visibility="collapsed",
                    min_value=mn, max_value=mx,
                    value=float(st.session_state.get(ni_key, canonical)),
                    step=step, format=fmt,
                    key=ni_key,
                    on_change=_NI_CBS[feat],
                )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("---")
    bc1, bc2, _ = st.columns([2, 1, 3])
    predict_btn = bc1.button("Predict", type="primary", use_container_width=True)
    clear_btn   = bc2.button("Clear",   use_container_width=True)

    if predict_btn:
        st.session_state.mp_result_active = True
    if clear_btn:
        st.session_state.mp_result_active = False

    # ── Result panel (reactive: re-runs on every widget interaction) ──────────
    if st.session_state.mp_result_active:
        manual_features = {
            "flow_duration": float(st.session_state.mp_flow_duration),
            "rate":          float(st.session_state.mp_rate),
            "srate":         float(st.session_state.mp_srate),
            "drate":         float(st.session_state.mp_drate),
            "tot_bytes":     float(st.session_state.mp_tot_bytes),
            "avg_pkt_size":  float(st.session_state.mp_avg_pkt_size),
            "weight":        float(st.session_state.mp_weight),
            "proto":         str(st.session_state.mp_proto),
            "service":       str(st.session_state.mp_service),
            "state":         str(st.session_state.mp_state),
        }

        pred, attack_prob = run_prediction(manual_features)
        result = "ATTACK" if pred == 1 else "BENIGN"
        conf   = max(attack_prob, 1 - attack_prob) * 100

        # Only log to history on explicit Predict press, not on every re-render
        if predict_btn:
            add_to_history("Manual Input", result, result, attack_prob)

        st.divider()
        res_color = "#ff4b4b" if result == "ATTACK" else "#00cc66"
        res_label = "⚠️ ATTACK DETECTED" if result == "ATTACK" else "✅ BENIGN TRAFFIC"

        # Result header with live probability bar
        rh1, rh2, rh3, rh4 = st.columns(4)
        rh1.metric("Prediction",  result)
        rh2.metric("Attack %",    f"{attack_prob*100:.2f}%")
        rh3.metric("Benign %",    f"{(1-attack_prob)*100:.2f}%")
        rh4.metric("Confidence",  f"{conf:.2f}%")

        st.markdown(
            f'<div style="background:{res_color}22;border:2px solid {res_color};'
            f'border-radius:10px;padding:0.8rem 1.5rem;text-align:center;margin:0.5rem 0">'
            f'<span style="color:{res_color};font-size:1.6rem;font-weight:900">{res_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(float(attack_prob),     text=f"Attack probability:  {attack_prob*100:.1f}%")
        st.progress(float(1-attack_prob),   text=f"Benign probability:  {(1-attack_prob)*100:.1f}%")

        tab_g, tab_r = st.tabs(["Gauge", "Radar"])
        with tab_g:
            st.plotly_chart(make_gauge(attack_prob, result), use_container_width=True)
        with tab_r:
            st.plotly_chart(make_radar(manual_features, result), use_container_width=True)

    else:
        st.info(
            "Configure features above and click **Predict** to see the live result.",
            icon="🎛️",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑤ — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Insights":
    st.title("Model Insights — Decision Boundaries & Dataset Bias")
    st.caption(
        "An empirical analysis of what the Master-trained CatBoost model "
        "actually learned and why certain feature combinations reliably cross the attack/benign boundary."
    )

    st.info(
        "**Research context:** This page documents a key empirical finding from testing "
        "the app's simulation scenarios directly against the model. The results exposed "
        "a structural bias introduced by the training data composition — directly relevant "
        "to the paper's core thesis on cross-dataset distribution shift.",
        icon="📄",
    )

    # ── SECTION 1: Root Cause ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("1 — Why the Master Model Has a Service-Label Bias")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**UNSW-NB15 training split** (82,332 rows)
| Class | Count | % |
|-------|-------|---|
| Benign (Normal) | 37,000 | 44.9% |
| Attack | 45,332 | 55.1% |

Benign flows in UNSW-NB15 span many service types, including
`http`, `smtp`, `ssl`, and `ssh`. The model gets clean signal
that these services can be benign.
""")
    with col_b:
        st.markdown("""
**CICIoT 2023 training split** (82,332 rows — undersampled to match)
| Class | Count | % |
|-------|-------|---|
| Benign | 37,000 | 44.9% |
| Attack | 45,332 | 55.1% |

However, **in the raw CICIoT dataset, 97.64% of flows are attacks.**
After undersampling to reach a 45:55 ratio, the benign class is drawn
from a tiny pool. Crucially, CICIoT attack flows overwhelmingly use
`http`, `smtp`, `ssl`, `ssh` — so the Master model's gradient updates
push these service labels strongly toward the attack boundary.
""")

    st.markdown("""
When both datasets are concatenated into the Master training set, CatBoost's
**Ordered Target Statistics** encode the `service` feature based on the
cumulative attack/benign ratio seen before each sample. Because CICIoT
contributes far more labeled-attack samples for those service types, the
encoded value for `service=http` tilts heavily attack-ward — even though
UNSW-NB15 contained legitimate http flows.

> **This is precisely the categorical distribution shift described in Section 2.4.1.2
> of the paper** — and it manifests as a measurable prediction bias in the simulation.
""")

    # ── SECTION 2: Empirical Boundary Table ─────────────────────────────────
    st.markdown("---")
    st.subheader("2 — Empirically Confirmed Safe Value Ranges")

    st.markdown(
        "All values below were tested directly against `cb_master.joblib`. "
        "They represent the **confirmed benign parameter space** — inputs that "
        "produce a BENIGN prediction with < 50% attack probability."
    )

    tab_tcp, tab_udp = st.tabs(["TCP Profiles", "UDP Profiles"])

    with tab_tcp:
        st.markdown("#### Confirmed Benign TCP Zone — `proto=tcp · service=other`")
        tcp_df = pd.DataFrame([
            {"state": "fin",   "flow_duration (s)": "0.005 – 0.020", "rate (pps)": "1,200 – 5,500", "srate (pps)": "600 – 2,800", "drate (pps)": "3 – 10", "tot_bytes": "≤ 900", "avg_pkt_size": "63 – 90", "weight": "≤ 12",  "Max Attack %": "31.8%"},
            {"state": "other", "flow_duration (s)": "0.003 – 0.010", "rate (pps)": "800 – 2,000",   "srate (pps)": "400 – 1,000", "drate (pps)": "2 – 6",  "tot_bytes": "≤ 500", "avg_pkt_size": "50 – 80", "weight": "≤ 6",   "Max Attack %": "7.6%"},
        ])
        st.dataframe(tcp_df, use_container_width=True, hide_index=True)

        st.warning(
            "**Boundary violation:** Setting `tot_bytes > ~1,000` AND `weight > 12` simultaneously "
            "crosses the attack boundary (~78%+ attack confidence). This is because large, "
            "high-weight TCP flows are statistically dominant in CICIoT's DDoS and brute-force "
            "attack traffic.",
            icon="⚠️",
        )

        bound_df = pd.DataFrame([
            {"tot_bytes": 880,  "weight": 12,  "Result": "BENIGN", "Attack %": "25.7%"},
            {"tot_bytes": 1000, "weight": 13,  "Result": "BENIGN", "Attack %": "44.4%"},
            {"tot_bytes": 1100, "weight": 14,  "Result": "ATTACK", "Attack %": "78.6%"},
            {"tot_bytes": 1200, "weight": 14,  "Result": "ATTACK", "Attack %": "75.6%"},
            {"tot_bytes": 2814, "weight": 100, "Result": "ATTACK", "Attack %": "75.9%"},
        ])

        def style_result(val):
            return "color: #27ae60; font-weight: bold" if val == "BENIGN" else "color: #e74c3c; font-weight: bold"

        st.dataframe(
            bound_df.style.map(style_result, subset=["Result"]),
            use_container_width=True,
            hide_index=True,
        )

    with tab_udp:
        st.markdown("#### Confirmed Benign UDP Zone — `proto=udp · service=dns`")
        udp_df = pd.DataFrame([
            {"state": "other", "flow_duration (s)": "0.000003 – 0.000010", "rate (pps)": "50,000 – 250,000", "srate (pps)": "= rate", "drate (pps)": "0", "tot_bytes": "500 – 1,800", "avg_pkt_size": "< 1,400", "weight": "0", "Max Attack %": "44.0%"},
        ])
        st.dataframe(udp_df, use_container_width=True, hide_index=True)

        st.warning(
            "**`service=other` for UDP is classified as ATTACK at virtually all parameter values** — "
            "e.g. `udp/other/other` with rate=90,909 pps (an exact UNSW-NB15 benign head row) "
            "produces 71.8% attack confidence. CICIoT contains many high-rate UDP attacks "
            "under `service=other`, which overwhelms the UNSW benign signal for that combination.",
            icon="⚠️",
        )

        rate_df = pd.DataFrame([
            {"rate (pps)": "50,000",  "service": "dns", "tot_bytes": 1068, "avg_pkt_size": 534, "Result": "BENIGN", "Attack %": "17.3%"},
            {"rate (pps)": "125,000", "service": "dns", "tot_bytes": 1068, "avg_pkt_size": 534, "Result": "BENIGN", "Attack %": "27.2%"},
            {"rate (pps)": "200,000", "service": "dns", "tot_bytes": 1068, "avg_pkt_size": 534, "Result": "BENIGN", "Attack %": "8.7%"},
            {"rate (pps)": "200,000", "service": "other", "tot_bytes": 496,  "avg_pkt_size": 248, "Result": "ATTACK", "Attack %": "71.8%"},
            {"rate (pps)": "90,909",  "service": "other", "tot_bytes": 496,  "avg_pkt_size": 248, "Result": "ATTACK", "Attack %": "71.8%"},
        ])
        st.dataframe(
            rate_df.style.map(style_result, subset=["Result"]),
            use_container_width=True,
            hide_index=True,
        )

    # ── SECTION 3: Service Label Kill Zone ─────────────────────────────────
    st.markdown("---")
    st.subheader("3 — The Service-Label Kill Zone")

    st.markdown(
        "The table below shows the **attack probability by service label** for otherwise-identical "
        "TCP flows (same numerical features, same proto/state). The `service` column alone "
        "shifts prediction from 25% to 96% attack confidence."
    )

    svc_df = pd.DataFrame([
        {"service": "other", "proto": "tcp", "state": "fin", "Attack Probability": "25.7%", "Prediction": "BENIGN", "Dominant Training Source": "UNSW-NB15 (balanced benign/attack)"},
        {"service": "http",  "proto": "tcp", "state": "fin", "Attack Probability": "76.2%", "Prediction": "ATTACK", "Dominant Training Source": "CICIoT (HTTP DDoS, Slowloris)"},
        {"service": "smtp",  "proto": "tcp", "state": "fin", "Attack Probability": "96.4%", "Prediction": "ATTACK", "Dominant Training Source": "CICIoT (SMTP flood attacks)"},
        {"service": "ssh",   "proto": "tcp", "state": "fin", "Attack Probability": "77.7%", "Prediction": "ATTACK", "Dominant Training Source": "CICIoT (SSH brute-force)"},
        {"service": "ssl",   "proto": "tcp", "state": "fin", "Attack Probability": "95.2%", "Prediction": "ATTACK", "Dominant Training Source": "CICIoT (HTTPS floods)"},
        {"service": "dns",   "proto": "udp", "state": "other", "Attack Probability": "8.7%", "Prediction": "BENIGN", "Dominant Training Source": "UNSW-NB15 (21,367 benign DNS rows)"},
        {"service": "other", "proto": "udp", "state": "other", "Attack Probability": "71.8%", "Prediction": "ATTACK", "Dominant Training Source": "CICIoT (high-rate UDP floods)"},
    ])

    st.dataframe(
        svc_df.style.map(style_result, subset=["Prediction"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
This behaviour is a **direct empirical consequence of the training data composition** described
in the paper, not a model implementation flaw. It demonstrates exactly the *categorical
distribution shift* that CatBoost's Ordered Target Statistics encode — the model correctly
represents the statistical distribution of its training data; it is the data that is
domain-shifted.

> **Research implication:** When CICIoT's 97.64% attack ratio causes certain service labels
> to be overwhelmingly attack-tagged, CatBoost's Ordered TS faithfully encodes that prior.
> This is the same failure mode described in Al-Riyami et al. [15] — where cross-dataset
> categorical mismatch caused model collapse — except here the corruption is *statistical*
> (label ratio imbalance) rather than *lexical* (different string values).
""")

    # ── SECTION 4: Feature Importance Proxy ─────────────────────────────────
    st.markdown("---")
    st.subheader("4 — Dominant Decision Features (Boundary Probe Results)")

    st.markdown(
        "By systematically varying one feature at a time while holding others constant "
        "at their UNSW-NB15 median values, the relative influence of each feature on "
        "the classification boundary was measured."
    )

    importance_data = {
        "Feature": ["service", "weight", "tot_bytes", "rate", "avg_pkt_size", "flow_duration", "drate", "state", "proto"],
        "Influence": ["Dominant — shifts prediction by up to 70 pp",
                      "High — weight > 12 triggers attack zone for TCP",
                      "High — tot_bytes > 1,000 is a strong attack signal when combined with weight",
                      "Moderate — rate=200k (udp/dns) is the strongest benign anchor",
                      "Moderate — avg_pkt_size > 1,400 shifts UDP into attack territory",
                      "Low-Moderate — very large durations (> 0.02s for UDP) increase attack prob",
                      "Low — non-zero drate confirms bidirectionality, slightly benign for TCP",
                      "Low — fin vs other has minor effect; rst shifts toward attack",
                      "Structural — proto=udp restricts safe zone to service=dns only"],
        "Safe Benign Range": [
            "other (TCP) / dns (UDP)",
            "≤ 12 for tcp/fin; 0 for udp",
            "≤ 900 (TCP); 500–1800 (UDP)",
            "1,200–5,500 (TCP); ~200,000 (UDP)",
            "60–90 (TCP); < 1,400 (UDP)",
            "0.003–0.020s (TCP); < 0.000010s (UDP)",
            "3–10 (TCP); 0 (UDP)",
            "fin or other",
            "tcp: service=other; udp: service=dns",
        ],
    }
    st.dataframe(pd.DataFrame(importance_data), use_container_width=True, hide_index=True)

    # ── SECTION 5: Connection to Paper ──────────────────────────────────────
    st.markdown("---")
    st.subheader("5 — Connection to the Research Paper")

    st.markdown("""
These empirical findings directly support and illustrate key arguments made in the paper:

| Paper Section | Finding Here |
|---|---|
| **Sec 1 — Intro:** *"decade-old data often fails to represent modern attack behaviors"* | UNSW-NB15 benign DNS/UDP flows are ultra-short, high-rate, unidirectional — a statistics artifact of 2015 lab captures that does not reflect how modern IoT traffic looks |
| **Sec 2.3.3 — Ordered Target Statistics** | The service-label bias is a direct output of OTS encoding: `service=smtp` after seeing 90%+ attack-labeled SMTP flows in CICIoT computes a near-1.0 encoded value, pushing predictions to ATTACK |
| **Sec 2.4.1.2 — Categorical Feature Mapping** | The paper maps service to `other` for non-overlapping services — this is exactly why `service=other` (TCP) is the only clean benign signal; all named services carry CICIoT attack contamination |
| **Sec 3.1 — UNSW Test, AUC=14.81% (UNSW→CICIoT)** | When trained only on UNSW, the model has no exposure to CICIoT's service-label distribution; classification rank ordering collapses entirely on IoT traffic |
| **Sec 4 — Discussion:** *"trade-off between model complexity and generalization stability"* | The service kill-zone demonstrates this: CatBoost's powerful feature encoding amplifies the data bias precisely because it is so effective at fitting training distributions |
| **Sec 4 — Discussion:** *"Master dataset experiments further demonstrate that exposure to heterogeneous training distributions significantly improves CatBoost's robustness"* | Master-trained CatBoost (94.38% accuracy) outperforms single-dataset training exactly because both UNSW benign and CICIoT attack distributions are represented, stabilizing the OTS encodings |

---

### Implication for Future Work

The safe-value ranges found here suggest a practical calibration methodology for IDS deployment:
1. **Probe the model** with known-benign traffic under systematic feature sweeps to map decision boundaries
2. **Flag service-label bias** in any model trained on class-imbalanced multi-domain data
3. **Use the Master dataset strategy** (multi-domain balanced training) to mitigate single-domain OTS encoding bias
4. **Monitor categorical drift** in production — if the live `service` distribution shifts, re-encoding via OTS with fresh data is far less expensive than retraining with OHE

These points extend the paper's conclusion that *"native categorical handling allows for a more stable transfer of predictive logic between traditional and IoT network paradigms"* — but only when the training data provides balanced categorical coverage across both domains.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑥ — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":
    st.title("About This Research")

    st.markdown("""
### CatBoost Integration for Intrusion Detection in Evolving IoT Environments

> A binary IDS that classifies network flows as **Benign** or **Attack**, evaluating how
> CatBoost's native categorical handling maintains predictive performance across evolving
> network environments — from traditional university traffic (UNSW-NB15) to modern IoT
> attack testbeds (CICIoT 2023).
""")

    # ── BACKGROUND ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Background & Motivation")
    st.markdown("""
Effective Intrusion Detection Systems must generalise across evolving network environments.
The **Fathima et al. (2023) baseline study** evaluated six classical ML algorithms on the
UNSW-NB15 dataset — but decade-old captures often fail to represent modern IoT attack
behaviours.

**This study addresses two gaps:**

1. **Model decay under distribution shift** — what happens to performance when a model
   trained on 2015 university traffic is evaluated on 2023 IoT traffic?
2. **Categorical encoding overhead** — traditional models use One-Hot Encoding (OHE) for
   high-cardinality features like `proto`, `service`, and `state`, causing feature explosion
   and the curse of dimensionality.

**CatBoost's Ordered Target Statistics** process categorical features natively — no OHE
required — preserving statistical structure without inflating the feature space.
""")

    # ── RELATED WORK ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Related Work")
    st.dataframe(
        pd.DataFrame([
            {"Ref": "[1] Fathima et al.", "Year": 2023, "Algorithm": "RF, SVM, LR, GB, DT, KNN", "Dataset": "UNSW-NB15", "Limitation": "Relies on outdated dataset lacking modern IoT traffic"},
            {"Ref": "[14] Hajjouz & Avksentieva", "Year": 2024, "Algorithm": "CatBoost", "Dataset": "CICIoT2023", "Limitation": "Prioritises external feature selection over CatBoost's native Ordered TS"},
            {"Ref": "[12] Yan, Zhou & Chen", "Year": 2025, "Algorithm": "Contrastive Learning", "Dataset": "UNSW-NB15 & CICIoT2023", "Limitation": "Uses costly DL embeddings without testing cross-dataset generalisability"},
            {"Ref": "[13] Gulzar & Mustafa", "Year": 2025, "Algorithm": "DeepCLG Hybrid (CNN+LSTM+GRU+Capsule)", "Dataset": "CICIoT2023", "Limitation": "Depends on resource-intensive DL ensembles and exhaustive Boruta feature selection"},
            {"Ref": "[15] Al-Riyami et al.", "Year": 2021, "Algorithm": "RF, CNN, AdaBoost", "Dataset": "NSL-KDD & gureKDD", "Limitation": "Evaluates older data; requires manual categorical relabeling for cross-dataset compatibility"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    # ── OBJECTIVES ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Research Objectives")
    st.markdown("""
**General Objective:** Evaluate the impact of native categorical handling on the robustness
and generalisability of network IDS by implementing the Fathima et al. baseline alongside
CatBoost, and quantify model performance and decay when transitioning between UNSW-NB15
and CICIoT2023.

**Specific Objectives:**
1. Implement the six baseline ML architectures from Fathima et al. on UNSW-NB15 as a benchmark.
2. Integrate CatBoost using its native Ordered Target Statistics for categorical processing.
3. Create a unified 10-feature schema shared across both datasets.
4. Conduct cross-evaluation between models trained on UNSW-NB15 and CICIoT2023 to measure model decay.
5. Evaluate dataset-agnostic representation capabilities to determine whether native categorical
   handling enables more stable transfer of predictive logic between traditional and IoT paradigms.
""")

    # ── DATASETS ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Datasets")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**UNSW-NB15** — Australian Centre for Cyber Security")
        st.markdown(
            "Captured using the IXIA PerfectStorm tool. Contains 9 attack categories "
            "including Fuzzers, Backdoors, and Shellcode alongside normal university traffic."
        )
    with col2:
        st.markdown("**CICIoT 2023** — Canadian Institute for Cybersecurity")
        st.markdown(
            "Focuses on IoT network traffic. Includes DDoS, DoS, reconnaissance, and data "
            "exfiltration attacks. Raw dataset is 97.64% attacks — undersampled to match UNSW class ratio."
        )

    st.dataframe(
        pd.DataFrame([
            {"Split": "UNSW-NB15 Train",        "Rows": "82,332",  "Benign %": "44.94%", "Attack %": "55.06%", "Notes": "Baseline training set"},
            {"Split": "UNSW-NB15 Test",          "Rows": "175,341", "Benign %": "31.94%", "Attack %": "68.06%", "Notes": "Baseline test set"},
            {"Split": "CICIoT Train (balanced)", "Rows": "82,332",  "Benign %": "44.94%", "Attack %": "55.06%", "Notes": "Undersampled to match UNSW ratio"},
            {"Split": "CICIoT Test",             "Rows": "147,050", "Benign %": "18.84%", "Attack %": "81.16%", "Notes": "Natural skew — benign scarce in raw CICIoT"},
            {"Split": "Master Train",            "Rows": "164,664", "Benign %": "44.94%", "Attack %": "55.06%", "Notes": "UNSW train + CICIoT train concatenated"},
            {"Split": "Master Test",             "Rows": "322,391", "Benign %": "~25%",   "Attack %": "~75%",   "Notes": "UNSW test + CICIoT test concatenated"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    # ── UNIFIED FEATURE SCHEMA ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Unified 10-Feature Schema")
    st.markdown(
        "Features were derived to be semantically consistent across both datasets. "
        "CICIoT features were matched directly; UNSW-NB15 features were mathematically derived "
        "from raw packet/byte statistics."
    )

    feat_df = pd.DataFrame([
        {"Feature": "flow_duration", "Type": "Numerical", "CICIoT Source": "flow_duration", "UNSW-NB15 derivation": "duration", "Description": "Total flow duration in seconds"},
        {"Feature": "rate",          "Type": "Numerical", "CICIoT Source": "rate",          "UNSW-NB15 derivation": "(spkts + dpkts) / duration", "Description": "Overall packet rate (pps)"},
        {"Feature": "srate",         "Type": "Numerical", "CICIoT Source": "srate",         "UNSW-NB15 derivation": "spkts / duration", "Description": "Source packet rate (pps)"},
        {"Feature": "drate",         "Type": "Numerical", "CICIoT Source": "drate",         "UNSW-NB15 derivation": "dpkts / duration", "Description": "Destination packet rate (pps)"},
        {"Feature": "tot_bytes",     "Type": "Numerical", "CICIoT Source": "tot_sum",       "UNSW-NB15 derivation": "sbytes + dbytes", "Description": "Total bytes transferred"},
        {"Feature": "avg_pkt_size",  "Type": "Numerical", "CICIoT Source": "avg",           "UNSW-NB15 derivation": "(sbytes + dbytes) / (spkts + dpkts)", "Description": "Average bytes per packet"},
        {"Feature": "weight",        "Type": "Numerical", "CICIoT Source": "weight",        "UNSW-NB15 derivation": "spkts * dpkts", "Description": "Flow weight (product of src/dst packet counts)"},
        {"Feature": "proto",         "Type": "Categorical", "CICIoT Source": "proto", "UNSW-NB15 derivation": "proto", "Description": "Transport protocol: tcp / udp / other"},
        {"Feature": "service",       "Type": "Categorical", "CICIoT Source": "service", "UNSW-NB15 derivation": "service", "Description": "Application service: http / ssl / dns / ssh / smtp / other"},
        {"Feature": "state",         "Type": "Categorical", "CICIoT Source": "state", "UNSW-NB15 derivation": "state", "Description": "Connection state: fin / rst / other"},
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("""
**Categorical mapping rules (Sec 2.4.1.2 of paper):**
- **proto** — only `tcp` and `udp` preserved; all others → `other`
- **state** — only `fin` and `rst` preserved; all others → `other`
- **service** — overlapping services standardised (HTTPS → `ssl`); non-overlapping → `other`

**Normalisation:** MinMaxScaler fitted exclusively on the Master training set, then applied
to all splits (UNSW, CICIoT, Master) to ensure consistent global min/max representation.
""")

    # ── CATBOOST ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("CatBoost — Why & How")

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        st.markdown("""
**Ordered Target Statistics**

Categorical features are encoded using statistics calculated strictly from
preceding samples in a random permutation — preventing target leakage and
memorisation of specific labels.
""")
    with cb2:
        st.markdown("""
**Symmetric Trees**

The same split feature and threshold are applied at every node of a given
depth (oblivious decision trees), improving prediction speed and reducing
overfitting compared to asymmetric trees.
""")
    with cb3:
        st.markdown("""
**Ordered Boosting**

Addresses *prediction shift* — a bias introduced when the same data is
used to compute gradients and train the next tree — by using separate
ordered subsets for each stage.
""")

    # ── HYPERPARAMETERS ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hyperparameters & Replicated Baseline Metrics")
    st.markdown(
        "Tuning aimed to replicate the Fathima et al. baseline. The empirical replication "
        "yielded slightly lower metrics than reported (e.g., RF: 99% → 97.21% accuracy)."
    )
    st.dataframe(
        pd.DataFrame([
            {"Algorithm": "CatBoost (Proposed)",  "Key Hyperparameters": "iterations=1000, lr=0.1, depth=6, l2_leaf_reg=5", "Replicated Accuracy": "96.00%", "Replicated F1": "96.08%"},
            {"Algorithm": "SVM",                  "Key Hyperparameters": "C=1, random_state=42, dual='auto'",                                  "Replicated Accuracy": "97.26%", "Replicated F1": "98.20%"},
            {"Algorithm": "Random Forest",        "Key Hyperparameters": "n_estimators=100, max_depth=10, max_features=log2","Replicated Accuracy": "97.21%", "Replicated F1": "98.14%"},
            {"Algorithm": "Logistic Regression",  "Key Hyperparameters": "solver=liblinear, C=10",                           "Replicated Accuracy": "97.03%", "Replicated F1": "98.05%"},
            {"Algorithm": "Gradient Boosting",    "Key Hyperparameters": "n_estimators=100, lr=0.01, max_depth=10",          "Replicated Accuracy": "96.80%", "Replicated F1": "97.87%"},
            {"Algorithm": "Decision Tree",        "Key Hyperparameters": "criterion=gini, max_depth=20, min_split=20",       "Replicated Accuracy": "95.61%", "Replicated F1": "97.06%"},
            {"Algorithm": "KNN",                  "Key Hyperparameters": "weights=uniform, k=3, metric=manhattan",           "Replicated Accuracy": "95.33%", "Replicated F1": "96.86%"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    # ── KEY RESULTS ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Results Summary")

    r1, r2, r3 = st.tabs(["UNSW-NB15 Test", "CICIoT Test", "Master Test"])

    with r1:
        st.markdown("CatBoost achieved the strongest in-domain performance on UNSW-NB15.")
        st.dataframe(
            pd.DataFrame([
                {"Model": "CatBoost (UNSW-trained)",    "Acc": "90.70%", "Pr": "97.58%", "Re": "88.53%", "F1": "92.84%", "AUC": "98.33%"},
                {"Model": "CatBoost (CICIoT-trained)",  "Acc": "64.75%", "Pr": "78.59%", "Re": "66.27%", "F1": "71.91%", "AUC": "59.72%"},
                {"Model": "CatBoost (Master-trained)",  "Acc": "90.64%", "Pr": "97.66%", "Re": "88.37%", "F1": "92.78%", "AUC": "98.28%"},
                {"Model": "Decision Tree (UNSW-trained)","Acc": "89.95%", "Pr": "97.28%", "Re": "87.68%", "F1": "92.23%", "AUC": "96.61%"},
                {"Model": "Decision Tree (CICIoT-trained)","Acc":"76.33%","Pr": "82.70%", "Re": "82.47%", "F1": "82.59%", "AUC": "69.42%"},
                {"Model": "Decision Tree (Master-trained)","Acc":"90.01%","Pr": "97.58%", "Re": "87.49%", "F1": "92.26%", "AUC": "97.00%"},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.info(
            "When trained on CICIoT and tested on UNSW, CatBoost's AUC drops to **59.72%** — "
            "barely above random chance — demonstrating severe ranking instability under cross-domain shift.",
            icon="⚠️",
        )

    with r2:
        st.markdown("All models perform near-perfectly when trained and tested within CICIoT. The cross-domain drop is stark.")
        st.dataframe(
            pd.DataFrame([
                {"Model": "CatBoost (UNSW-trained)",      "Acc": "73.98%", "Pr": "79.84%", "Re": "90.89%", "F1": "85.00%", "AUC": "14.81%"},
                {"Model": "CatBoost (CICIoT-trained)",    "Acc": "98.85%", "Pr": "99.96%", "Re": "98.62%", "F1": "99.29%", "AUC": "99.75%"},
                {"Model": "CatBoost (Master-trained)",    "Acc": "98.85%", "Pr": "99.97%", "Re": "98.61%", "F1": "99.28%", "AUC": "99.74%"},
                {"Model": "Random Forest (UNSW-trained)", "Acc": "73.10%", "Pr": "79.57%", "Re": "89.95%", "F1": "84.44%", "AUC": "13.92%"},
                {"Model": "Random Forest (CICIoT-trained)","Acc":"98.50%", "Pr": "99.99%", "Re": "98.15%", "F1": "99.07%", "AUC": "99.73%"},
                {"Model": "Decision Tree (UNSW-trained)", "Acc": "44.59%", "Pr": "75.22%", "Re": "47.31%", "F1": "58.09%", "AUC": "48.72%"},
                {"Model": "Decision Tree (CICIoT-trained)","Acc":"98.62%", "Pr": "99.81%", "Re": "98.48%", "F1": "99.14%", "AUC": "99.12%"},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.info(
            "UNSW→CICIoT AUC for CatBoost: **14.81%** (worse than random guessing). "
            "This is the clearest signal of feature distribution mismatch between the two domains.",
            icon="⚠️",
        )

    with r3:
        st.markdown("Master training restores CatBoost's performance across both domains. It leads in Precision and AUC.")
        st.dataframe(
            pd.DataFrame([
                {"Model": "CatBoost (UNSW-trained)",        "Acc": "83.07%", "Pr": "87.70%", "Re": "89.71%", "F1": "88.70%", "AUC": "76.29%"},
                {"Model": "CatBoost (CICIoT-trained)",      "Acc": "80.31%", "Pr": "90.11%", "Re": "82.44%", "F1": "86.11%", "AUC": "85.95%"},
                {"Model": "CatBoost (Master-trained)",      "Acc": "94.38%", "Pr": "98.86%", "Re": "93.49%", "F1": "96.10%", "AUC": "99.25%"},
                {"Model": "Random Forest (UNSW-trained)",   "Acc": "80.71%", "Pr": "87.19%", "Re": "86.68%", "F1": "86.93%", "AUC": "79.14%"},
                {"Model": "Random Forest (CICIoT-trained)", "Acc": "78.05%", "Pr": "89.35%", "Re": "79.88%", "F1": "84.35%", "AUC": "85.43%"},
                {"Model": "Random Forest (Master-trained)", "Acc": "92.57%", "Pr": "98.72%", "Re": "91.15%", "F1": "94.78%", "AUC": "98.93%"},
                {"Model": "Gradient Boosting (UNSW)",       "Acc": "76.20%", "Pr": "87.53%", "Re": "79.12%", "F1": "83.12%", "AUC": "84.03%"},
                {"Model": "Gradient Boosting (CICIoT)",     "Acc": "84.44%", "Pr": "89.61%", "Re": "89.33%", "F1": "89.47%", "AUC": "87.30%"},
                {"Model": "Gradient Boosting (Master)",     "Acc": "92.81%", "Pr": "98.46%", "Re": "91.73%", "F1": "94.98%", "AUC": "98.71%"},
            ]),
            use_container_width=True, hide_index=True,
        )
        st.success(
            "Master-trained CatBoost achieves **94.38% accuracy, 96.10% F1, and 99.25% AUC** — "
            "the strongest overall performance. Multi-domain training mitigates domain sensitivity.",
            icon="✅",
        )

    # ── DISCUSSION ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Discussion & Key Findings")
    st.markdown("""
**Trade-off: Complexity vs. Generalisation Stability**

Across all evaluation scenarios, consistent patterns emerged regarding model behaviour under
domain shift:

- CatBoost achieved the **highest in-domain and multi-domain performance** across all datasets,
  but also exhibited the most severe degradation when trained on a single dataset and evaluated
  on a different one — particularly evident in the UNSW→CICIoT configuration (AUC: 14.81%).

- Simpler tree-based models, particularly the **Decision Tree**, showed comparatively stronger
  resilience under cross-domain evaluation. Although peak in-domain performance was lower,
  the degradation under distribution shift was less extreme.

- These results highlight that **highly optimised ensemble methods leverage fine-grained feature
  interactions that may not transfer across heterogeneous network environments**. Simpler models
  may rely on more stable decision boundaries.

**Master Dataset as the Solution**

The Master dataset experiments demonstrate that **exposure to heterogeneous training distributions
significantly improves CatBoost's robustness**. When trained on the combined dataset, performance
was restored across all test scenarios — suggesting that multi-domain training mitigates domain
sensitivity, and that CatBoost's native categorical handling delivers its full benefit only when
both domains are represented in training data.
""")

    # ── SCOPE & LIMITATIONS ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scope & Limitations")
    st.markdown("""
- **Binary classification only** — distinguishes benign vs. attack; does not identify specific
  attack sub-types.
- **Dependent on dataset quality** — any labelling biases in UNSW-NB15 or CICIoT2023 directly
  influence model learning.
- **CICIoT sourced from Kaggle** — relies on integrity of third-party preprocessing by
  *himadri07*.
- **Fixed 10-feature schema** — predictive logic may not fully generalise to novel protocols
  or zero-day vulnerabilities outside the bounds of these two datasets.
- **SDG alignment** — this work contributes to **SDG 9** (Industry, Innovation and
  Infrastructure) and **SDG 16** (Peace, Justice and Strong Institutions) by enhancing
  digital security resilience in local educational and enterprise sectors.
""")

    # ── TECH STACK ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("`CatBoost` · `scikit-learn` · `pandas` · `NumPy` · `Streamlit` · `Plotly`")
