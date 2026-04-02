import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
from datetime import datetime
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="YOLOv8 Detector", layout="wide")
st.title("🎯 Real-Time Object Detection")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
ALL_CLASSES = list(model.names.values())

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

alert_classes = st.sidebar.multiselect(
    "🚨 Alert Classes", ALL_CLASSES, default=["person"]
)

record = st.sidebar.checkbox("🎥 Record Video")

# ------------------ WEBRTC CONFIG ------------------
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ------------------ VIDEO PROCESSOR ------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.5
        self.alert_classes = []
        self.record = False

        self.counts = {}
        self.alert_triggered = False

        self.writer = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=self.conf, verbose=False)

        counts = {}
        triggered = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = model.names[int(box.cls[0])]

                counts[cls] = counts.get(cls, 0) + 1

                is_alert = cls in self.alert_classes
                color = (0, 0, 255) if is_alert else (0, 255, 0)

                if is_alert:
                    triggered = True

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{cls} {conf:.0%}"
                cv2.putText(
                    img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # Alert Banner
        if triggered:
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 200), -1)
            cv2.putText(
                img,
                "🚨 ALERT: Target object detected!",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Recording logic
        if self.record:
            if self.writer is None:
                h, w = img.shape[:2]
                filename = f"recording_{datetime.now().strftime('%H%M%S')}.mp4"
                self.writer = cv2.VideoWriter(
                    filename,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (w, h)
                )
            self.writer.write(img)
        else:
            if self.writer:
                self.writer.release()
                self.writer = None

        # Thread-safe update
        with self.lock:
            self.counts = counts
            self.alert_triggered = triggered

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        if self.writer:
            self.writer.release()

# ------------------ LAYOUT ------------------
col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="yolo",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Live Counts")
    count_placeholder = st.empty()

    st.subheader("🚨 Alerts")
    alert_placeholder = st.empty()

# ------------------ UPDATE UI ------------------
# ------------------ AUTO REFRESH ------------------
refresh_rate = 0.5  # seconds

if ctx.video_processor is not None:
    vp = ctx.video_processor

    vp.conf = conf_thresh
    vp.alert_classes = alert_classes
    vp.record = record

    try:
        with vp.lock:
            counts = vp.counts.copy()
            triggered = vp.alert_triggered
    except:
        counts = {}
        triggered = False

    # ✅ LIVE COUNTS
    if counts:
        df = pd.DataFrame({
            "Object": list(counts.keys()),
            "Count": list(counts.values())
        })
        count_placeholder.dataframe(df, use_container_width=True)
    else:
        count_placeholder.info("No objects detected")

    # ✅ ALERTS
    if triggered:
        detected_items = ", ".join(counts.keys())
        alert_placeholder.error(f"🚨 Detected: {detected_items}")
    else:
        alert_placeholder.success("✅ No alerts")

else:
    st.warning("⚠️ Click 'Start' to begin detection")

# 🔁 AUTO REFRESH UI
time.sleep(refresh_rate)
st.rerun()