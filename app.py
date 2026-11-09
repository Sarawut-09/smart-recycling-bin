import base64
import json
import time
import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "recycling_model_3class.keras"
CONFIG_PATH = APP_DIR / "recycling_model_config.json"


@st.cache_data(ttl=3600, show_spinner=False)
def get_cloudflare_ice_servers(turn_key_id: str, turn_api_token: str):
    """Generate short-lived Cloudflare TURN credentials on the server."""
    url = (
        "https://rtc.live.cloudflare.com/v1/turn/keys/"
        f"{turn_key_id}/credentials/generate-ice-servers"
    )
    request = urllib.request.Request(
        url,
        data=b'{"ttl":86400}',
        headers={
            "Authorization": f"Bearer {turn_api_token}",
            "Content-Type": "application/json",
            "User-Agent": "streamlit-webrtc",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))["iceServers"]


def get_rtc_configuration():
    """Use TURN in the cloud, with Google STUN as a local-development fallback."""
    fallback = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    try:
        turn_key_id = st.secrets["CLOUDFLARE_TURN_KEY_ID"]
        turn_api_token = st.secrets["CLOUDFLARE_TURN_KEY_API_TOKEN"]
    except (KeyError, FileNotFoundError):
        return fallback

    try:
        return {
            "iceServers": get_cloudflare_ice_servers(
                str(turn_key_id), str(turn_api_token)
            )
        }
    except Exception:
        st.warning(
            "เชื่อมต่อ TURN server ไม่สำเร็จ กำลังลองเชื่อมต่อกล้องแบบปกติ "
            "โปรดลองรีเฟรชหน้าเว็บอีกครั้ง"
        )
        return fallback


@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    return model, config


@st.cache_resource
def get_shared_result():
    # เก็บผลทำนายล่าสุดไว้ที่เดียว เพื่อให้ทั้ง process_frame (thread กล้อง)
    # และส่วนแสดงผล UI (thread หลักของ Streamlit) อ่านค่าเดียวกันได้
    return {
        "time": 0.0,
        "class_name": "กำลังรอภาพ",
        "recyclable_score": 0.0,
    }


@st.cache_resource
def font_face_css() -> str:
    # ฝังฟอนต์ Bebas Neue ลงในหน้าเว็บ (วางไฟล์ .ttf ไว้ข้าง app.py หรือในโฟลเดอร์ fonts)
    for path in (
        APP_DIR / "BebasNeue-Regular.ttf",
        APP_DIR / "fonts" / "BebasNeue-Regular.ttf",
    ):
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return (
                "<style>@font-face{font-family:'Bebas Neue';font-style:normal;"
                "font-weight:400;font-display:swap;"
                f"src:url(data:font/ttf;base64,{data}) format('truetype');}}</style>"
            )
    return ""


# ----------------------------------------------------------------------------
# UI : ธีมสี เขียว / ฟ้า / แดง / เหลือง
# ----------------------------------------------------------------------------
APP_CSS = """
<style>
:root{
  --bg:#F3F8F4; --card:#FFFFFF; --line:#D3E3D8;
  --brand:#1F7A43; --brand-dark:#175F34;
  --ink:#12261A; --muted:#5A6F61;
  --green:#2E9E5B; --green-dark:#1F7A43;
  --red:#D64545; --red-dark:#A93232;
}

.stApp{ background:var(--bg); color:var(--ink); }
.stMarkdown, .stMarkdown *{ font-family:'Bebas Neue','Prompt',system-ui,sans-serif; text-transform:uppercase; font-weight:700; }
.block-container{ padding-top:1.4rem; padding-bottom:1rem; max-width:1500px; }
[data-testid="stHeader"]{ background:transparent; }
.stAppDeployButton, [data-testid="stAppDeployButton"], footer{ display:none !important; }
.stApp iframe{ border-radius:12px; }

/* ---------- Hero ---------- */
.hero{
  border-radius:12px; padding:18px 24px; margin-bottom:20px;
  background:var(--brand);
}
.hero-top{ display:flex; gap:16px; align-items:center; }
.hero-icon{
  width:52px; height:52px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  font-size:1.8rem; background:#fff;
}
.hero-title{ font-size:3rem; font-weight:700; line-height:1.05; color:#fff; letter-spacing:.07em; }

/* ---------- Cards ---------- */
.card{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; margin-bottom:14px;
}
.section-title{ font-weight:700; font-size:1.7rem; line-height:1.2; letter-spacing:.12em; color:var(--ink); margin:0 0 12px; height:2rem; }

/* ---------- Bin ---------- */
.bin-stage{ display:flex; justify-content:center; padding:6px 0 2px; }
.bin-wrap{ container-type:inline-size; }
/* ความสูงการ์ดถัง = ความสูงกล้อง (วิดีโอ 4:3 กว้าง ~1.75 เท่าของคอลัมน์ถัง + แถบปุ่ม ~55px) */
.bin-card{
  box-sizing:border-box; margin:0; height:calc(131cqw + 55px);
  display:flex; align-items:center; justify-content:center;
}
.bin-card .bin-stage{ width:100%; padding:0; }
.bin-svg{ width:100%; max-width:300px; height:auto; overflow:visible; }

.lid{ transform-origin:50px 140px; }
.lid.open{ animation:lid-open .6s cubic-bezier(.2,.9,.3,1.2) forwards; }
.lid.closing{ animation:lid-close .45s ease-in forwards; }
.item{ animation:drop 1.1s ease-in .35s both; }

@keyframes lid-open{ from{ transform:rotate(0deg); } to{ transform:rotate(-62deg); } }
@keyframes lid-close{ from{ transform:rotate(-62deg); } to{ transform:rotate(0deg); } }
@keyframes drop{
  0%{ transform:translateY(-20px); opacity:0; }
  15%{ opacity:1; }
  100%{ transform:translateY(100px); opacity:1; }
}

/* ---------- Lights ---------- */
.lights{ display:flex; justify-content:center; gap:44px; margin-top:16px; }
.light{ display:flex; flex-direction:column; align-items:center; gap:8px; }
.bulb{ width:58px; height:58px; border-radius:50%; transition:all .3s ease; }
.bulb.g{ background:var(--green); }
.bulb.r{ background:var(--red); }
.bulb.off{ opacity:.18; }
.bulb.on{ opacity:1; }
.bulb.on.g{ box-shadow:0 0 0 5px rgba(46,158,91,.22); }
.bulb.on.r{ box-shadow:0 0 0 5px rgba(214,69,69,.22); }
.light-label{ font-size:1.5rem; font-weight:700; letter-spacing:.14em; color:var(--muted); transition:color .3s; }
.light-label.on.g{ color:var(--green-dark); }
.light-label.on.r{ color:var(--red-dark); }
</style>
"""


def compact(html: str) -> str:
    # ย่อ HTML ให้เป็นบรรทัดเดียว กัน Markdown ตีความว่าเป็น code block
    return "".join(line.strip() for line in html.strip().splitlines())


def hero_html(threshold: float, n_classes: int) -> str:
    return compact(
        f"""
        <div class="hero">
          <div class="hero-top">
            <div class="hero-icon">♻️</div>
            <div>
              <div class="hero-title">Smart Recycling Bin</div>
            </div>
          </div>
        </div>
        """
    )


def bin_card(mode: str, closing: bool = False) -> str:
    """mode: idle / open / closed"""
    is_open = mode == "open"
    if is_open:
        lid_cls, lid_color, handle_color = "open", "#43C06F", "#2E9E5B"
        led_color = "#43C06F"
        item = (
            '<g class="item">'
            '<rect x="108" y="62" width="24" height="40" rx="7" fill="#BFDCC8"/>'
            '<rect x="114" y="52" width="12" height="12" rx="3" fill="#1F7A43"/>'
            "</g>"
        )
    elif mode == "closed":
        lid_cls = "closing" if closing else ""
        lid_color, handle_color = "#D64545", "#A93232"
        led_color = "#D64545"
        item = ""
    else:
        lid_cls, lid_color, handle_color = "", "#8AA595", "#5F7A6B"
        led_color = "#8AA595"
        item = ""

    return compact(
        f"""
        <div class="bin-wrap">
        <div class="card bin-card">
          <div class="bin-stage {mode}">
            <svg viewBox="0 0 240 340" xmlns="http://www.w3.org/2000/svg" class="bin-svg">
              <ellipse cx="120" cy="331" rx="70" ry="7" fill="#12261A" opacity=".12"/>
              <ellipse cx="120" cy="150" rx="66" ry="9" fill="#0D3A20"/>
              {item}
              <path d="M54 150 L70 308 Q71 320 84 320 L156 320 Q169 320 170 308 L186 150 Z" fill="#238A4B"/>
              <path d="M92 158 L96 302 M148 158 L144 302" stroke="#fff" stroke-opacity=".10" stroke-width="3" fill="none"/>
              <polygon points="65.3,262 174.7,262 173.2,276 66.8,276" fill="#fff" fill-opacity=".9"/>
              <text x="120" y="238" text-anchor="middle" font-size="72" fill="#fff" fill-opacity="1">&#9851;&#xFE0E;</text>
              <circle cx="168" cy="172" r="5.5" fill="{led_color}" stroke="#fff" stroke-width="1.5"/>
              <g class="lid {lid_cls}">
                <rect x="44" y="128" width="152" height="24" rx="9" fill="{lid_color}"/>
                <rect x="104" y="116" width="32" height="14" rx="6" fill="{handle_color}"/>
              </g>
            </svg>
          </div>
        </div>
        </div>
        """
    )


def info_card(mode: str, class_name: str, score: float, threshold: float) -> str:
    if mode == "open":
        icon, title, sub = "✅", "OPEN (demo)", "เปิดฝาถัง — ผ่านเกณฑ์ recyclable"
    elif mode == "closed":
        icon, title, sub = "⛔", "KEEP CLOSED", "ปิดฝาไว้ — ยังไม่ผ่านเกณฑ์"
    else:
        icon, title, sub = "⏳", "STANDBY", "รอเปิดกล้อง กด START ทางซ้าย"

    if mode == "idle":
        pill = "b"
    elif class_name == "recyclable":
        pill = "g"
    else:
        pill = "r"

    shown_class = "—" if mode == "idle" else class_name
    shown_score = 0.0 if mode == "idle" else score
    fill = {"open": "g", "closed": "r"}.get(mode, "b")
    pct = max(0.0, min(100.0, shown_score * 100))
    thr_pct = max(0.0, min(100.0, threshold * 100))

    return compact(
        f"""
        <div class="card">
          <div class="card-title">📊 สถานะการตรวจจับ</div>
          <div class="status {mode}">
            <div class="status-ico">{icon}</div>
            <div>
              <div class="status-title">{title}</div>
              <div class="status-sub">{sub}</div>
            </div>
          </div>
          <div class="row"><span>ตรวจพบ</span><span class="pill {pill}">{shown_class}</span></div>
          <div class="meter-label"><span>Recyclable score</span><b>{shown_score:.1%}</b></div>
          <div class="meter">
            <div class="fill {fill}" style="width:{pct:.1f}%"></div>
            <div class="thr" style="left:calc({thr_pct:.1f}% - 1px)"></div>
          </div>
          <div class="meter-foot"><span>0%</span><span class="t">เกณฑ์เปิดฝา {threshold:.0%}</span><span>100%</span></div>
        </div>
        """
    )


def lights_html(mode: str) -> str:
    """mode: idle / open / closed -> หลอดไฟของสถานะปัจจุบันจะสว่าง อีกดวงจะมัวลง"""
    g = "on" if mode == "open" else "off"
    r = "on" if mode == "closed" else "off"
    return compact(
        f"""
        <div class="lights">
          <div class="light">
            <div class="bulb g {g}"></div>
            <div class="light-label g {g}">OPEN</div>
          </div>
          <div class="light">
            <div class="bulb r {r}"></div>
            <div class="light-label r {r}">CLOSED</div>
          </div>
        </div>
        """
    )



# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Recycling Bin",
    page_icon="♻️",
    layout="wide",
)
st.markdown(font_face_css() + APP_CSS, unsafe_allow_html=True)

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error("ไม่พบไฟล์โมเดลหรือ config ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

model, config = load_model_and_config()
class_names = config["class_names"]
threshold = 0.90  # เกณฑ์เปิดฝา 90%
image_size = tuple(config["image_size"])
recyclable_index = class_names.index("recyclable")

st.markdown(hero_html(threshold, len(class_names)), unsafe_allow_html=True)

last_result = get_shared_result()


def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    # รับภาพจากกล้อง แล้วกลับซ้าย–ขวาแบบกระจก
    image_bgr = frame.to_ndarray(format="bgr24")
    image_bgr = cv2.flip(image_bgr, 1)

    now = time.monotonic()

    # ทำนายทุกครึ่งวินาทีเพื่อลดภาระการประมวลผล
    if now - last_result["time"] >= 0.5:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, image_size)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

        probabilities = model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        last_result["class_name"] = class_names[predicted_index]
        last_result["recyclable_score"] = float(
            probabilities[recyclable_index]
        )
        last_result["time"] = now

    score = last_result["recyclable_score"]
    should_open = score >= threshold

    if should_open:
        status = "OPEN"
        color = (0, 180, 0)
    else:
        status = "CLOSED"
        color = (0, 0, 220)

    cv2.putText(
        image_bgr,
        status,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
    )

    return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


# กล้องใหญ่ทางซ้าย / ถังขยะ + สถานะทางขวา
cam_col, bin_col = st.columns([1.7, 1], gap="large")

with cam_col:
    st.markdown('<div class="section-title">CAMERA</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="recycling-camera",
        video_frame_callback=process_frame,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=get_rtc_configuration(),
    )

with bin_col:
    st.markdown('<div class="section-title">BIN</div>', unsafe_allow_html=True)
    bin_ph = st.empty()

lights_ph = st.empty()

# แสดงสถานะเริ่มต้น (ยังไม่เปิดกล้อง)
bin_ph.markdown(bin_card("idle"), unsafe_allow_html=True)
lights_ph.markdown(lights_html("idle"), unsafe_allow_html=True)

# อัปเดตถังขยะ/สถานะแบบเรียลไทม์ขณะกล้องทำงาน
if ctx.state.playing:
    prev_mode = None
    while ctx.state.playing:
        score = float(last_result["recyclable_score"])
        class_name = last_result["class_name"]
        mode = "open" if score >= threshold else "closed"

        # วาดถังใหม่เฉพาะตอนสถานะเปลี่ยน เพื่อให้แอนิเมชันเปิด/ปิดฝาเล่นครั้งเดียว
        if mode != prev_mode:
            bin_ph.markdown(
                bin_card(mode, closing=(prev_mode == "open")),
                unsafe_allow_html=True,
            )
            lights_ph.markdown(lights_html(mode), unsafe_allow_html=True)
            prev_mode = mode

        time.sleep(0.2)

    bin_ph.markdown(bin_card("idle"), unsafe_allow_html=True)
    lights_ph.markdown(lights_html("idle"), unsafe_allow_html=True)
