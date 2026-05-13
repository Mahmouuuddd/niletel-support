import streamlit as st
import requests
import uuid

# ============================================================
# CONFIG
# ============================================================

FASTAPI_URL = "https://excursion-scared-paralegal.ngrok-free.dev"

st.set_page_config(
    page_title="NileTel Support",
    page_icon="📡",
    layout="centered",
)

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

* { font-family: 'Cairo', sans-serif; }

.stApp { background-color: #f0f4f8; }

#MainMenu, footer, header { visibility: hidden; }

/* Top banner */
.top-banner {
    background: linear-gradient(135deg, #003366 0%, #0055a5 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    text-align: center;
}
.top-banner h1 { margin: 0; font-size: 24px; font-weight: 700; }
.top-banner p  { margin: 4px 0 0; font-size: 13px; opacity: 0.85; }

/* Chat bubbles */
.bubble-bot {
    background: #ffffff;
    border: 1px solid #dde3ed;
    color: #1a1a2e;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.7;
    direction: rtl;
    text-align: right;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    white-space: pre-line;
}
.bubble-user {
    background: #0055a5;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px auto;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.7;
    direction: rtl;
    text-align: right;
    box-shadow: 0 1px 4px rgba(0,85,165,0.3);
}
.bubble-wrapper-bot  { display: flex; justify-content: flex-start; margin: 4px 0; }
.bubble-wrapper-user { display: flex; justify-content: flex-end;   margin: 4px 0; }

/* Ticket summary card */
.ticket-card {
    background: #eaf4ff;
    border: 1.5px solid #0055a5;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    direction: rtl;
    text-align: right;
}
.ticket-card h4 {
    color: #003366;
    margin: 0 0 12px;
    font-size: 16px;
    font-weight: 700;
}
.ticket-card .field {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    padding: 6px 0;
    border-bottom: 1px solid #c8dff5;
    color: #1a1a2e;
}
.ticket-card .field:last-child { border-bottom: none; }
.ticket-card .label { color: #0055a5; font-weight: 600; min-width: 90px; text-align: left; }
.ticket-badge {
    display: inline-block;
    background: #003366;
    color: white;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}

/* Collecting indicator */
.collecting-indicator {
    background: #fff8e1;
    border-right: 3px solid #f5a623;
    padding: 8px 12px;
    border-radius: 8px 0 0 8px;
    font-size: 13px;
    color: #7a5c00;
    margin-bottom: 10px;
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []       # list of {role, content, meta}

if "collecting" not in st.session_state:
    st.session_state.collecting = False  # True while mid ticket-collection


# ============================================================
# HELPERS
# ============================================================

def call_fastapi(message: str) -> dict:
    """Sends message to FastAPI and returns the response dict."""
    try:
        resp = requests.post(
             f"{FASTAPI_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": "⚠️ مش قادر أتواصل مع السيرفر. تأكد إن FastAPI شغال على port 8000.",
            "needs_action": False,
            "sources": [],
            "session_id": st.session_state.session_id,
            "collecting": False,
        }
    except Exception as e:
        return {
            "answer": f"⚠️ حصل خطأ غير متوقع: {str(e)}",
            "needs_action": False,
            "sources": [],
            "session_id": st.session_state.session_id,
            "collecting": False,
        }


def render_ticket_card(ticket_data: dict):
    """Renders the completed ticket as a styled summary card."""
    st.markdown(f"""
    <div class="ticket-card">
        <span class="ticket-badge">📋 تذكرة جديدة</span>
        <h4>ملخص التذكرة</h4>
        <div class="field">
            <span>{ticket_data.get('name', '—')}</span>
            <span class="label">👤 الاسم</span>
        </div>
        <div class="field">
            <span>{ticket_data.get('email', '—')}</span>
            <span class="label">📧 الإيميل</span>
        </div>
        <div class="field">
            <span>{ticket_data.get('location', '—')}</span>
            <span class="label">📍 الموقع</span>
        </div>
        <div class="field">
            <span style="font-size:13px">{ticket_data.get('issue', '—')}</span>
            <span class="label">🔧 المشكلة</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(msg: dict):
    """Renders a single chat bubble (user or bot)."""
    role        = msg["role"]
    content     = msg["content"]
    ticket_data = msg.get("meta", {}).get("ticket_data")

    if role == "user":
        st.markdown(
            f'<div class="bubble-wrapper-user">'
            f'<div class="bubble-user">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-wrapper-bot">'
            f'<div class="bubble-bot">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show ticket card inline right after the confirmation message
        if ticket_data:
            render_ticket_card(ticket_data)


# ============================================================
# UI
# ============================================================

# Top banner
st.markdown("""
<div class="top-banner">
    <h1>📡 NileTel — خدمة العملاء</h1>
    <p>مساعدك الذكي لحل مشاكل الإنترنت والاتصالات</p>
</div>
""", unsafe_allow_html=True)

# Yellow bar shown only while bot is actively collecting ticket info
if st.session_state.collecting:
    st.markdown("""
    <div class="collecting-indicator">
        🎫 جاري إنشاء تذكرة — برجاء الإجابة على الأسئلة القادمة، أو قول "إلغاء" للتراجع
    </div>
    """, unsafe_allow_html=True)

# Chat history
with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div class="bubble-wrapper-bot">
            <div class="bubble-bot">
                أهلاً وسهلاً! 👋 معاك نيل بوت، مساعد خدمة عملاء NileTel.
                أقدر أساعدك في إيه النهاردة؟
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="message",
            label_visibility="collapsed",
            placeholder="اكتب رسالتك هنا...",
        )
    with col2:
        submitted = st.form_submit_button("إرسال", use_container_width=True)

# Reset button
if st.button("🔄 محادثة جديدة"):
    st.session_state.messages   = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.collecting = False
    st.rerun()

# ============================================================
# HANDLE SUBMISSION
# ============================================================

if submitted and user_input.strip():

    # 1. Add user bubble to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
    })

    # 2. Call FastAPI
    with st.spinner("جاري المعالجة..."):
        response = call_fastapi(user_input.strip())

    # 3. Extract fields
    answer      = response.get("answer", "")
    collecting  = response.get("collecting", False)
    ticket_data = response.get("ticket_data")   # only present when collection is done

    # Keep session_id in sync
    st.session_state.session_id = response.get("session_id", st.session_state.session_id)

    # Update collecting flag (controls the yellow bar)
    st.session_state.collecting = collecting

    # 4. Add bot bubble to history
    st.session_state.messages.append({
        "role": "bot",
        "content": answer,
        "meta": {
            "ticket_data": ticket_data,  # None until all fields are collected
        },
    })

    st.rerun()