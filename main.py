import os
import uuid
import csv
import httpx
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import init_rag, run_rag_pipeline

# ============================================================
# CONFIG
# ============================================================

N8N_WEBHOOK_URL = "https://amazingsaqr.app.n8n.cloud/webhook/chat"


# ============================================================
# LIFESPAN — runs once on startup / shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load documents, build embeddings and FAISS index on startup."""
    print("🚀 Starting up — loading RAG system...")
    init_rag()          # from rag.py — loads docs, embeddings, FAISS index
    print("✅ RAG system ready.")
    yield               # server is now running
    print("🛑 Shutting down.")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="NileTel Support API",
    description="RAG-powered customer support backend with ticket automation",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None   # Streamlit generates this once per user session


class ChatResponse(BaseModel):
    answer: str
    needs_action: bool
    sources: list[str]
    session_id: str
    collecting: bool = False        # True while bot is asking for name/email/location
    ticket_created: bool = False    # True when ticket was successfully sent to n8n


# ============================================================
# HELPERS
# ============================================================

async def send_to_n8n(ticket_data: dict) -> bool:
    """
    POSTs the completed ticket payload to the n8n webhook.
    Returns True on success, False on any failure.

    Payload sent to n8n:
    {
        "name":     "أحمد محمد",
        "email":    "ahmed@example.com",
        "location": "المنصورة، شارع الجمهورية",
        "issue":    "النت بطيء من أسبوع"
    }
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(N8N_WEBHOOK_URL, json=ticket_data)
            response.raise_for_status()
            print(f"✅ Ticket sent to n8n: {ticket_data}")
            return True
    except Exception as e:
        print(f"❌ Failed to send ticket to n8n: {e}")
        return False


# ============================================================
# CSV — local ticket log
# ============================================================
 
CSV_PATH = Path("tickets.csv")
CSV_HEADERS = ["ticket_number", "timestamp", "name", "email", "location", "issue"]
 
def generate_ticket_number() -> str:
    """
    Generates a unique ticket number in the format NTL-YYYYMMDD-XXXX
    where XXXX is a zero-padded sequential number based on today's tickets.
    Example: NTL-20260509-0001
    This guarantees uniqueness: date part separates days,
    sequential part separates tickets within the same day.
    """
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"NTL-{today}-"
 
    # Count how many tickets already exist for today
    count = 0
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ticket_number", "").startswith(prefix):
                    count += 1
 
    return f"{prefix}{str(count + 1).zfill(4)}"   # e.g. NTL-20260509-0001  NIL-DateOfToday-NumberOfTicketsMade
 
 
def save_ticket_to_csv(ticket_data: dict) -> str:
    """
    Generates a unique ticket number, appends the ticket to tickets.csv,
    and returns the ticket number so it can be shown to the user and sent to n8n.
    """
    ticket_number = generate_ticket_number()
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "ticket_number": ticket_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": ticket_data.get("name", ""),
            "email": ticket_data.get("email", ""),
            "location": ticket_data.get("location", ""),
            "issue": ticket_data.get("issue", ""),
        })
    print(f"📄 Ticket {ticket_number} saved to {CSV_PATH}")
    return ticket_number


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    """Health check — visit this in your browser to confirm the server is up."""
    return {"status": "ok", "message": "NileTel Support API is running 🚀"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main endpoint called by Streamlit on every message.

    Flow:
    1. If session_id is missing, generate a new one.
    2. Call run_rag_pipeline(message, session_id) from rag.py.
    3. If the response contains ticket_data → send to n8n webhook.
    4. Return the structured response to Streamlit.
    """

    # Generate a session ID if Streamlit didn't send one
    session_id = request.session_id or str(uuid.uuid4())

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # ── Call the RAG pipeline ───────────────────────────────
    try:
        result = run_rag_pipeline(
            query=request.message,
            session_id=session_id,
        )
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal error in RAG pipeline")

    # ── If ticket is complete, save to CSV + fire n8n webhook ───
    ticket_created = False
    if result.get("ticket_data"):
        # 1. Save locally and get the unique ticket number
        ticket_number = save_ticket_to_csv(result["ticket_data"])

        # 2. Inject ticket number into ticket_data so n8n gets it too
        result["ticket_data"]["ticket_number"] = ticket_number

        # 3. Update the answer to show the ticket number to the user
        result["answer"] = (
            f"تمام يا فندم! تم تسجيل طلبك بنجاح ✅\n"
            f"رقم التذكرة: {ticket_number}\n"
            f"الاسم: {result['ticket_data']['name']}\n"
            f"الإيميل: {result['ticket_data']['email']}\n"
            f"الموقع: {result['ticket_data']['location']}\n"
            f"المشكلة: {result['ticket_data']['issue']}\n"
            f"مهندس الدعم الفني هيتواصل معاك في أقرب وقت."
        )

        # 4. Send to n8n (includes ticket_number in payload)
        ticket_created = await send_to_n8n(result["ticket_data"])
        if not ticket_created:
            result["answer"] += "\n\n⚠️ للأسف حصل مشكلة في إرسال إشعار التذكرة، بس رقم التذكرة اتسجل بنجاح."

    # ── Build and return response ───────────────────────────
    return ChatResponse(
        answer=result.get("answer", ""),
        needs_action=result.get("needs_action", "NO") == "YES",
        sources=result.get("sources", []),
        session_id=session_id,
        collecting=result.get("collecting", False),
        ticket_created=ticket_created,
    )


@app.get("/session/{session_id}")
def session_status(session_id: str):
    """
    Optional debug endpoint.
    Returns whether a session is currently mid ticket-collection.
    Useful during development to inspect state.
    """
    from rag import is_collecting_ticket, _ticket_sessions
    collecting = is_collecting_ticket(session_id)
    session_data = _ticket_sessions.get(session_id, {})
    return {
        "session_id": session_id,
        "collecting": collecting,
        "step": session_data.get("step"),
        "fields_collected": {
            "name":     bool(session_data.get("name")),
            "email":    bool(session_data.get("email")),
            "location": bool(session_data.get("location")),
        }
    }