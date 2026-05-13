# 📡 NileTel — AI Customer Support System

A full-stack AI-powered customer support chatbot for a fictional Egyptian telecom company **NileTel**.  
Built with RAG (Retrieval-Augmented Generation), FastAPI, Streamlit, and n8n automation.

---

## 🎯 What This System Does

- Answers customer questions about internet and telecom services using a knowledge base of `.md` documents
- Automatically creates support tickets by collecting name, email, and location from the customer
- Sends a confirmation email to the customer via Gmail (through n8n)
- Logs every ticket to Google Sheets and a local CSV file
- Routes queries intelligently using a 3-layer hybrid system

---

## 🏗️ System Architecture

```
User types in Streamlit
        ↓
Streamlit → ngrok → FastAPI (port 8000)
        ↓
3-Layer Router:
  Layer 1   — Keyword matching (instant, free)
  Layer 1.5 — Embedding classifier (semantic, local)
  Layer 2   — LLM router via Groq (intelligent, fallback)
        ↓
  ┌─────────────────────────────────────┐
  │  greeting   → welcome message       │
  │  farewell   → goodbye message       │
  │  out_of_scope → polite refusal      │
  │  ticket     → collect info (4 steps)│
  │  chat       → RAG pipeline          │
  └─────────────────────────────────────┘
        ↓ (if ticket)
FastAPI → n8n webhook
        ↓
  ┌─────────────────────────────────┐
  │  Google Sheets (log ticket)     │
  │  Gmail (send confirmation)      │
  │  Local CSV (backup log)         │
  └─────────────────────────────────┘
        ↓
Answer returned to Streamlit ✅
```

---

## 🧠 RAG Pipeline

```
Documents (.md files)
        ↓
Chunking (700 char paragraphs)
        ↓
Embeddings (intfloat/multilingual-e5-large)
        ↓
FAISS Index (IndexFlatIP = cosine similarity)
        ↓
Query → embed → search → top 6 chunks
        ↓
Groq LLM (llama-3.1-8b-instant) generates answer
```

---

## 🎫 Ticket Collection Flow (Multi-turn)

```
User: "اعمل تذكرة"
Bot:  "ممكن تشرح المشكلة؟"         ← Step 0: issue
User: "النت بطيء من أسبوع"
Bot:  "ممكن اسمك الكريم؟"           ← Step 1: name
User: "أحمد محمد"
Bot:  "وإيميلك الإلكتروني؟"         ← Step 2: email
User: "ahmed@example.com"
Bot:  "وموقعك أو عنوانك؟"           ← Step 3: location
User: "المنصورة، شارع الجمهورية"
Bot:  "تم تسجيل طلبك ✅ رقم التذكرة: NTL-20260509-0001"
        ↓
FastAPI → n8n → Gmail + Google Sheets + tickets.csv
```

---

## 📁 Project Structure

```
niletel-support/
│
├── rag.py              # RAG pipeline, routing, ticket state machine
├── main.py             # FastAPI backend
├── app.py              # Streamlit frontend
├── keywords.json       # All keywords and classifier examples (edit this, not the code)
│
├── data/               # Knowledge base documents
│   ├── 5g_throttling_troubleshooting.md
│   ├── faq_5g_not_connecting.md
│   └── ... (all .md files)
│
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Files excluded from git
└── README.md           # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/niletel-support.git
cd niletel-support
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in your keys
GROQ_API_KEY=your_groq_api_key_here
N8N_WEBHOOK_URL=https://your-username.app.n8n.cloud/webhook/chat
```

### 4. Add your knowledge base

Place your `.md` documents inside the `data/` folder.

### 5. Run the system

Open **3 terminals**:

```bash
# Terminal 1 — FastAPI backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app.py

# Terminal 3 — ngrok tunnel (to expose FastAPI publicly)
ngrok http 8000
```

### 6. Update Streamlit with ngrok URL

Open `app.py` and update line 7:
```python
FASTAPI_URL = "https://your-ngrok-url.ngrok-free.app/chat"
```

### 7. Open the chat UI

```
http://localhost:8501
```

---

## 🔑 Getting API Keys

| Service | Where to get it | Free tier |
|---------|----------------|-----------|
| Groq API | [console.groq.com](https://console.groq.com) | ✅ Yes |
| ngrok | [dashboard.ngrok.com](https://dashboard.ngrok.com) | ✅ Yes |
| n8n | [n8n.io](https://n8n.io) | ✅ Yes (14 days trial, then free plan) |

---

## 🔧 n8n Workflow Setup

Your n8n workflow should have these 4 nodes in order:

```
Webhook → Google Sheets → Gmail → Respond to Webhook
```

**Webhook node:**
- Method: `POST`
- Path: `chat`

**Google Sheets node — field mapping:**
```
Ticket Number → {{ $('Webhook').item.json.body.ticket_number }}
Name          → {{ $('Webhook').item.json.body.name }}
Email         → {{ $('Webhook').item.json.body.email }}
Location      → {{ $('Webhook').item.json.body.location }}
Issue         → {{ $('Webhook').item.json.body.issue }}
Timestamp     → {{ $now }}
```

**Gmail node:**
```
To      → {{ $('Webhook').item.json.body.email }}
Subject → [{{ $('Webhook').item.json.body.ticket_number }}] تأكيد استلام طلب الدعم — NileTel
```

---

## 📊 keywords.json Structure

All routing keywords and classifier examples live in `keywords.json`.  
**Edit this file to add new keywords — no Python code changes needed.**

```json
{
  "greetings":         [...],   // greeting phrases
  "farewells_thanks":  [...],   // farewell/thanks phrases
  "out_of_scope":      [...],   // off-topic keywords
  "ticket_hard":       [...],   // explicit ticket request phrases
  "cancel_triggers":   [...],   // short cancel words (HARD_CANCEL)
  "billing_rewrites":  {...},   // query rewriting for billing questions
  "classification_examples": {  // embedding classifier training examples
    "greeting":     [...],
    "farewell":     [...],
    "ticket":       [...],
    "out_of_scope": [...],
    "chat":         [...],
    "cancel":       [...]        // longer cancel phrases for classifier
  }
}
```

---

## 🧪 Test Cases

| Input | Expected Route | Expected Behavior |
|-------|---------------|-------------------|
| `ازيك` | greeting | Welcome message |
| `اعمل تذكرة` | ticket | Starts 4-step collection |
| `صباح الخير، اعمل تذكرة` | ticket | Ticket wins over greeting |
| `ازاي أحل مشكلة 5G؟` | chat | RAG answer |
| `قولي نكتة` | out_of_scope | Polite refusal |
| `عايز بيتزا` | out_of_scope | Polite refusal |
| `شكراً جزيلاً` | farewell | Goodbye message |
| `الغاء` (during ticket) | — | Cancels ticket flow |

---

## 🚀 Key Features & Improvements

### 3-Layer Hybrid Router
- **Layer 1** — Keyword matching: instant, zero cost, handles obvious cases
- **Layer 1.5** — Embedding classifier: semantic understanding, handles typos and slang without API calls
- **Layer 2** — LLM (Groq/LLaMA): handles truly ambiguous queries

### Smart Ticket Collection
- Multi-turn conversation state machine
- Validates email format
- Detects off-topic inputs mid-flow and redirects
- Cancel detection with two layers (hard words + semantic)
- Unique ticket numbers: `NTL-YYYYMMDD-XXXX`

### Arabic NLP
- Full Arabic normalisation (hamza, ta-marbuta, alef variants)
- Multilingual embedding model (`intfloat/multilingual-e5-large`)
- Egyptian dialect support in responses

### Data Management
- Local CSV backup (`tickets.csv`) with UTF-8-sig encoding for Excel
- Google Sheets logging via n8n
- All keywords externalized to `keywords.json`

---

## 📝 Summary

### What worked well
- The 3-layer routing system is robust and handles a wide variety of Arabic inputs
- The embedding classifier catches spelling variants and slang without any keyword lists
- The multi-turn ticket collection flow works smoothly with proper validation
- Externalizing keywords to JSON makes the system easy to maintain

### Challenges faced
- Arabic text normalization (hamza/ta-marbuta variants) caused false negatives in keyword matching
- Short words like "لا" caused false positives in cancel detection — solved with word boundary matching
- Mixed greeting + ticket queries ("صباح الخير، اعمل تذكرة") required priority ordering in the router

### What to improve next
- Add authentication to the FastAPI endpoints
- Persist ticket sessions to a database instead of in-memory dict
- Add a proper admin dashboard to view and manage tickets
- Implement hybrid retrieval (BM25 + semantic) with RRF for better RAG results
- Deploy to a cloud server so ngrok is not needed

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | intfloat/multilingual-e5-large |
| Vector Search | FAISS |
| Automation | n8n |
| Tunnel | ngrok |
| Email | Gmail (via n8n) |
| Logging | Google Sheets + Local CSV |