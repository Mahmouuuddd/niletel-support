import os
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

DATA_PATH = "./data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# 1. CHUNKING
# ============================================================
def chunk_text(text):
    """Splits long documents into smaller paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > 700:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# 2. EMBEDDINGS
# ============================================================
def create_embeddings(chunks):
    """Converts text chunks into vector embeddings."""
    print("Creating embeddings...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    print("Embeddings created successfully!")
    print(f"Embedding shape: {embeddings.shape}")

    return model, embeddings


# ============================================================
# 3. FAISS INDEX
# ============================================================
def build_faiss_index(embeddings):
    """Builds FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings) # type: ignore

    print("FAISS index built successfully!")
    print(f"Total vectors: {index.ntotal}")

    return index


# ============================================================
# 4. RETRIEVAL
# ============================================================
def retrieve(query, model, index, chunks, metadata, top_k=6):
    """Retrieves the most relevant chunks for a given query."""

    # Rewrite vague billing queries for better semantic match
    billing_rewrites = _kw.get("billing_rewrites", {})
    search_query = query
    for trigger, rewrite in billing_rewrites.items():
        if trigger in query:
            search_query = rewrite
            break

    print(f"\nSearching for: {search_query}")

    query_emb = model.encode([search_query], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if score > 0.45:
            results.append({
                "text": chunks[idx],
                "source": metadata[idx]["source"],
                "score": float(score)
            })

    print(f"Found {len(results)} results")
    return results


# ============================================================
# 5. ROUTING — improved
# ============================================================

# Load keywords from keywords.json — edit that file, not this one
with open("keywords.json", "r", encoding="utf-8") as _f:
    _kw = json.load(_f)

GREETINGS         = set(_kw["greetings"])
FAREWELLS_THANKS  = set(_kw["farewells_thanks"])
OUT_OF_SCOPE_HARD = set(_kw["out_of_scope"])
TICKET_HARD       = set(_kw["ticket_hard"])


# ============================================================
# 5b. EMBEDDING CLASSIFIER (Layer 1.5)
# ============================================================

class EmbeddingClassifier:
    """
    Classifies queries by finding the closest category example
    using cosine similarity on embeddings.

    Sits between keyword matching (Layer 1) and LLM routing (Layer 2).
    Handles spelling variants, slang, and words not in the keyword sets
    without making any API calls.

    Examples are loaded from keywords.json under "classification_examples".
    """

    def __init__(self, model, examples: dict):
        self.model   = model
        self.labels  = []
        self.vectors = []

        print("Building embedding classifier...")
        for label, phrases in examples.items():
            for phrase in phrases:
                self.labels.append(label)
                self.vectors.append(
                    model.encode(phrase, normalize_embeddings=True)
                )
        self.vectors = np.array(self.vectors, dtype="float32")
        print(f"Classifier ready — {len(self.labels)} examples across {len(examples)} categories")

    def predict(self, query: str, threshold: float = 0.75) -> str | None:
        """
        Returns the predicted category if best similarity >= threshold,
        otherwise returns None so the LLM router takes over.
        """
        q_vec  = self.model.encode(query, normalize_embeddings=True).astype("float32")
        scores = self.vectors @ q_vec
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        print(f"  [Classifier] best={self.labels[best_idx]} score={best_score:.2f}")

        if best_score >= threshold:
            return self.labels[best_idx]
        return None   # not confident → fall through to LLM

    def predict_cancel(self, query: str, threshold: float = 0.80) -> bool:
        """
        Checks if the query expresses cancel intent using embeddings.
        Uses a higher threshold (0.80) than general classification
        because false positives here are costly — we don't want to
        accidentally cancel a real location like "الاسكندرية".
        Returns True if cancel intent is detected, False otherwise.
        """
        q_vec = self.model.encode(query, normalize_embeddings=True).astype("float32")
        scores = self.vectors @ q_vec

        # Only look at scores for cancel-labelled examples
        cancel_scores = [
            scores[i]
            for i, label in enumerate(self.labels)
            if label == "cancel"
        ]
        if not cancel_scores:
            return False

        best_cancel_score = float(max(cancel_scores))
        print(f"  [Classifier] cancel score={best_cancel_score:.2f}")
        return best_cancel_score >= threshold


# Global classifier instance — initialised in init_rag()
classifier: EmbeddingClassifier | None = None


def route_query(query: str) -> str:
    """
    Two-layer routing:
    Layer 1 — keyword match (fast, deterministic, free)
    Layer 2 — LLM classification for ambiguous cases (Groq)

    FIX: ticket check runs FIRST so "صباح الخير ... اعمل تذكرة"
    is never swallowed by the greeting check.
    Greeting/farewell checks are length-guarded (≤5 words) so a
    long message that starts with a greeting still falls through to
    the LLM router.

    Returns: "chat" | "out_of_scope" | "ticket" | "greeting" | "farewell"
    """
    # Normalise hamza variants and ta-marbuta so keywords always match
    # regardless of how the user typed Arabic letters
    q = query.strip().lower()
    q = q.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    q = q.replace("ة", "ه").replace("ى", "ي")

    # Layer 1a: explicit ticket phrases — HIGHEST PRIORITY
    # Must run before greeting check so "صباح الخير، اعمل تذكرة" → ticket
    if any(phrase in q for phrase in TICKET_HARD):
        return "ticket"

    # Layer 1b: hard out-of-scope
    if any(w in q for w in OUT_OF_SCOPE_HARD):  # q is already normalised
        return "out_of_scope"

    # Layer 1c: greetings — only if the message is SHORT (pure greeting)
    # A long message that starts with "صباح الخير" is NOT just a greeting
    if any(g in q for g in GREETINGS) and len(q.split()) <= 5:
        return "greeting"

    # Layer 1d: farewells / thanks — only if short
    if any(t in q for t in FAREWELLS_THANKS) and len(q.split()) <= 5:
        return "farewell"

    # Layer 1.5: embedding classifier — handles slang, typos, variants
    # Only runs when keyword layer had no match
    if classifier is not None:
        result = classifier.predict(query, threshold=0.75)
        if result:
            return result

    # Layer 2: LLM routing for everything else
    return _llm_route(query)


def _llm_route(query: str) -> str:
    """Calls Groq to classify intent for ambiguous queries."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""أنت نظام تصنيف لاستفسارات عملاء شركة اتصالات.

صنّف الاستفسار التالي إلى واحدة فقط من هذه الفئات:
- ticket: العميل يريد رفع شكوى أو إنشاء تذكرة أو إرسال مهندس أو تصعيد رسمي
- out_of_scope: الموضوع لا علاقة له بالاتصالات أو خدمات الشركة تماماً
- chat: أي شيء آخر (استفسار، شكوى، استعلام، إحباط من الخدمة)

مهم جداً:
- "عايز ارفع شكوى" أو "عايز اشتكي" = ticket
- الإحباط أو الشكوى العامة من الخدمة = chat وليس ticket
- ما لم يطلب العميل صراحةً إجراءً رسمياً = chat

الاستفسار: {query}

أجب بكلمة واحدة فقط: ticket أو out_of_scope أو chat"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in ("ticket", "out_of_scope", "chat"):
            return label
        return "chat"
    except Exception:
        return "chat"  # safe fallback


# ============================================================
# 6. TICKET COLLECTION — state machine
# ============================================================

# Each session is stored by session_id.
# Structure:
#   {
#       "step": 0 | 1 | 2 | 3,   # which field we are collecting
#       "issue": str,             # the original complaint the user described
#       "name": str,
#       "email": str,
#       "location": str,
#   }
#
# Steps:
#   0 → ask for name       (bot message: "ممكن اسمك الكريم؟")
#   1 → ask for email      (bot message: "وإيميلك؟")
#   2 → ask for location   (bot message: "وموقعك أو عنوانك؟")
#   3 → all collected → create ticket

_ticket_sessions: dict = {}


def _validate_email(email: str) -> bool:
    """Basic email format check."""
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email.strip()))


def start_ticket_collection(session_id: str, trigger: str) -> dict:
    """
    Called the first time route == 'ticket'.
    Asks the user to describe their issue first — even if they already
    typed something like 'اعمل تذكرة' with no real problem description.

    Steps:
        0 -> ask for issue description
        1 -> ask for name
        2 -> ask for email
        3 -> ask for location  -> build ticket
    """
    _ticket_sessions[session_id] = {
        "step": 0,
        "issue": "",
        "name": "",
        "email": "",
        "location": "",
    }
    return {
        "answer": "تمام يا فندم، هساعدك في إنشاء التذكرة.\nممكن تشرح المشكلة اللي بتواجهها بشكل مختصر؟",
        "needs_action": "NO",
        "sources": [],
        "collecting": True,
    }


def continue_ticket_collection(session_id: str, user_input: str) -> dict:
    """
    Called on every follow-up message while a ticket session is open.
    Advances the state machine one step at a time.
    Returns a question, a validation error, or the final ticket payload.
    """
    session = _ticket_sessions.get(session_id)
    if session is None:
        # Safety net: session expired or missing
        return {
            "answer": "حصل خطأ في الجلسة يا فندم، ممكن تبدأ طلبك من الأول؟",
            "needs_action": "NO",
            "sources": [],
            "collecting": False,
        }

    step = session["step"]

    # ── Normalise Arabic before any matching ──────────────────
    # Users type إ/أ/آ interchangeably with ا, and ة/ه interchangeably.
    # Normalise everything to bare alef + ha so our keyword sets always match.
    def _normalise(text: str) -> str:
        text = text.strip().lower()
        text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
        text = text.replace("ة", "ه")
        text = text.replace("ى", "ي")
        return text

    normalised = _normalise(user_input)

    # ── Cancel detection — two layers ────────────────────────────
    # Layer A: short words loaded from keywords.json → cancel_triggers
    # Too short for embeddings to work reliably on
    HARD_CANCEL    = set(_kw.get("cancel_triggers", []))
    input_words    = set(normalised.split())
    is_hard_cancel = bool(input_words & HARD_CANCEL)

    # Layer B: embedding classifier for longer cancel phrases
    # "مش محتاجش", "خليها", "نسى الموضوع", "وقت تاني" etc.
    is_soft_cancel = (
        not is_hard_cancel and
        classifier is not None and
        classifier.predict_cancel(normalised, threshold=0.80)
    )

    is_cancel = is_hard_cancel or is_soft_cancel
    if is_cancel:
        del _ticket_sessions[session_id]
        return {
            "answer": "تمام يا فندم، تم إلغاء إنشاء التذكرة. لو احتجت أي حاجة تاني أنا هنا.",
            "needs_action": "NO",
            "sources": [],
            "collecting": False,
        }

    # ── Off-topic detection (skip for step 0 — issue can be long) ──
    STEP_LABELS = {
        0: "وصف المشكلة",
        1: "اسمك الكريم",
        2: "إيميلك الإلكتروني",
        3: "موقعك أو عنوانك",
    }
    is_question  = normalised.endswith("؟") or normalised.endswith("?")
    is_out_scope = any(w in normalised for w in OUT_OF_SCOPE_HARD)
    # Only flag long random input for name/email/location — not for issue (step 0)
    is_long_random = len(normalised.split()) > 8 and step in (1, 3)

    if is_question or is_out_scope or is_long_random:
        return {
            "answer": (
                f"يا فندم، إحنا في خطوة إنشاء التذكرة دلوقتي. 😊\n"
                f"محتاج منك {STEP_LABELS[step]} بس علشان نكمل.\n"
                f"لو عايز تلغي التذكرة، قول 'إلغاء'."
            ),
            "needs_action": "NO",
            "sources": [],
            "collecting": True,
        }

    # ── Step 0: save issue description, ask for name ──────────
    if step == 0:
        issue = user_input.strip()
        if len(issue) < 5:
            return {
                "answer": "معلش يا فندم، ممكن تشرح المشكلة بشكل أوضح شوية؟",
                "needs_action": "NO",
                "sources": [],
                "collecting": True,
            }
        session["issue"] = issue
        session["step"] = 1
        return {
            "answer": "تمام، فاهم المشكلة. ممكن اسمك الكريم؟",
            "needs_action": "NO",
            "sources": [],
            "collecting": True,
        }

    # ── Step 1: save name, ask for email ──────────────────────
    if step == 1:
        name = user_input.strip()
        if len(name) < 2:
            return {
                "answer": "معلش يا فندم، الاسم قصير جداً. ممكن تكتب اسمك الكامل؟",
                "needs_action": "NO",
                "sources": [],
                "collecting": True,
            }
        session["name"] = name
        session["step"] = 2
        return {
            "answer": f"شكراً {name}! وإيميلك الإلكتروني؟",
            "needs_action": "NO",
            "sources": [],
            "collecting": True,
        }

    # ── Step 2: save email, ask for location ──────────────────
    if step == 2:
        email = user_input.strip()
        if not _validate_email(email):
            return {
                "answer": "معلش يا فندم، الإيميل مش صحيح. ممكن تكتبه تاني؟ (مثال: name@example.com)",
                "needs_action": "NO",
                "sources": [],
                "collecting": True,
            }
        session["email"] = email
        session["step"] = 3
        return {
            "answer": "تمام! وموقعك أو عنوانك؟ (مثال: المنصورة، شارع الجمهورية)",
            "needs_action": "NO",
            "sources": [],
            "collecting": True,
        }

    # ── Step 3: save location → all collected → fire ticket ───
    if step == 3:
        location = user_input.strip()
        if len(location) < 3:
            return {
                "answer": "ممكن تكتب موقعك بشكل أوضح يا فندم؟",
                "needs_action": "NO",
                "sources": [],
                "collecting": True,
            }
        session["location"] = location
        session["step"] = 4

        # Build the final ticket payload and clean up the session
        ticket_data = {
            "name":     session["name"],
            "email":    session["email"],
            "location": session["location"],
            "issue":    session["issue"],
        }
        del _ticket_sessions[session_id]

        return {
            "answer": (
                f"تمام يا فندم! تم تسجيل طلبك بنجاح ✅\n"
                f"الاسم: {ticket_data['name']}\n"
                f"الإيميل: {ticket_data['email']}\n"
                f"الموقع: {ticket_data['location']}\n"
                f"المشكلة: {ticket_data['issue']}\n"
                f"مهندس الدعم الفني هيتواصل معاك في أقرب وقت."
            ),
            "needs_action": "YES",
            "sources": [],
            "collecting": False,
            "ticket_data": ticket_data,
        }

    # Should never reach here, but just in case
    return {
        "answer": "حصل خطأ غير متوقع يا فندم، ممكن تبدأ من الأول؟",
        "needs_action": "NO",
        "sources": [],
        "collecting": False,
    }


def is_collecting_ticket(session_id: str) -> bool:
    """Returns True if this session is mid-way through ticket collection."""
    return session_id in _ticket_sessions


# ============================================================
# 7. GENERATE ANSWER
# ============================================================
def generate_answer(query, retrieved_results):
    """Generates answer using Groq with strict output format."""
    if not retrieved_results:
        return {
            "answer": "مش متأكد من البيانات المتاحة يا فندم.",
            "needs_action": "NO",
            "sources": []
        }

    context = "\n\n".join(
        f"Source: {res['source']}\n{res['text']}" for res in retrieved_results
    )

    system_message = (
        "أنت مساعد دعم عملاء محترف لشركة NileTel للاتصالات. اسمك 'نيل بوت'.\n"
        "اتبع القواعد الصارمة التالية:\n"
        "- أجب باللهجة المصرية الطبيعية وبلباقة (يا فندم، تمام، هنحلها...).\n"
        "- استخدم فقط المعلومات الموجودة في السياق المقدم. ممنوع التأليف.\n"
        "- إذا كان السياق لا يحتوي على الإجابة، قل: 'مش متأكد من البيانات المتاحة يا فندم.'\n"
        "- لو العميل زعلان أو محبط، ابدأ بجملة تعاطف قصيرة قبل الإجابة.\n"
        "- إذا كان السؤال استفسارًا معلوماتيًا فقط → needs_action: NO.\n"
        "- needs_action: YES فقط لو السياق يثبت إن المشكلة تحتاج تصعيد فعلي "
        "(مثل: تهديد بـ NTRA، outage موثق، مشكلة متكررة بدون حل).\n"
        "- لا تختلق أرقام تذاكر أو تفاصيل وهمية أبداً.\n""- ممنوع منعاً باتاً اختراع رقم تذكرة أو وعد بموعد أو اسم مهندس.\n\n"
        "تعليمات الصيغة — إلزامية تماماً:\n"
        "- السطر الأول يبدأ بـ: answer: ثم نص الإجابة فقط\n"
        "- السطر الثاني يكون: needs_action: YES أو needs_action: NO\n"
        "- ممنوع منعاً باتاً كتابة كلمة answer أو needs_action في أي مكان آخر في الرد\n"
        "- لا توجد أسطر إضافية، لا مقدمات، لا تعليقات"
    )

    user_message = f"السياق المتاح (استخدمه فقط):\n{context}\n\nالسؤال: {query}"

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=800,
        temperature=0.2
    )

    text = response.choices[0].message.content.strip()

    # Primary extraction: strict format "answer: ...\nneeds_action: YES/NO"
    pattern = r"^answer:\s*(.*?)\s*needs_action:\s*(YES|NO)\s*$"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        needs_action = match.group(2).upper()
    else:
        # Fallback: extract needs_action anywhere
        action_match = re.search(r"needs_action:\s*(YES|NO)", text, re.IGNORECASE)
        needs_action = action_match.group(1).upper() if action_match else "NO"

        # Extract text after first "answer:" only
        answer_match = re.search(r"^answer:\s*(.+?)(?=\s*needs_action:|$)", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Last resort: strip both tags from full text
            answer = re.sub(r"needs_action:\s*(YES|NO)", "", text, flags=re.IGNORECASE)
            answer = re.sub(r"answer:\s*", "", answer, flags=re.IGNORECASE).strip()

    # Clean any leaked fallback phrase the LLM may have inserted
    answer = re.sub(r"مش متأكد من البيانات المتاحة يا فندم[،.]?\s*", "", answer).strip()

    sources = [res["source"] for res in retrieved_results]

    return {
        "answer": answer,
        "needs_action": needs_action,
        "sources": sources
    }


# ============================================================
# 8. RUN RAG PIPELINE
# ============================================================
def run_rag_pipeline(query: str, session_id: str = "default"):
    """
    Main entry point called by FastAPI.

    New parameter: session_id
      - Every user/chat session gets a unique ID (FastAPI generates it).
      - If a ticket collection is already in progress for this session,
        we continue the flow instead of re-routing from scratch.
    """

    # ── If we are mid-collection, continue the state machine ──
    if is_collecting_ticket(session_id):
        return continue_ticket_collection(session_id, query)

    # ── Guard: reject empty or very short gibberish inputs ─────
    if len(query.strip()) < 2:
        return {
            "answer": "مش فاهم سؤالك يا فندم، ممكن تكتب أكتر؟",
            "needs_action": "NO", "sources": []
        }

    # ── Otherwise, route normally ──────────────────────────────
    route = route_query(query)
    print(f"Route: {route}")

    if route == "greeting":
        return {
            "answer": "أهلاً وسهلاً بيك يا فندم! معاك خدمة عملاء NileTel. أقدر أساعدك في إيه النهاردة؟",
            "needs_action": "NO", "sources": []
        }

    if route == "farewell":
        return {
            "answer": "العفو يا فندم! يسعدنا خدمتك دايماً. لو احتجت أي حاجة تاني، إحنا هنا.",
            "needs_action": "NO", "sources": []
        }

    if route == "out_of_scope":
        return {
            "answer": "آسف يا فندم، مش هقدر أساعدك في الموضوع ده. أنا متخصص في دعم عملاء NileTel.",
            "needs_action": "NO", "sources": []
        }

    if route == "ticket":
        # Don't create the ticket yet — start collecting user info first
        return start_ticket_collection(session_id, trigger=query)

    # route == "chat" → RAG
    results = retrieve(query, model, index, all_chunks, metadata, top_k=6)
    response = generate_answer(query, results)

    # Only override if LLM hallucinated YES for a clearly informational query
    ALWAYS_NO_PATTERNS = ["ازاي", "إيه فايدة", "هل ده", "عايز اعرف", "ليه الفاتورة"]
    if any(p in query for p in ALWAYS_NO_PATTERNS):
        response["needs_action"] = "NO"

    # Handle vague query with no real context match, empty answer, or hallucination
    no_context   = not results
    empty_answer = not response.get("answer", "").strip()
    unsure       = response["answer"] == "مش متأكد من البيانات المتاحة يا فندم."
    if no_context or empty_answer or unsure:
        response["answer"] = (
            "آسف يا فندم، مش هقدر أساعدك في الموضوع ده. "
            "أنا متخصص في دعم عملاء NileTel فقط."
        )
        response["needs_action"] = "NO"

    return response


# ============================================================
# 9. MAIN — module-level init so FastAPI can import safely
# ============================================================

# These are initialised once at import time so run_rag_pipeline
# can reference them whether called from __main__ or from FastAPI.
all_chunks = []
metadata   = []
model      = None
index      = None
classifier = None


def init_rag():
    """
    Loads documents, builds embeddings and FAISS index.
    Called once at startup (either here in __main__ or in FastAPI lifespan).
    """
    global all_chunks, metadata, model, index, classifier

    for file in os.listdir(DATA_PATH):
        if file.endswith(".md"):
            with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
                text = f.read()
            doc_chunks = chunk_text(text)
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                metadata.append({"source": file})

    model, embeddings = create_embeddings(all_chunks)
    index = build_faiss_index(embeddings)

    # Build embedding classifier using classification_examples from keywords.json
    global classifier
    examples = _kw.get("classification_examples", {})
    if examples:
        classifier = EmbeddingClassifier(model, examples)
    else:
        print("⚠️ No classification_examples found in keywords.json — Layer 1.5 disabled")

    print("\n=== RAG System Ready! ===\n")


if __name__ == "__main__":
    init_rag()

    test_queries = [
        # original tests
        ("ازاي أحل مشكلة 5G throttling؟",          "default"),
        ("النت مقطوع تماماً في المنصورة، اعمل تذكرة", "default"),
        ("ازيك",                                      "s2"),
        ("إيه رأيك في فيلم ريش؟",                    "s3"),
        ("عايز أطلب بيتزا",                          "s4"),
        # new: mixed greeting + ticket (the bug we fixed)
        ("صباح الخير النت مقطوع تماماً، اعمل تذكرة", "s5"),
        # multi-turn ticket collection simulation
        ("اعمل تذكرة عشان النت بطيء",               "ticket_session"),
        ("أحمد محمد",                                "ticket_session"),   # name
        ("ahmed@example.com",                        "ticket_session"),   # email
        ("المنصورة، شارع الجمهورية",                 "ticket_session"),   # location
    ]

    for q, sid in test_queries:
        print(f"\nTesting query: {q}  [session={sid}]")
        response = run_rag_pipeline(q, session_id=sid)
        if response:
            print("Answer:", response.get("answer", "")[:300])
            print("Needs Action:", response.get("needs_action", ""))
            print("Collecting:", response.get("collecting", False))
            if response.get("ticket_data"):
                print("Ticket Data:", response["ticket_data"])
        print("-" * 80)