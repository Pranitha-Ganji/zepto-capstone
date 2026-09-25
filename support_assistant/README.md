# Module 3 — Support Assistant (`/support_assistant`)
**Zepto Grounded GenAI Customer Support Assistant**

---

## 1. Overview & Business Context
In quick commerce, customer trust relies on transparent, rapid, and grounded communication regarding orders, delivery timelines, cancellation rules, and refund policies. The Zepto Support Assistant is an enterprise-grade GenAI service featuring:
- **Corpus Ingestion & Local Embeddings:** All 8 verified policy documents embedded locally via `all-MiniLM-L6-v2` into a persistent `ChromaDB` vector index (100% keyless and offline).
- **LangGraph Orchestrated Router & RAG:** A stateful `StateGraph` routing queries between dense policy retrieval and direct answers.
- **Deterministic Mock Baseline (Default):** Runs fully offline and deterministically without external API calls or keys (`MOCK_LLM=1`), ensuring 100% reproducible grading.
- **Structured Schema Enforcement:** Pydantic validation ensuring strict output schema (`answer`, `sources`, `confidence`) with corrective retry loops.
- **Production REST API & Docker Container:** Wrapped in FastAPI serving `POST /ask`, packaged in a self-contained Docker container.

---

## 2. RAG Pipeline Architecture & Data Flow

```
                                  +---------------------------------------+
                                  |         User Request (POST /ask)      |
                                  |         {"query": "..."}              |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         LangGraph Entry Point         |
                                  |           classify_intent             |
                                  +---------------------------------------+
                                                      |
                                 /-----------------------------------------\
                                /    Conditional Routing (route_intent)     \
                               \                                           /
                                \-----------------------------------------/
                                         /                         \
           [intent == 'policy_question']                           [intent == 'general_question']
                                       /                             \
                                      v                               v
        +------------------------------------------+    +------------------------------------------+
        |         retrieve_and_answer              |    |              direct_answer               |
        |------------------------------------------|    |------------------------------------------|
        | 1. Query Embed (all-MiniLM-L6-v2)        |    | Mock Mode (MOCK_LLM=1):                  |
        | 2. Dense Cosine Retrieval (ChromaDB)     |    | Canned policy-boundary response          |
        | 3. Top-3 Policy Chunks Retrieved         |    |                                          |
        | 4. Generation:                           |    | Optional Real LLM (MOCK_LLM=0):          |
        |    - Mock: Canned snippet template       |    | Polite conversational guidance           |
        |    - Real: Grounded LLM generation       |    +------------------------------------------+
        +------------------------------------------+                                  |
                                       \                                              /
                                        \--------------------------------------------/
                                                               |
                                                               v
                                              +----------------------------------+
                                              |       Pydantic Schema Validator   |
                                              |      AskResponse (answer,        |
                                              |      sources, confidence)        |
                                              +----------------------------------+
                                                               |
                                                               v
                                              +----------------------------------+
                                              |        Final JSON Output         |
                                              +----------------------------------+
```

### Stage-by-Stage Architecture Breakdown

| Stage | Responsible Component / File / Node | Mechanism & Data Flow | Mock vs Real LLM (`MOCK_LLM`) Behavior |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | `support_assistant/ingest.py` (`load_documents`, `ingest_corpus`) | Reads 8 policy documents (`doc_01.txt` - `doc_08.txt`) from `support_assistant/docs/`. Generates chunk metadata and IDs. | Identical in both modes (runs locally, no network). |
| **2. Embedding & Storage** | `support_assistant/ingest.py` (`SentenceTransformer`, `ChromaDB`) | Embeds chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2` and persists them into ChromaDB collection `zepto_policies` using cosine similarity (`hnsw:space: cosine`). | Identical in both modes (100% local, no API keys). |
| **3. Intent Routing** | `support_assistant/agent.py` (`classify_intent`, `route_intent`) | Inspects user query to determine whether policy grounding is necessary. | **Mock Mode (`MOCK_LLM=1`):** Fast keyword heuristic checking `["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]`.<br>**Real Mode (`MOCK_LLM=0`):** LLM intent classification prompt. |
| **4. Retrieval** | `support_assistant/agent.py` (`retrieve_and_answer`) | Encodes query and queries ChromaDB for top-3 nearest neighbor chunks via cosine similarity. | **Always runs for real in both modes.** Cosine retrieval is identical whether mock or real LLM is used. |
| **5. Generation** | `support_assistant/agent.py` (`retrieve_and_answer` / `direct_answer`) | Synthesizes response from retrieved context or handles out-of-scope inquiry. | **Mock Mode (`MOCK_LLM=1`):** Deterministic canned template: `f"Based on the retrieved context: {top_chunk_snippet}"` for policy queries, or `"I can only answer questions about Zepto policies right now."` for general.<br>**Real Mode (`MOCK_LLM=0`):** Calls real LLM (e.g. Groq free tier) with structured prompt template and retries up to 2 times on validation error. |
| **6. Schema Validation** | `support_assistant/agent.py` & `main.py` (`AskResponse`) | Enforces Pydantic schema: `answer` (str), `sources` (list[str]), `confidence` (float 0.0-1.0). | In Mock Mode, populated deterministically from graph state. In Real Mode, validated against LLM JSON output. |

---

## 3. Structured Prompt Template
The agent implements an industry-standard prompt following the **Role–Context–Task–Constraints–Format–Length–FewShot** skeleton:

```text
### ROLE
You are Zepto's AI Policy & Support Assistant. You provide accurate, grounded, and polite answers to customer questions regarding Zepto's delivery, returns, refunds, membership, tracking, cancellations, and support policies.

### CONTEXT
The following verified Zepto policy documents have been retrieved:
{context}

### TASK
Answer the customer query below based solely on the provided policy documents.
Customer Query: {query}

### CONSTRAINTS (NEGATIVE CONSTRAINT)
- Do NOT answer using any external knowledge, assumptions, or information not explicitly present in the provided context. If the context does not contain enough information to answer the question, state: "I do not have enough information from Zepto's verified policies to answer this question."
- Do NOT hallucinate policy terms, fees, or timelines.

### FORMAT
Provide a concise, direct response formatted as a strict JSON object adhering to this schema:
{
  "answer": "<grounded response text>",
  "sources": ["<document id, e.g., doc_01>"],
  "confidence": <confidence score between 0.0 and 1.0>
}

### LENGTH
Keep the answer concise and direct (under 100 words).

### FEW-SHOT EXAMPLE
Example 1:
Context:
[doc_01] Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Query: What is the delivery fee for orders below 149?
Response:
{
  "answer": "Standard delivery is free on orders over INR 149, while orders below INR 149 incur a flat delivery fee of INR 25.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

---

## 4. Example API Calls & Responses (Default Mock Mode)

### Example 1: Policy Question (Triggers Retrieval)
**Request:**
```bash
curl -X POST "http://127.0.0.1:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the return policy for grocery and perishable items?"}'
```

**Response (Raw JSON):**
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened,",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_01"
  ],
  "confidence": 1.0
}
```
*Verification:*
- Intent routed: `policy_question`
- Retrieval executed: Yes (top source `doc_02` directly addresses Returns & Refunds).
- Canned snippet returned matching `doc_02` text.

### Example 2: General / Out-of-Scope Question (Direct Answer)
**Request:**
```bash
curl -X POST "http://127.0.0.1:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

**Response (Raw JSON):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
*Verification:*
- Intent routed: `general_question`
- Retrieval bypassed: Yes (`sources` is empty).
- Response conforms strictly to Pydantic schema.

---

## 5. Local Execution & Dockerization

### Run Service Locally
```bash
# Ingest documents into ChromaDB
python -m support_assistant.ingest

# Launch FastAPI service
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

### Build & Run with Docker
```bash
# Build Docker image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

# Run container locally
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```
The endpoint is available at `http://localhost:7860/ask`.
