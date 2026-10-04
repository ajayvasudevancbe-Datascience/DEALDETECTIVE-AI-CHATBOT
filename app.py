import html
import json
import os
import re
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="DealDetective AI", page_icon="🔎", layout="wide")

SYSTEM_PROMPT = """
You are DealDetective AI, a smart shopping assistant.

Analyze products using available price, discount, rating, reviews, features,
availability, and comparisons. Never invent product information, prices,
discounts, ratings, review counts, or availability.

For product or deal evaluations, format the response with these fields, each on
its own line:

**Product:** ...
**Current Price:** ...
**Discount:** ...
**Rating:** ...
**Reviews:** ...
**Availability:** ...
**Deal Score:** N/100 or Not enough information
**Verdict:** BUY, WAIT, AVOID, or ALTERNATIVE
**Deal Analysis:** Explain the recommendation clearly and briefly.

Use "Not provided" for unavailable product details. Give a numeric deal score
only when there is enough evidence to assess the deal; otherwise say
"Not enough information". Base the score on the evidence, not an impression.
Use ALTERNATIVE only when suggesting a specific better-fit alternative.
"""

MODELS = {
    "Qwen2.5 72B Instruct (recommended)": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
    "Qwen2.5 32B Instruct (faster)": "Qwen/Qwen2.5-32B-Instruct",
}

RUNTIME_NOTES = """
## RUNTIME NOTES
- You have NO internet access yourself. Live web search results are supplied in the
  user message under "LIVE SEARCH RESULTS". Use ONLY those for prices, ratings, reviews
  and availability. Never use memory for prices.
- Quote the source URL and the observation date for every price you report.
- If the results do not contain a price, say "I could not verify the current price."
- Default currency is INR (₹) and market is India unless the user says otherwise.
- Follow the RESPONSE FORMAT exactly for single products. Keep answers concise.
"""


# ---------- helpers ----------
def get_token() -> str:
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        st.error("HF_TOKEN is not set. Add it in Streamlit secrets.")
        st.stop()
    return token


def load_system_prompt() -> str:
    return SYSTEM_PROMPT + "\n" + RUNTIME_NOTES


def chat(model: str, messages: list, max_tokens: int = 2000, temperature: float = 0.3) -> str:
    client = InferenceClient(model=model, token=get_token())
    out = client.chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return out.choices[0].message.content


def make_queries(model: str, user_msg: str) -> list[str]:
    """Ask the model for up to 3 focused web-search queries."""
    prompt = (
        "Turn the shopping request into at most 3 short web search queries for finding "
        "current price in India, customer reviews/problems, and price history or "
        "alternatives. Reply ONLY with a JSON list of strings. If the message needs no "
        "product research, reply [].\n\nRequest: " + user_msg
    )
    try:
        raw = chat(model, [{"role": "user", "content": prompt}], max_tokens=200, temperature=0)
        match = re.search(r"\[.*\]", raw, re.S)
        queries = json.loads(match.group(0)) if match else []
        return [q for q in queries if isinstance(q, str)][:3]
    except Exception:
        return [user_msg]


def web_search(queries: list[str]) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "Search library not installed."

    blocks = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, region="in-en", max_results=6)
            except Exception as e:
                blocks.append(f"[Search failed for '{q}': {e}]")
                continue
            lines = [f"### Query: {q}"]
            for r in results:
                lines.append(f"- {r.get('title')} | {r.get('href')}\n  {r.get('body')}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No search results."


# ---------- UI ----------
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --navy: #071A2B;
        --blue: #0D2A43;
        --cyan: #00D4FF;
        --electric: #3B82F6;
        --text: #FFFFFF;
        --muted: #B8CCE0;
        --border: rgba(184, 204, 224, 0.14);
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #071A2B;
        color: #FFFFFF;
    }
    [data-testid="stHeader"] { background: rgba(7, 26, 43, 0.92); }
    [data-testid="stMainBlockContainer"] {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }
    [data-testid="stSidebar"] {
        background: #0D2A43;
        border-right: 1px solid rgba(0, 212, 255, 0.16);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: #FFFFFF;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(0, 212, 255, 0.22);
        border-radius: 18px;
        background: #0D2A43;
        padding: 1rem 1.15rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #103450;
        border-color: rgba(59, 130, 246, 0.32);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #0D2A43;
        border-color: rgba(0, 212, 255, 0.27);
    }
    [data-testid="stChatInput"] {
        background: #0D2A43;
        border: 1px solid rgba(184, 204, 224, 0.25);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #00D4FF;
        box-shadow: 0 0 0 1px #00D4FF, 0 10px 30px rgba(0, 0, 0, 0.22);
    }
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF;
        -webkit-text-fill-color: #FFFFFF;
        padding-top: 0.85rem;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #B8CCE0;
        -webkit-text-fill-color: #B8CCE0;
    }
    [data-testid="stChatInput"] button {
        color: #071A2B;
        background: #00D4FF;
        border-radius: 12px;
    }
    [data-testid="stChatInput"] button:hover { background: #67E5FF; }
    [data-testid="stBaseButton-primary"] {
        background: #00D4FF;
        color: #071A2B;
        border: 0;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: #67E5FF;
        color: #071A2B;
        border: 0;
    }
    [data-testid="stBaseButton-secondary"] {
        background: #0D2A43;
        color: #B8CCE0;
        border: 1px solid rgba(184, 204, 224, 0.16);
        border-radius: 14px;
        min-height: 3.5rem;
        text-align: left;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        color: #FFFFFF;
        border-color: #00D4FF;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] input,
    [data-testid="stExpander"] {
        background: #071A2B;
        color: #FFFFFF;
        border-color: rgba(184, 204, 224, 0.18);
    }
    [data-testid="stCaptionContainer"] { color: #B8CCE0; }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #3B82F6, #00D4FF);
    }
    .dd-header {
        padding: 1.15rem 1.4rem;
        margin: 0 0 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.22);
        border-radius: 20px;
        background: linear-gradient(115deg, #0D2A43, #0A2238);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }
    .dd-header-title {
        color: #FFFFFF;
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .dd-header-subtitle {
        color: #B8CCE0;
        margin: 0.3rem 0 0;
    }
    .welcome-title {
        text-align: center;
        font-size: clamp(1.6rem, 5vw, 2.15rem);
        font-weight: 650;
        letter-spacing: -0.03em;
        margin: 3.5rem 0 0.45rem;
        color: #FFFFFF;
    }
    .welcome-subtitle {
        text-align: center;
        color: #B8CCE0;
        margin-bottom: 1.8rem;
    }
    .deal-card {
        background: rgba(7, 26, 43, 0.68);
        border: 1px solid rgba(184, 204, 224, 0.13);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        min-height: 78px;
        margin: 0.3rem 0;
    }
    .deal-label {
        color: #B8CCE0;
        font-size: 0.78rem;
        margin-bottom: 0.28rem;
    }
    .deal-value {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }
    .verdict-card {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 0.2rem 0 0.9rem;
        background: rgba(0, 212, 255, 0.07);
        border: 1px solid rgba(0, 212, 255, 0.32);
    }
    .verdict-buy { border-color: rgba(34, 197, 94, 0.55); background: rgba(34, 197, 94, 0.09); }
    .verdict-wait { border-color: rgba(245, 158, 11, 0.55); background: rgba(245, 158, 11, 0.09); }
    .verdict-avoid { border-color: rgba(239, 68, 68, 0.55); background: rgba(239, 68, 68, 0.09); }
    .verdict-alternative { border-color: rgba(0, 212, 255, 0.55); background: rgba(0, 212, 255, 0.09); }
    .verdict-buy .deal-value { color: #4ADE80; }
    .verdict-wait .deal-value { color: #FBBF24; }
    .verdict-avoid .deal-value { color: #F87171; }
    .verdict-alternative .deal-value { color: #00D4FF; }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 0.8rem 0.8rem 6rem;
        }
        [data-testid="stChatMessage"] {
            padding: 0.8rem;
            border-radius: 15px;
        }
        .dd-header { padding: 1rem; border-radius: 16px; }
        .welcome-title { margin-top: 2.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dd-header">
        <div class="dd-header-title">🕵️ DealDetective AI</div>
        <div class="dd-header-subtitle">Find the deal. Check the value. Buy smarter.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🕵️ DealDetective")
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    with st.expander("⚙️ Chat settings", expanded=True):
        model_label = st.selectbox("Model", list(MODELS))
        model = MODELS[model_label]
        use_search = st.toggle(
            "Live web search",
            value=True,
            help="Turn off only if you don't need current prices.",
        )
    st.caption("Powered by Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome-title">What are you shopping for?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="welcome-subtitle">Compare the details, inspect the deal, and decide with confidence.</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "Is the boAt Airdopes 141 at ₹1,299 worth it?",
        "Best laptop under ₹50,000 for programming",
        "Is this 70% discount real?",
        "Compare these two products for me",
    ]
    columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with columns[index % 2]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.pending = suggestion

def render_assistant_message(content: str) -> None:
    field_names = {
        "Product": "📦 Product",
        "Current Price": "💰 Price",
        "Discount": "🏷️ Discount",
        "Rating": "⭐ Rating",
        "Reviews": "💬 Reviews",
        "Availability": "📦 Availability",
    }
    field_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?\*\*"
        r"(Product|Current Price|Discount|Rating|Reviews|Availability|Deal Score|Verdict|Deal Analysis)"
        r"\s*:\*\*\s*(.*?)\s*$",
        re.IGNORECASE,
    )
    fields: dict[str, str] = {}
    remaining_lines = []
    canonical_names = {name.lower(): name for name in (*field_names, "Deal Score", "Verdict", "Deal Analysis")}

    for line in content.splitlines():
        match = field_pattern.match(line)
        if match:
            name = canonical_names[match.group(1).lower()]
            fields[name] = match.group(2).strip()
        else:
            remaining_lines.append(line)

    if not fields:
        st.markdown(content)
        return

    verdict = fields.get("Verdict", "")
    verdict_match = re.search(r"\b(BUY|WAIT|AVOID|ALTERNATIVE)\b", verdict, re.IGNORECASE)
    verdict_key = verdict_match.group(1).lower() if verdict_match else "default"
    if verdict:
        st.markdown(
            (
                f'<div class="verdict-card verdict-{verdict_key}">'
                f'<div class="deal-label">🕵️ Deal Analysis · Verdict</div>'
                f'<div class="deal-value">{html.escape(verdict)}</div></div>'
            ),
            unsafe_allow_html=True,
        )

    metrics = [(name, fields[name]) for name in field_names if name in fields]
    if metrics:
        columns = st.columns(3)
        for index, (name, value) in enumerate(metrics):
            label = field_names[name]
            with columns[index % len(columns)]:
                st.markdown(
                    (
                        '<div class="deal-card">'
                        f'<div class="deal-label">{html.escape(label)}</div>'
                        f'<div class="deal-value">{html.escape(value)}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

    score_text = fields.get("Deal Score", "")
    score_match = re.search(r"\b(100|[1-9]?\d)\s*/\s*100\b", score_text)
    if score_match:
        score = int(score_match.group(1))
        st.markdown(f"**🕵️ DealDetective Score** · {score}/100")
        st.progress(score / 100)
    elif score_text:
        st.caption(f"🕵️ DealDetective Score: {score_text}")

    deal_analysis = fields.get("Deal Analysis")
    if deal_analysis:
        st.markdown(f"**🕵️ Deal Analysis**  \n{deal_analysis}")
    response_body = "\n".join(remaining_lines).strip()
    if response_body:
        st.markdown(response_body)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant_message(m["content"])
        else:
            st.markdown(m["content"])

user_input = st.chat_input("Ask me about a product, price, deal or comparison...")
if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            context = ""
            if use_search:
                with st.spinner("Searching the web..."):
                    queries = make_queries(model, user_input)
                    if queries:
                        results = web_search(queries)
                        now = datetime.now().strftime("%d %b %Y, %H:%M")
                        context = (
                            f"\n\n---\nLIVE SEARCH RESULTS (retrieved {now}):\n{results}\n---\n"
                            "Analyse using only the information above."
                        )

            # keep history short to save tokens; attach search results to latest turn only
            history = st.session_state.messages[-7:-1]
            api_messages = (
                [{"role": "system", "content": load_system_prompt()}]
                + history
                + [{"role": "user", "content": user_input + context}]
            )
            with st.spinner("Investigating the deal..."):
                answer = chat(model, api_messages)
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        render_assistant_message(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})