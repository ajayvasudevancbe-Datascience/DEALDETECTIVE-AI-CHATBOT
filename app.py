import json
import os
import re
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="DealDetective AI", page_icon="🔎", layout="wide")

SYSTEM_PROMPT = """
You are DealDetective AI, a smart shopping assistant.

Analyze products based on price, discount, rating, reviews, and features.

Give a simple recommendation:
BUY, WAIT, or AVOID.

Explain your recommendation clearly and briefly.

Never invent product information, prices, or reviews.
If information is missing, say so.
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
    .stApp {
        background: #ffffff;
        color: #202123;
    }
    [data-testid="stSidebar"] {
        background: #f9f9f9;
        border-right: 1px solid #ececec;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }
    [data-testid="stChatMessage"] {
        border: 0;
        padding: 1rem 0.5rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 1.5rem;
        border: 1px solid #d9d9e3;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stChatInput"] textarea {
        padding-top: 0.8rem;
    }
    .welcome-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        margin: 5rem 0 0.5rem;
        color: #202123;
    }
    .welcome-subtitle {
        text-align: center;
        color: #6e6e80;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🔎 DealDetective")
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
    st.markdown('<div class="welcome-title">What are you shopping for?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-subtitle">Check a deal, compare products, or find your next buy.</div>',
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

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Message DealDetective...")
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
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})