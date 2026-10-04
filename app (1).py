import os
from pathlib import Path

import anthropic
import streamlit as st

st.set_page_config(page_title="DealDetective AI", page_icon="🔎", layout="centered")

MODEL = os.getenv("DEALDETECTIVE_MODEL", "claude-sonnet-5-5")
PROMPT_FILE = Path(__file__).with_name("dealdetective_prompt.md")

RUNTIME_NOTES = """
## RUNTIME NOTES
- Use the web_search tool to retrieve CURRENT prices, ratings, reviews and availability.
- Default currency is INR (₹) and the default market is India (Amazon.in, Flipkart, Croma, etc.)
  unless the user says otherwise.
- Cite the source and observation date for every price you report.
- If search results do not show a price, say "I could not verify the current price."
"""

TOOLS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 8,
        "user_location": {"type": "approximate", "country": "IN"},
    }
]


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        st.error("ANTHROPIC_API_KEY is not set. Add it in Streamlit secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=key)


@st.cache_data
def load_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        st.error(f"Missing {PROMPT_FILE.name} in the app folder.")
        st.stop()
    return PROMPT_FILE.read_text(encoding="utf-8") + "\n" + RUNTIME_NOTES


def ask_claude(api_messages: list) -> str:
    client = get_client()
    messages = list(api_messages)
    while True:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=load_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------- UI ----------
st.title("🔎 DealDetective AI")
st.caption("Is this actually a good deal? Paste a product name or link and find out.")

with st.sidebar:
    st.subheader("Try asking")
    examples = [
        "Is the boAt Airdopes 141 at ₹1,299 worth it?",
        "Best laptop under ₹50,000 for programming",
        "Compare iPhone 15 vs Samsung S24 prices",
        "Is this 70% discount real?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": ..., "content": str}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask about a product or deal...")
if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Investigating prices and reviews..."):
            try:
                answer = ask_claude(st.session_state.messages)
            except anthropic.APIError as e:
                answer = f"⚠️ API error: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
