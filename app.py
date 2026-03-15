import streamlit as st

from agents.head_agent import Head_Agent

# Azure OpenAI 配置
AZURE_ENDPOINT = "https://haozez-0810-resource.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT = "gpt-4.1-nano"
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

# Pinecone 配置
INDEX_NAME = "mini2"
NAMESPACE = "ns2500"

# Page setup
st.set_page_config(page_title="Mini Project 3 - Multi-Agent Chatbot")
st.title("Mini Project Part 3: Multi-Agent Chatbot")

# Get API keys from secrets
try:
    azure_api_key = st.secrets["AZURE_OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
except KeyError:
    st.error("Please configure API keys in .streamlit/secrets.toml")
    st.stop()

# Sidebar: config
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"🤖 **Model:** `{AZURE_DEPLOYMENT}`")
    st.markdown(f"📚 **Index:** `{INDEX_NAME}`")
    st.markdown(f"🏷️ **Namespace:** `{NAMESPACE}`")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def _truncate(s: str, n: int = 350) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "..."


if "head_agent" not in st.session_state:
    try:
        st.session_state.head_agent = Head_Agent(
            azure_api_key=azure_api_key,
            azure_endpoint=AZURE_ENDPOINT,
            azure_api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT,
            azure_embedding_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            pinecone_key=pinecone_api_key,
            pinecone_index_name=INDEX_NAME,
            namespace=NAMESPACE
        )
        st.sidebar.success("Connected to Azure OpenAI + Pinecone!")
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        st.stop()

head = st.session_state.head_agent

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = AZURE_DEPLOYMENT

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about the indexed content..."):
    # 1) show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) run controller pipeline
    # Pass history WITHOUT the current user message (so rewrite agent sees prior context)
    result = head.handle(prompt, history=st.session_state.messages[:-1], k=5)

    with st.chat_message("assistant"):
        # "Brain Process" — show different message for greeting/obnoxious vs normal QA
        if result.get("is_greeting"):
            st.info("**Brain Process**: Greeting detected — replied with a short welcome.")
        elif result.get("is_obnoxious"):
            st.info("**Brain Process**: Query was flagged as impolite — asked user to rephrase.")
        else:
            st.info(f"**Brain Process**: I rephrased your question to: '*{result.get('rewritten_query', '')}*'")

        # retrieved docs expander
        with st.expander("Retrieved Context"):
            docs = result.get("docs", []) or []
            if not docs:
                st.caption("No documents retrieved.")
            else:
                for i, d in enumerate(docs, 1):
                    # d is RetrievedDoc (text, metadata)
                    page = d.metadata.get("page_number", d.metadata.get("page", "N/A"))
                    st.markdown(f"**Chunk {i} (Page {page})**")
                    st.caption(_truncate(d.text, 600))

        # 3) stream or print final answer
        if result.get("final_stream") is None:
            response_text = result.get("final_text", "")
            st.markdown(response_text)
        else:
            response_text = st.write_stream(result["final_stream"])

    # 4) persist assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
