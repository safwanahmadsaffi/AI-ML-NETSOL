import os
import re
import json
import requests
from typing import Annotated, Sequence

import gradio as gr

# ── Loaders & splitters ───────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings & vector store ─────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Tools ─────────────────────────────────────────────────────────────────────
from langchain_core.tools.retriever import create_retriever_tool

# ── Messages ──────────────────────────────────────────────────────────────────
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ── LLM ───────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 6
MIN_CHARS     = 300
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.2

GITHUB_BASE = "https://raw.githubusercontent.com/NETSOL-INTERNSHIP/NetSol-website-scraper/main"
JSON_URL    = f"{GITHUB_BASE}/netsol_data.json"

# ═════════════════════════════════════════════════════════════════════════════
# Suggested Questions (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

WEBSITE_SUGGESTIONS = [
    "What is NetSol Technologies and what does it do?",
    "What products and solutions does NetSol offer?",
    "Who are NetSol's key clients and partners?",
    "What is the Transcend Platform?",
    "What were NetSol's financial results for FY 2025?",
    "What certifications does NetSol hold?",
]

PDF_SUGGESTIONS = [
    "What is this document about?",
    "Summarize the key information in this document.",
    "What are the main sections or chapters?",
    "What important facts and figures are mentioned?",
    "List the key people or organizations mentioned.",
    "What conclusions or findings are presented?",
]

# ═════════════════════════════════════════════════════════════════════════════
# Global state
# ═════════════════════════════════════════════════════════════════════════════

_pdf_graph = None
_website_graph = None
_pdf_status = "No PDF uploaded."
_website_status = "Loading NETSOL website data..."

_pdf_retriever = None
_website_retriever = None
_pdf_raw_text = ""

# ═════════════════════════════════════════════════════════════════════════════
# Website Ingestion Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def normalize_website_content(text: str) -> str:
    text = text.replace("|", " ")
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def load_netsol_data() -> list[Document]:
    resp = requests.get(JSON_URL, timeout=60)
    resp.raise_for_status()
    pages = resp.json()
    documents = []
    skipped = 0
    for page in pages:
        raw_content = page.get("content", "")
        url = page.get("url", "")
        title = page.get("title", "")
        content = normalize_website_content(raw_content)
        if len(content) < MIN_CHARS:
            skipped += 1
            continue
        full_text = f"Page Title: {title}\nSource URL: {url}\n\n{content}"
        documents.append(Document(
            page_content=full_text,
            metadata={"source": url, "title": title},
        ))
    print(f"[Loader] {len(documents)} pages loaded, {skipped} skipped")
    return documents

def build_website_vector_store():
    docs = load_netsol_data()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks), len(docs)

# ═════════════════════════════════════════════════════════════════════════════
# PDF Ingestion Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def extract_pdf_content(pdf_path):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                for table in (page.extract_tables() or []):
                    for row in table:
                        if row:
                            text += " ".join(str(c) if c else "" for c in row) + "\n"
                if text.strip():
                    pages.append({"page_num": i, "text": text.strip()})
        return pages
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return [{"page_num": i + 1, "text": page.extract_text().strip()}
                for i, page in enumerate(reader.pages)
                if page.extract_text() and page.extract_text().strip()]

def build_pdf_vector_store(pdf_path: str):
    pages = extract_pdf_content(pdf_path)
    if not pages:
        raise ValueError("Could not extract text from PDF.")
    documents = []
    for page in pages:
        documents.append(Document(
            page_content=page["text"],
            metadata={"page_num": page["page_num"], "source": os.path.basename(pdf_path)}
        ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    global _pdf_raw_text
    _pdf_raw_text = "\n".join(p["text"] for p in pages)
    return vector_store, len(chunks), len(pages)

# ═════════════════════════════════════════════════════════════════════════════
# LangGraph Agent
# ═════════════════════════════════════════════════════════════════════════════

class AgentState(dict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_graph(retriever_tool_name: str, retriever_tool_description: str, retriever):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in Space secrets.")
    llm = ChatGroq(
        api_key=api_key, model=LLM_MODEL,
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    )
    retriever_tool = create_retriever_tool(
        retriever, name=retriever_tool_name, description=retriever_tool_description,
    )
    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        f"You are an expert AI assistant. "
        f"You have access to a retrieval tool named '{retriever_tool_name}' that searches content. "
        f"ALWAYS call the '{retriever_tool_name}' tool before answering factual questions. "
        f"Base your answers ONLY on the retrieved context. "
        f"If the context does not contain the answer, say exactly: "
        f"'This information was not found in the document.' "
        f"Respond in formal English only."
    )

    def agent_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_executor_node(state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        tool_results = []
        tool_map = {t.name: t for t in tools}
        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                if isinstance(result, list):
                    content = "\n\n---\n\n".join(
                        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
                        for d in result
                    )
                else:
                    content = str(result)
            else:
                content = f"Tool '{tool_name}' not found."
            tool_results.append(ToolMessage(content=content, tool_call_id=tool_id))
        return {"messages": tool_results}

    def should_use_tool(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tool_exec"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_exec", tool_executor_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_use_tool,
        {"tool_exec": "tool_exec", END: END},
    )
    graph.add_edge("tool_exec", "agent")
    return graph.compile()

# ═════════════════════════════════════════════════════════════════════════════
# PDF processing & suggestion generation
# ═════════════════════════════════════════════════════════════════════════════

def process_pdf_file(file_path):
    global _pdf_graph, _pdf_retriever, _pdf_status
    if file_path is None:
        _pdf_status = "Please upload a PDF file."
        return _pdf_status
    try:
        vector_store, n_chunks, n_pages = build_pdf_vector_store(file_path)
        _pdf_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K},
        )
        _pdf_graph = build_graph(
            retriever_tool_name="retrieve_pdf_data",
            retriever_tool_description="Search the uploaded PDF document content.",
            retriever=_pdf_retriever,
        )
        _pdf_status = f"✅ Ready — {n_pages} pages · {n_chunks} chunks indexed from PDF."
        return _pdf_status
    except Exception as e:
        _pdf_status = f"❌ Error processing PDF: {e}"
        return _pdf_status


def generate_pdf_suggestions(raw_text: str) -> list:
    if not raw_text or len(raw_text) < 100:
        return list(PDF_SUGGESTIONS)
    try:
        suggestions = []
        suggestions.append("What is this document about? Give me a summary.")

        lines = raw_text.split("\n")
        headings = []
        for l in lines:
            s = l.strip()
            if s and 10 < len(s) < 120 and re.match(r'^[A-Z][A-Za-z0-9\s\-–—:]+$', s) and not s.endswith('.'):
                headings.append(s)

        money = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion|M|B|thousand)?', raw_text)
        orgs = re.findall(
            r'[A-Z][A-Za-z]+ (?:Technologies|Inc|Corp|LLC|Ltd|Group|Company|International|Solutions|Software|Platform)',
            raw_text
        )
        urls = re.findall(r'https?://[^\s]+', raw_text)

        used = set()
        if len(headings) >= 2:
            t = headings[1].strip().rstrip(":")
            if t not in used:
                suggestions.append(f"Tell me more about: {t}")
                used.add(t)
        if orgs and orgs[0] not in used:
            suggestions.append(f"What is mentioned about {orgs[0]}?")
            used.add(orgs[0])
        if money:
            suggestions.append(f"What do the financial figures like {money[0]} represent?")
            used.add(money[0])
        if len(headings) >= 4:
            t = headings[3].strip().rstrip(":")
            if t not in used:
                suggestions.append(f"What does the document say about {t}?")
                used.add(t)
        if urls:
            d = urls[0].split("//")[-1].split("/")[0]
            if d not in used:
                suggestions.append(f"What source or website ({d}) is mentioned?")

        backups = [
            "What are the key facts and figures?",
            "List the main sections or topics covered.",
            "What conclusions or recommendations are presented?",
        ]
        for g in backups:
            if len(suggestions) < 6:
                suggestions.append(g)
        return suggestions[:6]
    except Exception:
        return list(PDF_SUGGESTIONS)


def handle_pdf_upload(file_path):
    status = process_pdf_file(file_path)
    pdf_suggestions = generate_pdf_suggestions(_pdf_raw_text)
    return status, pdf_suggestions

# ═════════════════════════════════════════════════════════════════════════════
# Query Handler
# ═════════════════════════════════════════════════════════════════════════════

def answer_query_combined(message: str, history: list, source_choice: str):
    lc_messages = []
    for user_msg, asst_msg in history:
        lc_messages.append(HumanMessage(content=user_msg))
        if asst_msg:
            lc_messages.append(AIMessage(content=asst_msg))
    lc_messages.append(HumanMessage(content=message))

    if source_choice == "PDF":
        if _pdf_graph is None:
            return f"📄 PDF system not ready. Status: {_pdf_status}"
        current_graph = _pdf_graph
    elif source_choice == "Website":
        if _website_graph is None:
            return f"🌐 Website system not ready. Status: {_website_status}"
        current_graph = _website_graph
    else:
        return "Please select a valid source (PDF or Website)."

    try:
        result = current_graph.invoke({"messages": lc_messages})
        return result["messages"][-1].content.strip()
    except Exception as e:
        return f"Error: {e}"

# ═════════════════════════════════════════════════════════════════════════════
# Chat UI handlers
# ═════════════════════════════════════════════════════════════════════════════

def on_user_submit(message: str, chat_history: list, source: str):
    """Handle user typing a message and pressing Enter/click Send.
    Chat history uses Gradio 6.0 dict format: [{'role':'user','content':...}, {'role':'assistant','content':...}]
    """
    if not message or not message.strip():
        return chat_history, ""
    chat_history = list(chat_history) if chat_history else []

    # Convert dict-format history to old tuple format for the LLM
    llm_history = []
    i = 0
    while i < len(chat_history):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]["content"]
            asst_msg = ""
            if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant":
                asst_msg = chat_history[i + 1]["content"]
            llm_history.append([user_msg, asst_msg])
            i += 2
        else:
            i += 1

    # Add new user message to the chat
    chat_history.append({"role": "user", "content": message})

    try:
        response = answer_query_combined(message, llm_history, source)
        chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Error: {e}"})

    return chat_history, ""


def get_suggestions_for_source(source: str, pdf_suggestions: list) -> list:
    """Return the right suggestion list for the given source."""
    if source == "PDF":
        return pdf_suggestions if pdf_suggestions else PDF_SUGGESTIONS
    return WEBSITE_SUGGESTIONS


def update_suggestion_buttons(source: str, pdf_suggestions: list):
    """Return gr.update objects to refresh the 6 suggestion buttons."""
    questions = get_suggestions_for_source(source, pdf_suggestions)
    updates = []
    for i in range(6):
        if i < len(questions):
            updates.append(gr.update(visible=True, value=questions[i]))
        else:
            updates.append(gr.update(visible=False, value=""))
    return updates


# ═════════════════════════════════════════════════════════════════════════════
# Startup: Load Website Data
# ═════════════════════════════════════════════════════════════════════════════

try:
    website_vector_store, n_website_chunks, n_website_pages = build_website_vector_store()
    _website_retriever = website_vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K},
    )
    _website_graph = build_graph(
        retriever_tool_name="retrieve_website_data",
        retriever_tool_description="Search the NETSOL Technologies official website content.",
        retriever=_website_retriever,
    )
    _website_status = (
        f"✅ Ready — {n_website_pages} pages · {n_website_chunks} chunks from NETSOL website."
    )
except Exception as e:
    _website_status = f"❌ Website startup error: {e}"

# ═════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Universal RAG Assistant") as demo:
    gr.HTML("<h1>📚🌐 Universal RAG Assistant (PDF & Website)</h1>")
    gr.Markdown(
        "Upload a PDF document or query the pre-indexed NETSOL Technologies website. "
        "Select your source below, then type a question or **click a suggestion**."
    )

    # ── Persistent state ──────────────────────────────────────────────────
    chat_state = gr.State([])
    pdf_suggestions_state = gr.State(list(PDF_SUGGESTIONS))

    with gr.Row():
        # ── Left: sources ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📄 PDF Document Source")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            pdf_status_text = gr.Textbox(
                label="PDF Status", value=_pdf_status, interactive=False
            )

            gr.Markdown("### 🌐 Website Source")
            website_status_text = gr.Textbox(
                label="Website Status", value=_website_status, interactive=False
            )

            source_choice = gr.Radio(
                ["PDF", "Website"],
                label="Source for Q&A",
                value="Website",
                interactive=True,
            )

        # ── Right: chat + suggestions ────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat with your Data")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=380,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your question",
                    placeholder="Type here and press Enter, or click a suggestion below...",
                    scale=4,
                    container=True,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            # ── Suggestion buttons ──────────────────────────────────────
            gr.Markdown("#### 💡 Suggested Questions — click any to ask:")
            suggestion_btns = []
            with gr.Row():
                for i in range(6):
                    with gr.Column(scale=1, min_width=100):
                        btn = gr.Button(
                            value="",
                            size="sm",
                            visible=False,
                            elem_id=f"sugg_{i}",
                        )
                        suggestion_btns.append(btn)

    # ═════════════════════════════════════════════════════════════════════
    # Event wiring
    # ═════════════════════════════════════════════════════════════════════

    # ── Text submit: Enter key or Send button ───────────────────────────
    msg_input.submit(
        fn=on_user_submit,
        inputs=[msg_input, chat_state, source_choice],
        outputs=[chatbot, msg_input],
    )
    send_btn.click(
        fn=on_user_submit,
        inputs=[msg_input, chat_state, source_choice],
        outputs=[chatbot, msg_input],
    )

    # ── Suggestion buttons: pass all 6 button values to the handler ────
    for i, btn in enumerate(suggestion_btns):
        def make_handler(idx):
            def handler(chat_hist, src, *all_btn_labels):
                question = all_btn_labels[idx] if idx < len(all_btn_labels) else ""
                return on_user_submit(question, chat_hist, src)
            return handler

        btn.click(
            fn=make_handler(i),
            inputs=[chat_state, source_choice] + suggestion_btns,
            outputs=[chatbot, msg_input],
        )

    # ── Source change: refresh suggestion buttons ──────────────────────
    source_choice.change(
        fn=update_suggestion_buttons,
        inputs=[source_choice, pdf_suggestions_state],
        outputs=suggestion_btns,
    )

    # ── PDF upload: process + generate suggestions + refresh buttons ───
    file_input.upload(
        fn=handle_pdf_upload,
        inputs=[file_input],
        outputs=[pdf_status_text, pdf_suggestions_state],
    ).then(
        fn=update_suggestion_buttons,
        inputs=[source_choice, pdf_suggestions_state],
        outputs=suggestion_btns,
    )

    # ── On load: show Website suggestions ──────────────────────────────
    demo.load(
        fn=lambda: update_suggestion_buttons("Website", list(PDF_SUGGESTIONS)),
        inputs=None,
        outputs=suggestion_btns,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Launch
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())


# import os
# import re
# import json
# import requests
# from typing import Annotated, Sequence, Literal

# import gradio as gr

# # ── Loaders & splitters ───────────────────────────────────────────────────────
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ── Embeddings & vector store ─────────────────────────────────────────────────
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document

# # ── Tools ─────────────────────────────────────────────────────────────────────
# from langchain_core.tools.retriever import create_retriever_tool

# # ── Messages ──────────────────────────────────────────────────────────────────
# from langchain_core.messages import (
#     AIMessage,
#     BaseMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )

# # ── LLM ───────────────────────────────────────────────────────────────────────
# from langchain_groq import ChatGroq

# # ── LangGraph ─────────────────────────────────────────────────────────────────
# from langgraph.graph import END, StateGraph
# from langgraph.graph.message import add_messages

# # ═════════════════════════════════════════════════════════════════════════════
# # Config
# # ═════════════════════════════════════════════════════════════════════════════

# CHUNK_SIZE    = 800
# CHUNK_OVERLAP = 150
# TOP_K         = 6
# MIN_CHARS     = 300   # skip pages shorter than this (nav/boilerplate)
# EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL     = "llama-3.3-70b-versatile"
# MAX_TOKENS    = 1024
# TEMPERATURE   = 0.2

# GITHUB_BASE = "https://raw.githubusercontent.com/NETSOL-INTERNSHIP/NetSol-website-scraper/main"
# JSON_URL    = f"{GITHUB_BASE}/netsol_data.json"

# # ═════════════════════════════════════════════════════════════════════════════
# # Global state
# # ═════════════════════════════════════════════════════════════════════════════

# _pdf_graph = None
# _website_graph = None
# _pdf_status = "No PDF uploaded."
# _website_status = "Loading NETSOL website data..."

# _pdf_retriever = None
# _website_retriever = None

# # ═════════════════════════════════════════════════════════════════════════════
# # Website Ingestion Pipeline (adapted from user's website RAG code)
# # ═════════════════════════════════════════════════════════════════════════════

# def normalize_website_content(text: str) -> str:
#     """
#     Clean scraped web content before chunking.
#     Order matters: do structural fixes before whitespace collapsing.
#     """
#     text = text.replace("|", " ")
#     text = re.sub(r" +\n", "\n", text)
#     text = re.sub(r"\n +", "\n", text)
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     text = re.sub(r" {2,}", " ", text)
#     return text.strip()

# def load_netsol_data() -> list[Document]:
#     resp = requests.get(JSON_URL, timeout=60)
#     resp.raise_for_status()
#     pages = resp.json()  # list of {url, title, content}

#     documents  = []
#     skipped    = 0

#     for page in pages:
#         raw_content = page.get("content", "")
#         url         = page.get("url", "")
#         title       = page.get("title", "")

#         content = normalize_website_content(raw_content)

#         if len(content) < MIN_CHARS:
#             skipped += 1
#             continue

#         full_text = f"Page Title: {title}\nSource URL: {url}\n\n{content}"

#         documents.append(Document(
#             page_content=full_text,
#             metadata={"source": url, "title": title},
#         ))

#     print(f"[Loader] {len(documents)} pages loaded, {skipped} skipped (too short)")
#     return documents

# def build_website_vector_store():
#     docs = load_netsol_data()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size    = CHUNK_SIZE,
#         chunk_overlap = CHUNK_OVERLAP,
#         separators    = ["\n\n", "\n", ". ", " ", ""],
#     )
#     chunks = splitter.split_documents(docs)
#     print(f"[Splitter] {len(chunks)} chunks created for website data")

#     embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     print(f"[FAISS] Website index built")

#     return vector_store, len(chunks), len(docs)

# # ═════════════════════════════════════════════════════════════════════════════
# # PDF Ingestion Pipeline (adapted from user's PDF RAG code, using FAISS)
# # ═════════════════════════════════════════════════════════════════════════════

# def extract_pdf_content(pdf_path):
#     try:
#         import pdfplumber
#         pages = []
#         with pdfplumber.open(pdf_path) as pdf:
#             for i, page in enumerate(pdf.pages, 1):
#                 text = page.extract_text() or ""
#                 for table in (page.extract_tables() or []):
#                     for row in table:
#                         if row:
#                             text += " ".join(str(c) if c else "" for c in row) + "\n"
#                 if text.strip():
#                     pages.append({"page_num": i, "text": text.strip()})
#         return pages
#     except ImportError:
#         from pypdf import PdfReader
#         reader = PdfReader(pdf_path)
#         return [{"page_num": i + 1, "text": page.extract_text().strip()} 
#                 for i, page in enumerate(reader.pages) 
#                 if page.extract_text() and page.extract_text().strip()]

# def build_pdf_vector_store(pdf_path: str):
#     pages = extract_pdf_content(pdf_path)
#     if not pages:
#         raise ValueError("Could not extract text from PDF.")

#     documents = []
#     for page in pages:
#         documents.append(Document(
#             page_content=page["text"],
#             metadata={"page_num": page["page_num"], "source": os.path.basename(pdf_path)}
#         ))

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size    = CHUNK_SIZE,
#         chunk_overlap = CHUNK_OVERLAP,
#         separators    = ["\n\n", "\n", " ", ""],
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"[Splitter] {len(chunks)} chunks created for PDF data")

#     embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     print(f"[FAISS] PDF index built")

#     return vector_store, len(chunks), len(pages)

# # ═════════════════════════════════════════════════════════════════════════════
# # LangGraph Agent (unified)
# # ═════════════════════════════════════════════════════════════════════════════

# class AgentState(dict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# def build_graph(retriever_tool_name: str, retriever_tool_description: str, retriever):
#     api_key = os.environ.get("GROQ_API_KEY", "")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY is not set in Space secrets.")

#     llm = ChatGroq(
#         api_key     = api_key,
#         model       = LLM_MODEL,
#         max_tokens  = MAX_TOKENS,
#         temperature = TEMPERATURE,
#     )

#     retriever_tool = create_retriever_tool(
#         retriever,
#         name        = retriever_tool_name,
#         description = retriever_tool_description,
#     )
#     tools          = [retriever_tool]
#     llm_with_tools = llm.bind_tools(tools)

#     system_prompt = (
#         f"You are an expert AI assistant. "
#         f"You have access to a retrieval tool named '{retriever_tool_name}' that searches content. "
#         f"ALWAYS call the '{retriever_tool_name}' tool before answering factual questions. "
#         f"Base your answers ONLY on the retrieved context. "
#         f"If the context does not contain the answer, say exactly: "
#         f"'This information was not found in the document.' "
#         f"Respond in formal English only."
#     )

#     def agent_node(state: AgentState) -> AgentState:
#         messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
#         response = llm_with_tools.invoke(messages)
#         return {"messages": [response]}

#     def tool_executor_node(state: AgentState) -> AgentState:
#         last_msg     = state["messages"][-1]
#         tool_results = []
#         tool_map     = {t.name: t for t in tools}

#         for tc in last_msg.tool_calls:
#             tool_name = tc["name"]
#             tool_args = tc["args"]
#             tool_id   = tc["id"]

#             if tool_name in tool_map:
#                 result = tool_map[tool_name].invoke(tool_args)
#                 if isinstance(result, list):
#                     content = "\n\n---\n\n".join(
#                         f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
#                         for d in result
#                     )
#                 else:
#                     content = str(result)
#             else:
#                 content = f"Tool '{tool_name}' not found."

#             tool_results.append(ToolMessage(content=content, tool_call_id=tool_id))

#         return {"messages": tool_results}

#     def should_use_tool(state: AgentState) -> str:
#         last_msg = state["messages"][-1]
#         if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
#             return "tool_exec"
#         return END

#     graph = StateGraph(AgentState)
#     graph.add_node("agent",     agent_node)
#     graph.add_node("tool_exec", tool_executor_node)
#     graph.set_entry_point("agent")
#     graph.add_conditional_edges(
#         "agent",
#         should_use_tool,
#         {"tool_exec": "tool_exec", END: END},
#     )
#     graph.add_edge("tool_exec", "agent")
#     return graph.compile()

# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio Handlers
# # ═════════════════════════════════════════════════════════════════════════════

# def process_pdf_file(file_path):
#     global _pdf_graph, _pdf_retriever, _pdf_status
#     if file_path is None:
#         _pdf_status = "Please upload a PDF file."
#         return _pdf_status
    
#     try:
#         vector_store, n_chunks, n_pages = build_pdf_vector_store(file_path)
#         _pdf_retriever = vector_store.as_retriever(
#             search_type   = "similarity",
#             search_kwargs = {"k": TOP_K},
#         )
#         _pdf_graph = build_graph(
#             retriever_tool_name="retrieve_pdf_data",
#             retriever_tool_description="Search the uploaded PDF document content.",
#             retriever=_pdf_retriever
#         )
#         _pdf_status = f"✅ Ready — {n_pages} pages · {n_chunks} chunks indexed from PDF."
#         return _pdf_status
#     except Exception as e:
#         _pdf_status = f"❌ Error processing PDF: {e}"
#         return _pdf_status

# def answer_query_combined(message: str, history: list, source_choice: Literal["PDF", "Website", "Both"]):
#     lc_messages: list[BaseMessage] = []
#     for user_msg, asst_msg in history:
#         lc_messages.append(HumanMessage(content=user_msg))
#         if asst_msg:
#             lc_messages.append(AIMessage(content=asst_msg))
#     lc_messages.append(HumanMessage(content=message))

#     current_graph = None
#     if source_choice == "PDF":
#         if _pdf_graph is None:
#             return f"PDF system not ready. Status: {_pdf_status}"
#         current_graph = _pdf_graph
#     elif source_choice == "Website":
#         if _website_graph is None:
#             return f"Website system not ready. Status: {_website_status}"
#         current_graph = _website_graph
#     elif source_choice == "Both":
#         # For 'Both', we need a more complex graph that can decide which tool to use
#         # or query both and combine results. For simplicity, let's just pick one for now
#         # or indicate that 'Both' is not fully implemented yet.
#         return "'Both' source selection is not yet fully implemented. Please choose either 'PDF' or 'Website'."
#     else:
#         return "Please select a valid source (PDF or Website)."

#     try:
#         result    = current_graph.invoke({"messages": lc_messages})
#         final_msg = result["messages"][-1]
#         return final_msg.content.strip()
#     except Exception as e:
#         return f"Error: {e}"

# # ═════════════════════════════════════════════════════════════════════════════
# # Startup: Load Website Data
# # ═════════════════════════════════════════════════════════════════════════════

# try:
#     website_vector_store, n_website_chunks, n_website_pages = build_website_vector_store()
#     _website_retriever = website_vector_store.as_retriever(
#         search_type   = "similarity",
#         search_kwargs = {"k": TOP_K},
#     )
#     _website_graph = build_graph(
#         retriever_tool_name="retrieve_website_data",
#         retriever_tool_description="Search the NETSOL Technologies official website content.",
#         retriever=_website_retriever
#     )
#     _website_status = (
#         f"✅ Ready — {n_website_pages} pages · {n_website_chunks} chunks indexed from NETSOL website."
#     )
# except Exception as e:
#     _website_status = f"❌ Website startup error: {e}"

# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio UI
# # ═════════════════════════════════════════════════════════════════════════════

# with gr.Blocks(title="Universal RAG Assistant") as demo:
#     gr.HTML("<h1>📚🌐 Universal RAG Assistant (PDF & Website)</h1>")
#     gr.Markdown(
#         "Upload a PDF document or query the pre-indexed NETSOL Technologies website. "
#         "Select your preferred source below."
#     )
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             gr.Markdown("### PDF Document Source")
#             file_input = gr.File(label="1. Upload PDF", file_types=[".pdf"], type="filepath")
#             pdf_status_text = gr.Textbox(label="PDF System Status", value=_pdf_status, interactive=False)
            
#             gr.Markdown("### Website Source")
#             website_status_text = gr.Textbox(label="Website System Status", value=_website_status, interactive=False)
            
#             source_choice = gr.Radio(
#                 ["PDF", "Website"], 
#                 label="Choose Source for Q&A", 
#                 value="Website", # Default to Website as it's pre-loaded
#                 interactive=True
#             )
            
#         with gr.Column(scale=2):
#             gr.Markdown("### 2. Chat with your Data")
#             gr.ChatInterface(
#                 fn=answer_query_combined,
#                 additional_inputs=[source_choice]
#             )
        
#     file_input.upload(fn=process_pdf_file, inputs=file_input, outputs=pdf_status_text)

# if __name__ == "__main__":
#     demo.launch()

    
# import os
# import re
# import math
# import pickle
# import shutil
# from collections import Counter
# import gradio as gr
# import chromadb
# from groq import Groq
# # Configuration
# VOCAB_SIZE = 8000
# CHUNK_SIZE = 400
# CHUNK_OVERLAP = 80
# TOP_K = 5
# DEFAULT_MODEL = "llama-3.3-70b-versatile"
# MAX_TOKENS = 1024
# TEMPERATURE = 0.3
# VECTOR_DB_PATH = "./chroma_db"

# # Updated to handle ANY document
# SYSTEM_PROMPT = """You are an expert document analysis assistant.
# Answer the user's questions based ONLY on the provided context from the uploaded document.
# If the answer is not found in the context, say exactly: 'This information was not found in the document.'
# Be precise and cite specific numbers, dates, and figures when available.
# IMPORTANT: Always respond in formal English only."""

# class TFIDFEmbedder:
#     def __init__(self, vocab_size=VOCAB_SIZE):
#         self.vocab_size = vocab_size
#         self.vocab = {}
#         self.idf = []
#         self._fitted = False

#     def _tokenize(self, text):
#         text = text.lower()
#         text = re.sub(r"[^a-z0-9\s]", " ", text)
#         return [t for t in text.split() if len(t) > 1]

#     def fit(self, documents):
#         N = len(documents)
#         df = Counter()
#         for doc in documents:
#             df.update(set(self._tokenize(doc)))
#         top_words = [w for w, _ in df.most_common(self.vocab_size)]
#         self.vocab = {w: i for i, w in enumerate(top_words)}
#         self.idf = [math.log((N + 1) / (df[w] + 1)) + 1 for w in top_words]
#         self._fitted = True

#     def transform(self, texts):
#         vectors = []
#         for text in texts:
#             tokens = self._tokenize(text)
#             tf = Counter(tokens)
#             total = max(len(tokens), 1)
#             vec = [0.0] * len(self.vocab)
#             for word, count in tf.items():
#                 if word in self.vocab:
#                     idx = self.vocab[word]
#                     vec[idx] = (count / total) * self.idf[idx]
#             norm = math.sqrt(sum(x * x for x in vec)) or 1.0
#             vectors.append([x / norm for x in vec])
#         return vectors

# class OfflineEF(chromadb.EmbeddingFunction):
#     def __init__(self, embedder):
#         self.embedder = embedder

#     def __call__(self, input_texts):
#         if not self.embedder._fitted:
#             raise RuntimeError("Embedder not fitted")
#         return self.embedder.transform(list(input_texts))

# def extract_pdf(pdf_path):
#     try:
#         import pdfplumber
#         pages = []
#         with pdfplumber.open(pdf_path) as pdf:
#             for i, page in enumerate(pdf.pages, 1):
#                 text = page.extract_text() or ""
#                 for table in (page.extract_tables() or []):
#                     for row in table:
#                         if row:
#                             text += " ".join(str(c) if c else "" for c in row) + "\n"
#                 if text.strip():
#                     pages.append({"page_num": i, "text": text.strip()})
#         return pages
#     except ImportError:
#         from pypdf import PdfReader
#         reader = PdfReader(pdf_path)
#         return [{"page_num": i + 1, "text": page.extract_text().strip()} 
#                 for i, page in enumerate(reader.pages) 
#                 if page.extract_text() and page.extract_text().strip()]

# def make_chunks(pages):
#     chunks = []
#     cid = 0
#     for page in pages:
#         text, pg = page["text"], page["page_num"]
#         start = 0
#         while start < len(text):
#             end = min(start + CHUNK_SIZE, len(text))
#             if end < len(text):
#                 sp = text.rfind(" ", start, end)
#                 if sp > start:
#                     end = sp
#             chunk = text[start:end].strip()
#             if chunk:
#                 chunks.append({"chunk_id": f"chunk_{cid:06d}", "page_num": pg, "text": chunk})
#                 cid += 1
#             start = end - CHUNK_OVERLAP if end < len(text) else end
#     return chunks

# class VectorStore:
#     def __init__(self, db_path=VECTOR_DB_PATH):
#         self.db_path = db_path
#         self.client = chromadb.PersistentClient(path=self.db_path)
#         self.embedder = TFIDFEmbedder(VOCAB_SIZE)
#         self.ef = OfflineEF(self.embedder)
#         self.col = self.client.get_or_create_collection(
#             name="document_rag",
#             embedding_function=self.ef,
#             metadata={"hnsw:space": "cosine"}
#         )

#     def reset_and_ingest(self, chunks):
#         # Clear existing data for the new PDF
#         try:
#             self.client.delete_collection("document_rag")
#         except:
#             pass
            
#         self.embedder = TFIDFEmbedder(VOCAB_SIZE)
#         self.ef = OfflineEF(self.embedder)
#         self.col = self.client.get_or_create_collection(
#             name="document_rag",
#             embedding_function=self.ef,
#             metadata={"hnsw:space": "cosine"}
#         )

#         docs = [c["text"] for c in chunks]
#         self.embedder.fit(docs)
        
#         for i in range(0, len(chunks), 50):
#             batch = chunks[i:i+50]
#             self.col.add(
#                 ids=[c["chunk_id"] for c in batch],
#                 documents=[c["text"] for c in batch],
#                 metadatas=[{"page_num": c["page_num"]} for c in batch],
#             )
#         return f"Success! Extracted and stored {len(chunks)} chunks. You can now ask questions."

#     def search(self, query, top_k=TOP_K):
#         try:
#             count = self.col.count()
#             if count == 0:
#                 return []
#             res = self.col.query(
#                 query_texts=[query],
#                 n_results=min(top_k, count),
#                 include=["documents", "metadatas", "distances"],
#             )
#             return [{"text": doc, "page_num": meta["page_num"], "similarity": round(1 - dist, 4)} 
#                     for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
#         except Exception:
#             return []

# store = VectorStore()

# def process_pdf(file_path):
#     if file_path is None:
#         return "Please upload a PDF file."
    
#     try:
#         pages = extract_pdf(file_path)
#         if not pages:
#             return "Error: Could not extract text from this PDF. It might be scanned images."
            
#         chunks = make_chunks(pages)
#         msg = store.reset_and_ingest(chunks)
#         return msg
#     except Exception as e:
#         return f"Error during processing: {str(e)}"

# def answer_query(message, history):
#     api_key = os.environ.get("GROQ_API_KEY", "")
#     if not api_key:
#         return "Error: GROQ_API_KEY is not set in the Space Settings (Variables and Secrets)."
    
#     client = Groq(api_key=api_key)
#     chunks = store.search(message, top_k=TOP_K)
    
#     if not chunks:
#         return "Database is empty. Please upload a PDF document first and wait for the success message."
        
#     context = "\n\n".join(f"[Page {c['page_num']}]\n{c['text']}" for c in chunks)
    
#     formatted_history = []
#     for user_msg, ast_msg in history:
#         formatted_history.append({"role": "user", "content": user_msg})
#         formatted_history.append({"role": "assistant", "content": ast_msg})
        
#     messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     messages += formatted_history[-8:]
#     messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {message}"})
    
#     try:
#         resp = client.chat.completions.create(
#             model=DEFAULT_MODEL,
#             max_tokens=MAX_TOKENS,
#             temperature=TEMPERATURE,
#             messages=messages,
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         return f"API Error: {str(e)}"

# with gr.Blocks(title="Universal Document RAG") as demo:
#     gr.HTML("<h1>Universal PDF Assistant</h1>")
#     gr.Markdown("Upload **any** PDF document. The system will automatically process it so you can ask questions about its contents.")
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             # type="filepath" directly gives us the string path, preventing the Gradio object error
#             file_input = gr.File(label="1. Upload PDF", file_types=[".pdf"], type="filepath")
#             status_text = gr.Textbox(label="System Status", value="Waiting for file upload...", interactive=False)
            
#         with gr.Column(scale=2):
#             gr.Markdown("### 2. Chat with your Document")
#             gr.ChatInterface(fn=answer_query)
        
#     # Trigger processing immediately when a file is uploaded
#     file_input.upload(fn=process_pdf, inputs=file_input, outputs=status_text)

# if __name__ == "__main__":
#     demo.launch()



















# """
# Universal PDF RAG — LangGraph + Tool Calling Edition
# =====================================================
# Architecture:
#   PDF  →  PyPDFLoader  →  RecursiveCharacterTextSplitter
#        →  FAISS (HuggingFace embeddings)
#        →  retriever bound as a tool
#        →  LangGraph StateGraph agent loop
#              agent node  ──(tool_calls?)──►  tool executor node
#                  ▲                                  │
#                  └──────── ToolMessage ─────────────┘
#        →  Gradio ChatInterface
# """

# import os
# from typing import Annotated, Sequence

# import gradio as gr

# # ── Loaders & splitters ───────────────────────────────────────────────────────
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ── Embeddings & vector store ─────────────────────────────────────────────────
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # ── Tools ─────────────────────────────────────────────────────────────────────
# from langchain_core.tools.retriever import create_retriever_tool

# # ── Messages ──────────────────────────────────────────────────────────────────
# from langchain_core.messages import (
#     AIMessage,
#     BaseMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )

# # ── LLM ───────────────────────────────────────────────────────────────────────
# from langchain_groq import ChatGroq

# # ── LangGraph ─────────────────────────────────────────────────────────────────
# from langgraph.graph import END, StateGraph
# from langgraph.graph.message import add_messages

# # ═════════════════════════════════════════════════════════════════════════════
# # Config
# # ═════════════════════════════════════════════════════════════════════════════

# CHUNK_SIZE  = 500
# CHUNK_OVERLAP = 80
# TOP_K       = 5
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL   = "llama-3.3-70b-versatile"
# MAX_TOKENS  = 1024
# TEMPERATURE = 0.3

# SYSTEM_PROMPT = (
#     "You are an expert document analysis assistant. "
#     "You have access to a retrieval tool that searches the uploaded PDF. "
#     "ALWAYS call the retrieve_documents tool before answering factual questions. "
#     "Base your answers ONLY on the retrieved context. "
#     "If the context does not contain the answer, say exactly: "
#     "'This information was not found in the document.' "
#     "Respond in formal English only."
# )

# # ═════════════════════════════════════════════════════════════════════════════
# # Global state — rebuilt on each PDF upload
# # ═════════════════════════════════════════════════════════════════════════════

# _graph = None

# # ═════════════════════════════════════════════════════════════════════════════
# # Step 1 — Ingestion pipeline
# # ═════════════════════════════════════════════════════════════════════════════

# def build_vector_store(pdf_path: str):
#     """Load PDF → split → embed → return FAISS vector store."""

#     loader = PyPDFLoader(pdf_path)
#     pages  = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size    = CHUNK_SIZE,
#         chunk_overlap = CHUNK_OVERLAP,
#         separators    = ["\n\n", "\n", " ", ""],
#     )
#     chunks = splitter.split_documents(pages)

#     embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     return vector_store, len(chunks)

# # ═════════════════════════════════════════════════════════════════════════════
# # Step 2 — LangGraph agent
# # ═════════════════════════════════════════════════════════════════════════════

# class AgentState(dict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]


# def build_graph(retriever):
#     api_key = os.environ.get("GROQ_API_KEY", "")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY environment variable is not set.")

#     llm = ChatGroq(
#         api_key     = api_key,
#         model       = LLM_MODEL,
#         max_tokens  = MAX_TOKENS,
#         temperature = TEMPERATURE,
#     )

#     retriever_tool = create_retriever_tool(
#         retriever,
#         name        = "retrieve_documents",
#         description = (
#             "Search and return relevant passages from the uploaded PDF document. "
#             "Use this tool whenever the user asks a question about the document content."
#         ),
#     )
#     tools          = [retriever_tool]
#     llm_with_tools = llm.bind_tools(tools)

#     def agent_node(state: AgentState) -> AgentState:
#         messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
#         response = llm_with_tools.invoke(messages)
#         return {"messages": [response]}

#     def tool_executor_node(state: AgentState) -> AgentState:
#         last_msg     = state["messages"][-1]
#         tool_results = []
#         tool_map     = {t.name: t for t in tools}

#         for tc in last_msg.tool_calls:
#             tool_name = tc["name"]
#             tool_args = tc["args"]
#             tool_id   = tc["id"]

#             if tool_name in tool_map:
#                 result = tool_map[tool_name].invoke(tool_args)
#                 if isinstance(result, list):
#                     content = "\n\n".join(
#                         f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
#                         for d in result
#                     )
#                 else:
#                     content = str(result)
#             else:
#                 content = f"Tool '{tool_name}' not found."

#             tool_results.append(ToolMessage(content=content, tool_call_id=tool_id))

#         return {"messages": tool_results}

#     def should_use_tool(state: AgentState) -> str:
#         last_msg = state["messages"][-1]
#         if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
#             return "tool_exec"
#         return END

#     graph = StateGraph(AgentState)
#     graph.add_node("agent",     agent_node)
#     graph.add_node("tool_exec", tool_executor_node)
#     graph.set_entry_point("agent")
#     graph.add_conditional_edges(
#         "agent",
#         should_use_tool,
#         {"tool_exec": "tool_exec", END: END},
#     )
#     graph.add_edge("tool_exec", "agent")

#     return graph.compile()

# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio handlers
# # ═════════════════════════════════════════════════════════════════════════════

# def process_pdf(file_path: str) -> str:
#     global _graph

#     if file_path is None:
#         return "Please upload a PDF file."

#     try:
#         vector_store, n_chunks = build_vector_store(file_path)
#         retriever = vector_store.as_retriever(
#             search_type   = "similarity",
#             search_kwargs = {"k": TOP_K},
#         )
#         _graph = build_graph(retriever)
#         return f"✅ Processed {n_chunks} chunks. You can now ask questions!"

#     except ValueError as e:
#         return f"Configuration error: {e}"
#     except Exception as e:
#         return f"Error during processing: {e}"


# def answer_query(message: str, history: list) -> str:
#     global _graph

#     if _graph is None:
#         return "Please upload a PDF first and wait for the success message."

#     lc_messages: list[BaseMessage] = []
#     for user_msg, asst_msg in history:
#         lc_messages.append(HumanMessage(content=user_msg))
#         if asst_msg:
#             lc_messages.append(AIMessage(content=asst_msg))
#     lc_messages.append(HumanMessage(content=message))

#     result    = _graph.invoke({"messages": lc_messages})
#     final_msg = result["messages"][-1]
#     return final_msg.content.strip()

# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio UI
# # ═════════════════════════════════════════════════════════════════════════════

# with gr.Blocks(title="PDF RAG — LangGraph Agent") as demo:
#     gr.HTML("<h1>Universal PDF Assistant</h1>")
#     gr.Markdown(
#         "Upload **any** PDF. The LangGraph agent will use **tool calling** "
#         "to retrieve relevant passages before answering your questions."
#     )

#     with gr.Row():
#         with gr.Column(scale=1):
#             file_input  = gr.File(
#                 label      = "1. Upload PDF",
#                 file_types = [".pdf"],
#                 type       = "filepath",
#             )
#             status_text = gr.Textbox(
#                 label       = "System status",
#                 value       = "Waiting for PDF upload…",
#                 interactive = False,
#             )

#         with gr.Column(scale=2):
#             gr.Markdown("### 2. Chat with your document")
#             gr.ChatInterface(fn=answer_query)

#     file_input.upload(
#         fn      = process_pdf,
#         inputs  = file_input,
#         outputs = status_text,
#     )

# if __name__ == "__main__":
#     demo.launch()











# """
# NETSOL Technologies RAG Chatbot — LangGraph + Tool Calling
# ===========================================================
# Pipeline: Fetch → Normalize → Filter → Chunk → Embed → FAISS → LangGraph agent
# """

# import os
# import re
# import json
# import requests
# from typing import Annotated, Sequence

# import gradio as gr

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.tools.retriever import create_retriever_tool
# from langchain_core.messages import (
#     AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
# )
# from langchain_groq import ChatGroq
# from langgraph.graph import END, StateGraph
# from langgraph.graph.message import add_messages

# # ═════════════════════════════════════════════════════════════════════════════
# # Config
# # ═════════════════════════════════════════════════════════════════════════════

# CHUNK_SIZE    = 800
# CHUNK_OVERLAP = 150
# TOP_K         = 6
# MIN_CHARS     = 300   # skip pages shorter than this (nav/boilerplate)
# EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL     = "llama-3.3-70b-versatile"
# MAX_TOKENS    = 1024
# TEMPERATURE   = 0.2

# GITHUB_BASE = "https://raw.githubusercontent.com/NETSOL-INTERNSHIP/NetSol-website-scraper/main"
# JSON_URL    = f"{GITHUB_BASE}/netsol_data.json"

# SYSTEM_PROMPT = """You are an expert AI assistant for NETSOL Technologies.
# You have access to a tool that searches content from the official NETSOL Technologies website.

# RULES:
# 1. ALWAYS call the retrieve_netsol_data tool first before answering ANY question.
# 2. Answer based ONLY on the retrieved content.
# 3. If the retrieved content does not contain the answer, say: 'This information was not found in the NETSOL website data.'
# 4. Cite the source URL when available in the context.
# 5. Respond in formal English only.

# NETSOL Technologies is a global leader in asset finance and digital transformation,
# known for NFS Ascent, FleetEdge, and the Transcend Platform."""

# # ═════════════════════════════════════════════════════════════════════════════
# # Step 1 — Normalize
# # ═════════════════════════════════════════════════════════════════════════════

# def normalize(text: str) -> str:
#     """
#     Clean scraped web content before chunking.
#     Order matters: do structural fixes before whitespace collapsing.
#     """
#     # 1. Replace pipe chars used as separators in scraped JS text
#     text = text.replace("|", " ")

#     # 2. Strip trailing spaces from every line
#     text = re.sub(r" +\n", "\n", text)

#     # 3. Strip leading spaces from every line
#     text = re.sub(r"\n +", "\n", text)

#     # 4. Collapse 3+ consecutive newlines → 2 (paragraph boundary)
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     # 5. Collapse multiple spaces → single space
#     text = re.sub(r" {2,}", " ", text)

#     # 6. Strip leading/trailing whitespace
#     return text.strip()


# # ═════════════════════════════════════════════════════════════════════════════
# # Step 2 — Load, normalize, filter
# # ═════════════════════════════════════════════════════════════════════════════

# def load_netsol_data() -> list[Document]:
#     resp = requests.get(JSON_URL, timeout=60)
#     resp.raise_for_status()
#     pages = resp.json()  # list of {url, title, content}

#     documents  = []
#     skipped    = 0

#     for page in pages:
#         raw_content = page.get("content", "")
#         url         = page.get("url", "")
#         title       = page.get("title", "")

#         # ── Normalize ────────────────────────────────────────────────────────
#         content = normalize(raw_content)

#         # ── Filter: skip nav-only / boilerplate pages ────────────────────────
#         if len(content) < MIN_CHARS:
#             skipped += 1
#             continue

#         # ── Prepend title + URL so they're searchable in every chunk ─────────
#         full_text = f"Page Title: {title}\nSource URL: {url}\n\n{content}"

#         documents.append(Document(
#             page_content=full_text,
#             metadata={"source": url, "title": title},
#         ))

#     print(f"[Loader] {len(documents)} pages loaded, {skipped} skipped (too short)")
#     return documents


# # ═════════════════════════════════════════════════════════════════════════════
# # Step 3 — Chunk + Embed → FAISS
# # ═════════════════════════════════════════════════════════════════════════════

# def build_vector_store():
#     docs = load_netsol_data()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size    = CHUNK_SIZE,
#         chunk_overlap = CHUNK_OVERLAP,
#         separators    = ["\n\n", "\n", ". ", " ", ""],
#     )
#     chunks = splitter.split_documents(docs)
#     print(f"[Splitter] {len(chunks)} chunks created")

#     embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     print(f"[FAISS] Index built")

#     return vector_store, len(chunks), len(docs)


# # ═════════════════════════════════════════════════════════════════════════════
# # Step 4 — LangGraph agent
# # ═════════════════════════════════════════════════════════════════════════════

# class AgentState(dict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]


# def build_graph(retriever):
#     api_key = os.environ.get("GROQ_API_KEY", "")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY is not set in Space secrets.")

#     llm = ChatGroq(
#         api_key     = api_key,
#         model       = LLM_MODEL,
#         max_tokens  = MAX_TOKENS,
#         temperature = TEMPERATURE,
#     )

#     retriever_tool = create_retriever_tool(
#         retriever,
#         name        = "retrieve_netsol_data",
#         description = (
#             "Search the NETSOL Technologies official website content. "
#             "Use this for ANY question about NETSOL — products (NFS Ascent, FleetEdge, "
#             "Transcend Platform), services, clients, industries, locations, careers, "
#             "company info, news, or technology solutions."
#         ),
#     )
#     tools          = [retriever_tool]
#     llm_with_tools = llm.bind_tools(tools)

#     def agent_node(state: AgentState) -> AgentState:
#         messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
#         response = llm_with_tools.invoke(messages)
#         return {"messages": [response]}

#     def tool_executor_node(state: AgentState) -> AgentState:
#         last_msg     = state["messages"][-1]
#         tool_results = []
#         tool_map     = {t.name: t for t in tools}

#         for tc in last_msg.tool_calls:
#             tool_name = tc["name"]
#             tool_args = tc["args"]
#             tool_id   = tc["id"]

#             if tool_name in tool_map:
#                 result = tool_map[tool_name].invoke(tool_args)
#                 if isinstance(result, list):
#                     content = "\n\n---\n\n".join(
#                         f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
#                         for d in result
#                     )
#                 else:
#                     content = str(result)
#             else:
#                 content = f"Tool '{tool_name}' not found."

#             tool_results.append(ToolMessage(content=content, tool_call_id=tool_id))

#         return {"messages": tool_results}

#     def should_use_tool(state: AgentState) -> str:
#         last_msg = state["messages"][-1]
#         if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
#             return "tool_exec"
#         return END

#     graph = StateGraph(AgentState)
#     graph.add_node("agent",     agent_node)
#     graph.add_node("tool_exec", tool_executor_node)
#     graph.set_entry_point("agent")
#     graph.add_conditional_edges(
#         "agent",
#         should_use_tool,
#         {"tool_exec": "tool_exec", END: END},
#     )
#     graph.add_edge("tool_exec", "agent")
#     return graph.compile()


# # ═════════════════════════════════════════════════════════════════════════════
# # Startup
# # ═════════════════════════════════════════════════════════════════════════════

# _graph          = None
# _startup_status = "⏳ Loading and indexing NETSOL website data…"

# try:
#     vector_store, n_chunks, n_pages = build_vector_store()
#     retriever = vector_store.as_retriever(
#         search_type   = "similarity",
#         search_kwargs = {"k": TOP_K},
#     )
#     _graph = build_graph(retriever)
#     _startup_status = (
#         f"✅ Ready — {n_pages} pages · {n_chunks} chunks indexed from NETSOL website."
#     )
# except Exception as e:
#     _startup_status = f"❌ Startup error: {e}"


# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio handler
# # ═════════════════════════════════════════════════════════════════════════════

# def answer_query(message: str, history: list) -> str:
#     if _graph is None:
#         return f"System not ready. Status: {_startup_status}"

#     lc_messages: list[BaseMessage] = []
#     for user_msg, asst_msg in history:
#         lc_messages.append(HumanMessage(content=user_msg))
#         if asst_msg:
#             lc_messages.append(AIMessage(content=asst_msg))
#     lc_messages.append(HumanMessage(content=message))

#     try:
#         result    = _graph.invoke({"messages": lc_messages})
#         final_msg = result["messages"][-1]
#         return final_msg.content.strip()
#     except Exception as e:
#         return f"Error: {e}"


# # ═════════════════════════════════════════════════════════════════════════════
# # Gradio UI
# # ═════════════════════════════════════════════════════════════════════════════

# with gr.Blocks(title="NETSOL Technologies AI Assistant") as demo:
#     gr.HTML("<h1>🔷 NETSOL Technologies AI Assistant</h1>")
#     gr.Markdown(
#         "Ask anything about **NETSOL Technologies** — NFS Ascent, FleetEdge, "
#         "Transcend Platform, clients, services, industries, and more."
#     )
#     gr.Textbox(
#         value       = _startup_status,
#         label       = "System status",
#         interactive = False,
#     )
#     gr.ChatInterface(fn=answer_query)

# if __name__ == "__main__":
#     demo.launch()
