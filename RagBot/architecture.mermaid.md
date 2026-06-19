```mermaid
flowchart TB
    subgraph INGESTION["① DATA INGESTION"]
        direction TB
        WS["NetSol Website\nGitHub JSON (311 pages)"]
        PDF["PDF Upload\npdfplumber / pypdf"]
        NORM["Normalize & Filter\nregex clean · 300 min chars"]
        EXTRACT["Extract Text + Tables\npdfplumber.extract_tables()"]
        CHUNK["RecursiveCharacterTextSplitter\nchunk_size=800 · overlap=150"]
        EMBED["HuggingFaceEmbeddings\nall-MiniLM-L6-v2 · 384-dim"]
        FAISS["FAISS Vector Store\ncosine similarity · Top-K=6"]

        WS --> NORM
        PDF --> EXTRACT
        NORM --> CHUNK
        EXTRACT --> CHUNK
        CHUNK --> EMBED --> FAISS
    end

    subgraph AGENT["② LANGGRAPH AGENT LOOP"]
        direction TB
        RET_TOOL["create_retriever_tool\nWraps FAISS as LLM-callable tool"]
        LLM["Groq LLM\nllama-3.3-70b-versatile\nbind_tools → function calling"]
        AGENT_NODE["Agent Node\nSystemPrompt + History\n→ LLM.invoke()"]
        TOOL_NODE["Tool Executor Node\nParse tool_calls\n→ invoke tool\n→ format ToolMessage"]

        AGENT_NODE -->|"has tool_calls?"| COND{ }
        COND -->|YES| TOOL_NODE
        COND -->|NO| END["Return answer"]
        TOOL_NODE -->|ToolMessage| AGENT_NODE
    end

    subgraph UI["③ GRADIO WEB UI"]
        direction TB
        LEFT["Left Panel\nPDF Upload · Status\nSource Radio (PDF/Website)"]
        CHAT["Chat Interface\nChatbot · Textbox · Send Button"]
        SUGG["6× Suggestion Buttons\nWebsite: NetSol questions\nPDF: dynamic from content"]

        LEFT --> CHAT
        SUGG --> CHAT
    end

    subgraph GLOBAL["Global Python State"]
        GS["_website_graph · _pdf_graph\n_website_retriever · _pdf_retriever\n_pdf_raw_text · _pdf_status"]
    end

    FAISS -.-> RET_TOOL
    RET_TOOL -.-> TOOL_NODE

    CHAT -->|user message| AGENT_NODE
    AGENT_NODE -->|answer| CHAT

    GS -.->|graph references| AGENT_NODE
    GS -.->|retriever| RET_TOOL
