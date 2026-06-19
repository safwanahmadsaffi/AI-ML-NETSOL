const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;

function getSessionId() {
  const key = "rag_chat_session_id";
  let sessionId = localStorage.getItem(key);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(key, sessionId);
  }
  return sessionId;
}

// Populate metadata labels on startup
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("sessionLabel").textContent = getSessionId();
  document.getElementById("apiBase").textContent = API_BASE;
  loadActiveDocuments();
  
  // File drag-and-drop & selection handling
  const fileInput = document.getElementById("fileInput");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const dropzone = document.getElementById("dropzone");
  
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      displaySelectedFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--border-focus)";
    dropzone.style.background = "rgba(59, 130, 246, 0.08)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    dropzone.style.background = "transparent";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    dropzone.style.background = "transparent";
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      displaySelectedFile(e.dataTransfer.files[0]);
    }
  });
});

function displaySelectedFile(file) {
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("fileInfo").style.display = "flex";
}

// Fetch lists of loaded knowledge documents
async function loadActiveDocuments() {
  const docList = document.getElementById("docList");
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (res.ok) {
      const documents = await res.json();
      docList.innerHTML = "";
      if (documents.length === 0) {
        docList.innerHTML = `<li class="doc-item empty">No documents uploaded.</li>`;
      } else {
        documents.forEach(doc => {
          const li = document.createElement("li");
          li.className = "doc-item";
          li.textContent = doc;
          docList.appendChild(li);
        });
      }
    }
  } catch (err) {
    docList.innerHTML = `<li class="doc-item empty" style="color:var(--btn-reset-hover)">Failed to load knowledge list.</li>`;
  }
}

// Document Upload Handler
const uploadForm = document.getElementById("uploadForm");
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");
  const uploadBtn = document.getElementById("uploadBtn");
  
  if (fileInput.files.length === 0) return;
  
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  
  uploadBtn.disabled = true;
  uploadStatus.className = "status-msg info";
  uploadStatus.innerHTML = `<span class="spinner" style="display:inline-block; vertical-align:middle; margin-right:6px;"></span> Indexing document...`;
  
  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData
    });
    
    if (res.ok) {
      const data = await res.json();
      uploadStatus.className = "status-msg success";
      uploadStatus.textContent = "Indexed successfully!";
      // Clear file selection
      fileInput.value = "";
      document.getElementById("fileInfo").style.display = "none";
      loadActiveDocuments(); // Refresh sidebar list!
    } else {
      const err = await res.json();
      uploadStatus.className = "status-msg error";
      uploadStatus.textContent = err.detail || "Indexing failed.";
    }
  } catch (err) {
    uploadStatus.className = "status-msg error";
    uploadStatus.textContent = "Server connection lost.";
  } finally {
    uploadBtn.disabled = false;
  }
});

// Quick prompt helper
function fillPrompt(text) {
  const input = document.getElementById("messageInput");
  input.value = text;
  input.focus();
}

function addMessage(role, text, sources = []) {
  const chat = document.getElementById("chat");
  
  // Hide welcome box on first chat
  const welcome = document.getElementById("welcomeBox");
  if (welcome) welcome.style.display = "none";
  
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  
  // Set main answer text
  const textNode = document.createElement("div");
  textNode.className = "bubble-text";
  textNode.textContent = text;
  bubble.appendChild(textNode);
  
  // Format citations if bot response contains source chunks
  if (role === "bot" && sources && sources.length > 0) {
    const citationsBox = document.createElement("div");
    citationsBox.className = "citations-box";
    citationsBox.innerHTML = `<div class="citations-title">Sources & Citations:</div>`;
    
    const list = document.createElement("div");
    list.className = "citations-list";
    
    // De-duplicate references by source name and page
    const uniqueRefs = [];
    sources.forEach(src => {
      const label = `${src.source || 'Doc'}${src.page ? ` (Page ${src.page})` : ''}`;
      if (!uniqueRefs.includes(label)) {
        uniqueRefs.push(label);
        
        const chip = document.createElement("span");
        chip.className = "citation-chip";
        chip.title = src.content; // Display snippet on hover
        chip.innerHTML = `📖 ${label}`;
        list.appendChild(chip);
      }
    });
    
    citationsBox.appendChild(list);
    bubble.appendChild(citationsBox);
  }
  
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
}

// Chat Send Handler
const chatForm = document.getElementById("chatForm");
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("messageInput");
  const query = input.value.trim();
  if (!query) return;
  
  // Add User Message bubble
  addMessage("user", query);
  input.value = "";
  
  // Add dynamic Loading Indicator bubble
  const chat = document.getElementById("chat");
  const loader = document.createElement("div");
  loader.className = "chat-bubble bot-loading";
  loader.id = "botLoader";
  loader.innerHTML = `<span class="spinner"></span> <span>Consulting index knowledge base...</span>`;
  chat.appendChild(loader);
  chat.scrollTop = chat.scrollHeight;
  
  const session_id = getSessionId();
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, message: query }),
    });
    
    // Remove loading indicator
    const activeLoader = document.getElementById("botLoader");
    if (activeLoader) activeLoader.remove();
    
    if (res.ok) {
      const data = await res.json();
      addMessage("bot", data.answer, data.sources);
    } else {
      const err = await res.json();
      addMessage("bot", `Error: ${err.detail || "Server error"}`);
    }
  } catch (err) {
    const activeLoader = document.getElementById("botLoader");
    if (activeLoader) activeLoader.remove();
    addMessage("bot", "Network Timeout/Loss: Please check that the server is active.");
  }
});

// Session Reset Handler
const resetBtn = document.getElementById("resetBtn");
resetBtn.addEventListener("click", async () => {
  if (!confirm("Are you sure you want to clear the entire chat history for this session?")) return;
  
  const session_id = getSessionId();
  try {
    const res = await fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id }),
    });
    
    if (res.ok) {
      const chat = document.getElementById("chat");
      chat.innerHTML = `
        <div class="welcome-box" id="welcomeBox">
          <span class="welcome-icon">🗑️</span>
          <h2>Session Cleared</h2>
          <p>The conversational history has been purged. Ask a new question to start fresh!</p>
          <div class="quick-prompts">
            <button class="quick-prompt-btn" onclick="fillPrompt('What is the primary topic of the document?')">"What is the primary topic of the document?"</button>
            <button class="quick-prompt-btn" onclick="fillPrompt('Can you summarize the key findings?')">"Can you summarize the key findings?"</button>
          </div>
        </div>
      `;
      // Clear localStorage session to trigger a fresh new session UUID on refresh
      localStorage.removeItem("rag_chat_session_id");
      document.getElementById("sessionLabel").textContent = getSessionId();
    }
  } catch (err) {
    alert("Failed to reset session. Check server status.");
  }
});
