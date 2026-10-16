import { useState } from 'react'
import './App.css'

const features = [
  { icon: '📚', title: 'My Knowledge', description: 'Your notes, organized and searchable', value: '0 documents' },
  { icon: '🧠', title: 'Smart Memory', description: 'Find facts without unnecessary AI calls', value: 'Ready to learn' },
  { icon: '🛡️', title: 'Private by Design', description: 'Built for local-first document processing', value: 'Privacy first' },
]

function App() {
  const [question, setQuestion] = useState('')
  const [documents, setDocuments] = useState([])
  const [messages, setMessages] = useState([])

 async function handleUpload(event) {
  const files = Array.from(event.target.files || [])
  event.target.value = ''

  for (const file of files) {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/api/documents', {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.detail || 'Upload failed')
      }

      setDocuments((current) => [
        ...current,
        {
          name: result.name,
          size: file.size,
          id: result.id,
          pages: result.pages,
          status: result.status,
        },
      ])
    } catch (error) {
      alert(`Could not upload ${file.name}: ${error.message}`)
    }
  }
}

  function handleAsk(event) {
    event.preventDefault()
    if (!question.trim()) return

    setMessages((current) => [
      ...current,
      { role: 'user', text: question.trim() },
      {
        role: 'assistant',
        text: 'Your study assistant is being set up. Connect the Medha backend to retrieve evidence and generate answers from your documents.',
      },
    ])
    setQuestion('')
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home">
          <span className="brand-icon">M</span>
          <span>medha<span className="brand-ai">.ai</span></span>
        </a>

        <p className="nav-label">WORKSPACE</p>
        <a className="nav-item active" href="#home">⌂ <span>Overview</span></a>
        <a className="nav-item" href="#knowledge">▤ <span>My knowledge</span></a>
        <a className="nav-item" href="#assistant">✳ <span>AI assistant</span></a>

        <div className="sidebar-bottom">
          <div className="privacy-card">
            <span>◈</span>
            <strong>Your knowledge, your control.</strong>
            <p>Designed with privacy at its core.</p>
          </div>
          <span className="version">MEDHA AI · EARLY ACCESS</span>
        </div>
      </aside>

      <section className="main-content" id="home">
        <header className="topbar">
          <span className="breadcrumb">Workspace / <strong>Overview</strong></span>
          <span className="local-status"><span className="status-dot" /> Local-first design</span>
        </header>

        <div className="content-wrap">
          <section className="welcome">
            <span className="eyebrow">YOUR PERSONAL KNOWLEDGE SPACE</span>
            <h1>Your mind, <span>with more room.</span></h1>
            <p>Bring your study material together. Find what matters. Learn at your own pace.</p>
          </section>

          <section className="feature-grid">
            {features.map((feature) => (
              <article className="feature-card" key={feature.title}>
                <div className="feature-icon">{feature.icon}</div>
                <h2>{feature.title}</h2>
                <p>{feature.description}</p>
                <span className="feature-value">{feature.value}</span>
              </article>
            ))}
          </section>

          <section className="workspace-grid">
            <article className="panel knowledge-panel" id="knowledge">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">YOUR LIBRARY</span>
                  <h2>Knowledge vault</h2>
                </div>
                <span className="count-badge">{documents.length}</span>
              </div>

              <label className="upload-zone">
                <span className="upload-icon">↑</span>
                <strong>Add your study materials</strong>
                <span>Choose PDF, TXT, or other study files</span>
                <span className="upload-button">Choose files</span>
                <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    multiple
                    onChange={handleUpload}
                  />
              </label>

              <div className="document-list">
                {documents.length === 0 ? (
                  <div className="empty-state">
                    <span>▤</span>
                    <p>Your knowledge vault is waiting.</p>
                    <small>Add a document to get started.</small>
                  </div>
                ) : documents.map((document, index) => (
                  <div className="document-row" key={`${document.name}-${index}`}>
                    <span className="file-icon">▤</span>
                    <div className="document-info">
                      <strong>{document.name}</strong>
                      <small>{(document.size / 1024).toFixed(1)} KB · {document.pages} page(s) · {document.status}
                        </small>
                    </div>
                  </div>
                ))}
              </div>
              <p className="disclaimer">Prototype: selecting files does not upload, parse, or index them yet.</p>
            </article>

            <article className="panel assistant-panel" id="assistant">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">SYSTEM 1 → SYSTEM 2</span>
                  <h2>Ask your memory</h2>
                </div>
                <span className="ai-orb">✳</span>
              </div>

              <div className="chat-area">
                {messages.length === 0 ? (
                  <div className="chat-empty">
                    <div className="chat-illustration">✳</div>
                    <h3>Curiosity starts here.</h3>
                    <p>Ask a question about your notes. Evidence-backed answers will be available once retrieval is connected.</p>
                  </div>
                ) : messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={index}>
                    <span>{message.role === 'user' ? 'You' : 'Medha'}</span>
                    <p>{message.text}</p>
                  </div>
                ))}
              </div>

              <form className="question-form" onSubmit={handleAsk}>
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask something about your studies..."
                  aria-label="Ask Medha a question"
                />
                <button type="submit" aria-label="Send question">↑</button>
              </form>
              <p className="chat-note">Answers will be connected to your sources in the next stage.</p>
            </article>
          </section>

          <footer className="footer">
            <span>Medha AI <span className="footer-muted">· Learn smarter. Remember longer.</span></span>
            <span>Built for thoughtful learning.</span>
          </footer>
        </div>
      </section>
    </main>
  )
}

export default App
