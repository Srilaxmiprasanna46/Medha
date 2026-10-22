import { useState, useEffect, useRef } from 'react'
import './App.css'

// ---------------------------------------------------------------------------
// Feature-card data (document count is filled in dynamically)
// ---------------------------------------------------------------------------
const BASE_FEATURES = [
  {
    icon: '📚',
    title: 'My Knowledge',
    description: 'Your study materials, all in one place.',
    valueKey: 'docs',       // replaced at render time
  },
  {
    icon: '🧠',
    title: 'Smart Memory',
    description: 'Find passages across all your notes instantly.',
    value: 'Ready to search',
  },
  {
    icon: '🛡️',
    title: 'Stays on your device',
    description: 'Everything runs locally — nothing leaves your computer.',
    value: 'Private by design',
  },
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [question, setQuestion] = useState('')
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState(null)
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const chatBottomRef = useRef(null)

  // -------------------------------------------------------------------------
  // Feature A + B: load saved documents on mount
  // -------------------------------------------------------------------------
  useEffect(() => {
    async function fetchDocuments() {
      try {
        const response = await fetch('http://localhost:8000/api/documents')
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        setDocuments(Array.isArray(data.documents) ? data.documents : [])
      } catch {
        setDocsError('Could not load your saved documents. Is the Medha backend running?')
      } finally {
        setDocsLoading(false)
      }
    }
    fetchDocuments()
  }, [])

  // Auto-scroll chat to latest message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // -------------------------------------------------------------------------
  // Upload handler
  // -------------------------------------------------------------------------
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
            size_bytes: result.size_bytes ?? file.size,
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

  // -------------------------------------------------------------------------
  // Chat handler
  // -------------------------------------------------------------------------
  async function handleAsk(event) {
    event.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isLoading) return

    setMessages((current) => [...current, { role: 'user', text: trimmedQuestion }])
    setQuestion('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion }),
      })

      let data = null
      try {
        data = await response.json()
      } catch {
        data = null
      }

      if (!response.ok) {
        const detail = data && typeof data.detail === 'string' ? data.detail : null
        throw new Error(detail || `The server returned an error (HTTP ${response.status}).`)
      }

      if (!data || typeof data.answer !== 'string') {
        throw new Error('Medha received an unexpected response. Please try again.')
      }

      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          text: data.answer,
          grounded: data.grounded === true,
          method: data.method,
          citations: Array.isArray(data.citations) ? data.citations : [],
        },
      ])
    } catch (error) {
      const isNetworkError = error instanceof TypeError
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          isError: true,
          text: isNetworkError
            ? 'Could not reach Medha. Please check that the backend is running on http://localhost:8000.'
            : error.message,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------

  function renderDocumentList() {
    if (docsLoading) {
      return (
        <div className="empty-state">
          <span>⏳</span>
          <p>Loading your library…</p>
        </div>
      )
    }
    if (docsError) {
      return (
        <div className="empty-state error-state">
          <span>⚠️</span>
          <p>{docsError}</p>
        </div>
      )
    }
    if (documents.length === 0) {
      return (
        <div className="empty-state">
          <span>▤</span>
          <p>Your knowledge vault is waiting.</p>
          <small>Upload a document to get started.</small>
        </div>
      )
    }
    return documents.map((document, index) => (
      <div className="document-row" key={document.id || `${document.name}-${index}`}>
        <span className="file-icon">▤</span>
        <div className="document-info">
          <strong>{document.name}</strong>
          <small>
            {formatSize(document.size_bytes ?? document.size)}
            {document.pages != null && ` · ${document.pages} page${document.pages !== 1 ? 's' : ''}`}
            {document.status && ` · ${document.status}`}
          </small>
        </div>
      </div>
    ))
  }

  const docCount = documents.length

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
        <a className="nav-item" href="#assistant">✳ <span>Ask Medha</span></a>

        <div className="sidebar-bottom">
          <div className="privacy-card">
            <span>◈</span>
            <strong>Your knowledge, your device.</strong>
            <p>Notes stay on your computer — not in the cloud.</p>
          </div>
          <span className="version">MEDHA AI · EARLY ACCESS</span>
        </div>
      </aside>

      <section className="main-content" id="home">
        <header className="topbar">
          <span className="breadcrumb">Workspace / <strong>Overview</strong></span>
          <span className="local-status"><span className="status-dot" /> Runs locally</span>
        </header>

        <div className="content-wrap">
          <section className="welcome">
            <span className="eyebrow">YOUR PERSONAL STUDY SPACE</span>
            <h1>Your mind, <span>with more room.</span></h1>
            <p>Bring your study material together. Find what matters. Learn at your own pace.</p>
          </section>

          <section className="feature-grid">
            {BASE_FEATURES.map((feature) => (
              <article className="feature-card" key={feature.title}>
                <div className="feature-icon">{feature.icon}</div>
                <h2>{feature.title}</h2>
                <p>{feature.description}</p>
                <span className="feature-value">
                  {feature.valueKey === 'docs'
                    ? `${docCount} document${docCount !== 1 ? 's' : ''}`
                    : feature.value}
                </span>
              </article>
            ))}
          </section>

          <section className="workspace-grid">
            {/* ---- Knowledge Vault ---- */}
            <article className="panel knowledge-panel" id="knowledge">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">YOUR LIBRARY</span>
                  <h2>Knowledge vault</h2>
                </div>
                <span className="count-badge">{docCount}</span>
              </div>

              <label className="upload-zone" id="upload-zone">
                <span className="upload-icon">↑</span>
                <strong>Add your study materials</strong>
                <span>PDF, TXT or MD · up to 75 MB each</span>
                <span className="upload-button">Choose files</span>
                <input
                  type="file"
                  accept=".pdf,.txt,.md"
                  multiple
                  onChange={handleUpload}
                  id="file-input"
                />
              </label>

              <div className="document-list">
                {renderDocumentList()}
              </div>
            </article>

            {/* ---- Ask Medha ---- */}
            <article className="panel assistant-panel" id="assistant">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">STUDY ASSISTANT</span>
                  <h2>Ask your notes</h2>
                </div>
                <span className="ai-orb">✳</span>
              </div>

              <div className="chat-area">
                {messages.length === 0 ? (
                  <div className="chat-empty">
                    <div className="chat-illustration">✳</div>
                    <h3>Curiosity starts here.</h3>
                    <p>Ask anything about your notes.</p>
                  </div>
                ) : messages.map((message, index) => (
                  <div
                    className={`message ${message.role}${message.isError ? ' error' : ''}`}
                    key={index}
                  >
                    <span>{message.role === 'user' ? 'You' : 'Medha'}</span>
                    {message.role === 'assistant' && !message.isError && (
                      <span className={`evidence-status ${message.grounded ? 'found' : 'missing'}`}>
                        {message.grounded ? '✓ Found in your notes' : '! Not enough information'}
                      </span>
                    )}
                    <p>{message.text}</p>
                    {message.citations && message.citations.length > 0 && (
                      <div className="citation-list">
                        {message.citations.map((citation, citationIndex) => (
                          <div className="citation-card" key={citationIndex}>
                            <div className="citation-header">
                              <span className="citation-source">
                                {citation.document ?? 'Unknown document'}
                                {citation.page != null && ` · Page ${citation.page}`}
                              </span>
                              {typeof citation.relevance === 'number' && (
                                <span className="citation-score">
                                  Match {citation.relevance.toFixed(2)}
                                </span>
                              )}
                            </div>
                            <blockquote className="citation-text">{citation.text}</blockquote>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="loading-message" role="status">
                    Searching your notes…
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              <form className="question-form" onSubmit={handleAsk} id="question-form">
                <input
                  id="question-input"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask something about your studies…"
                  aria-label="Ask Medha a question"
                />
                <button
                  id="send-button"
                  type="submit"
                  aria-label="Send question"
                  disabled={isLoading || !question.trim()}
                >
                  ↑
                </button>
              </form>
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
