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
  const [isLoading, setIsLoading] = useState(false)

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
        throw new Error('The server sent a response that Medha could not understand.')
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
            ? 'Could not reach the Medha backend. Please check that it is running on http://localhost:8000.'
            : error.message,
        },
      ])
    } finally {
      setIsLoading(false)
    }
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
              <p className="disclaimer">Files are sent to your local Medha backend and indexed in memory. They are lost when the backend restarts.</p>
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
                    <p>Ask a question about your notes. Answers show the matching passages from your uploaded documents.</p>
                  </div>
                ) : messages.map((message, index) => (
                  <div
                    className={`message ${message.role}${message.isError ? ' error' : ''}`}
                    key={index}
                  >
                    <span>{message.role === 'user' ? 'You' : 'Medha'}</span>
                    {message.role === 'assistant' && !message.isError && (
                      <span className={`evidence-status ${message.grounded ? 'found' : 'missing'}`}>
                        {message.grounded ? '✓ Evidence found' : '! Not enough evidence'}
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
                                  Keyword match {citation.relevance.toFixed(2)}
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
                    Searching your documents…
                  </div>
                )}
              </div>

              <form className="question-form" onSubmit={handleAsk}>
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask something about your studies..."
                  aria-label="Ask Medha a question"
                />
                <button
                  type="submit"
                  aria-label="Send question"
                  disabled={isLoading || !question.trim()}
                >
                  ↑
                </button>
              </form>
              <p className="chat-note">Answers come from keyword matching on your uploaded documents. No AI model is used.</p>
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
