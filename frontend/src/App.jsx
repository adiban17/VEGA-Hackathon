import { useState, useEffect, useRef } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

// --- 1. REUSABLE SCAN PAGE COMPONENT ---
const ScanPage = ({ title, scanType, description, icon }) => {
  const [scanFile, setScanFile] = useState(null)
  const [scanResult, setScanResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const analyzeScan = async () => {
    if (!scanFile) return alert("Please upload a scan first.")
    setLoading(true)
    const formData = new FormData()
    formData.append('scan_type', scanType)
    formData.append('file', scanFile)

    try {
      const res = await axios.post('http://127.0.0.1:8000/api/analyze-scan', formData)
      setScanResult(res.data)
    } catch (error) {
      console.error(error)
      alert("Error processing scan. Check the FastAPI terminal.")
    }
    setLoading(false)
  }

  return (
    <div className="container mt-5">
      <div className="mb-5 text-center">
        <h2 className="fw-bold" style={{ color: '#38bdf8' }}><i className={`bi ${icon} me-2`}></i>{title}</h2>
        <p className="text-light opacity-75">{description}</p>
      </div>
      <div className="row g-4 justify-content-center">
        <div className="col-md-5">
          <div className="card p-4 shadow-lg glass-panel h-100 rounded-4">
            <h5 className="mb-3 text-white"><i className="bi bi-cloud-arrow-up me-2"></i>Upload Medical Scan</h5>
            <label className="fw-semibold text-light opacity-75 small mb-2">Supported formats: JPG, PNG</label>
            <input type="file" className="form-control bg-dark text-light border-secondary mb-4 p-3" accept="image/*" onChange={e => setScanFile(e.target.files[0])} />
            <button className="btn w-100 fw-bold shadow-sm p-3 mt-auto" style={{ backgroundColor: '#0ea5e9', color: 'white', borderRadius: '12px' }} onClick={analyzeScan} disabled={loading}>
              {loading ? <span><span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing...</span> : 'Run AI Analysis'}
            </button>
          </div>
        </div>
        <div className="col-md-7">
          {scanResult ? (
            <div className="card p-4 shadow-lg glass-panel h-100 rounded-4">
              <div className="d-flex align-items-center mb-3">
                <i className={`bi ${scanResult.risk_title.includes('High') ? 'bi-exclamation-triangle-fill text-danger' : 'bi-check-circle-fill text-success'} fs-3 me-3`}></i>
                <h4 className={`fw-bold mb-0 ${scanResult.risk_title.includes('High') ? 'text-danger' : 'text-success'}`}>
                  {scanResult.risk_title}
                </h4>
              </div>
              <p className="lead mt-2 opacity-75 text-light">{scanResult.patient_explanation}</p>
              
              {scanResult.heatmap_generated && (
                <div className="mt-4 text-center bg-dark p-3 rounded-4 border border-secondary shadow-sm">
                  <h6 className="text-info mb-3 text-uppercase tracking-wider small fw-bold">AI Attention Heatmap (Grad-CAM)</h6>
                  <img src={scanResult.heatmap_base64} alt="Heatmap" className="img-fluid rounded shadow" style={{maxHeight: '350px'}} />
                </div>
              )}
              {!scanResult.heatmap_generated && scanResult.risk_title && (
                 <div className="mt-4 text-center p-3 rounded-4 bg-dark border border-secondary">
                  <span className="badge bg-secondary p-2"><i className="bi bi-info-circle me-2"></i>Visual Explainability Bypassed for this Architecture</span>
                 </div>
              )}
            </div>
          ) : (
            <div className="card p-4 shadow-lg glass-panel h-100 d-flex align-items-center justify-content-center rounded-4 text-center">
              <div className="opacity-50">
                <i className="bi bi-image fs-1 d-block mb-3"></i>
                <p>Upload a scan to view AI diagnostic insights.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// --- 2. VITALS PAGE COMPONENT ---
const VitalsPage = () => {
  const [vitals, setVitals] = useState({ hemoglobin: 14.0, fasting_sugar: 95, wbc_count: 6000 })
  const [vitalsResult, setVitalsResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const analyzeVitals = async () => {
    setLoading(true)
    try {
      const res = await axios.post('http://127.0.0.1:8000/api/analyze-vitals', vitals)
      setVitalsResult(res.data)
    } catch (error) {
      alert("Error connecting to backend.")
    }
    setLoading(false)
  }

  return (
    <div className="container mt-5">
      <div className="text-center mb-5">
        <h2 className="fw-bold" style={{ color: '#38bdf8' }}><i className="bi bi-activity me-2"></i>Structured Vitals Analysis</h2>
      </div>
      <div className="row g-4 justify-content-center">
        <div className="col-md-4">
          <div className="card p-4 shadow-lg glass-panel h-100 rounded-4">
            <h5 className="mb-4 text-white"><i className="bi bi-clipboard-data me-2"></i>Input Test Data</h5>
            
            <label className="fw-semibold text-info small text-uppercase">Hemoglobin (g/dL)</label>
            <input type="number" step="0.1" className="form-control bg-dark text-light border-secondary mb-3 p-2" value={vitals.hemoglobin} onChange={e => setVitals({...vitals, hemoglobin: parseFloat(e.target.value)})} />
            
            <label className="fw-semibold text-info small text-uppercase">Fasting Sugar (mg/dL)</label>
            <input type="number" className="form-control bg-dark text-light border-secondary mb-3 p-2" value={vitals.fasting_sugar} onChange={e => setVitals({...vitals, fasting_sugar: parseFloat(e.target.value)})} />
            
            <label className="fw-semibold text-info small text-uppercase">WBC Count (cells/mcL)</label>
            <input type="number" className="form-control bg-dark text-light border-secondary mb-4 p-2" value={vitals.wbc_count} onChange={e => setVitals({...vitals, wbc_count: parseFloat(e.target.value)})} />
            
            <button className="btn w-100 fw-bold shadow-sm p-3 mt-auto" style={{ backgroundColor: '#0ea5e9', color: 'white', borderRadius: '12px' }} onClick={analyzeVitals} disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Vitals'}
            </button>
          </div>
        </div>
        <div className="col-md-8">
          {vitalsResult ? (
            <div className="card p-4 shadow-lg glass-panel h-100 rounded-4">
              <h4 className={`fw-bold text-${vitalsResult.risk_level === 'High' ? 'danger' : vitalsResult.risk_level === 'Moderate' ? 'warning' : 'success'}`}>
                Risk Level: {vitalsResult.risk_level}
              </h4>
              <hr className="border-secondary my-4" />
              <h5 className="mb-4 text-white"><i className="bi bi-person-lines-fill me-2"></i>Patient Explanation</h5>
              {vitalsResult.abnormalities.length === 0 ? (
                <div className="alert bg-success text-white border-0 fw-bold rounded-3 shadow-sm"><i className="bi bi-check-circle me-2"></i>All parameters fall within healthy ranges.</div>
              ) : (
                vitalsResult.abnormalities.map((item, idx) => (
                  <div key={idx} className="alert bg-danger text-white border-0 shadow-sm rounded-3">
                    <strong><i className="bi bi-exclamation-circle-fill me-2"></i>{item.parameter} is {item.status} ({item.value}):</strong> {item.explanation}
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="card p-4 shadow-lg glass-panel h-100 d-flex align-items-center justify-content-center rounded-4 text-muted text-center">
              <div className="opacity-50">
                <i className="bi bi-keyboard fs-1 d-block mb-3"></i>
                <p>Enter patient vitals to see results.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// --- 3. DOCUMENT & CHAT PAGE COMPONENT ---
const DocsPage = () => {
  const GEMINI_API_KEY = "YOUR_API_KEY_HERE" // <-- PASTE YOUR KEY HERE

  const [docFile, setDocFile] = useState(null)
  const [loading, setLoading] = useState(false)
  
  const [messages, setMessages] = useState([])
  const [chatInput, setChatInput] = useState("")
  const [isChatting, setIsChatting] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const summarizeDoc = async () => {
    if (!docFile) return alert("Please upload a document first.")
    setLoading(true)
    const formData = new FormData()
    formData.append('api_key', GEMINI_API_KEY) 
    formData.append('file', docFile)

    try {
      const res = await axios.post('http://127.0.0.1:8000/api/summarize-report', formData)
      setMessages([{ role: 'model', text: res.data.summary_markdown }])
    } catch (error) {
      alert("Error summarizing document.")
    }
    setLoading(false)
  }

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const newUserMsg = { role: 'user', text: chatInput };
    const updatedMessages = [...messages, newUserMsg];
    
    setMessages(updatedMessages);
    setChatInput("");
    setIsChatting(true);

    try {
      const contents = updatedMessages.map(msg => ({
        role: msg.role === 'model' ? 'model' : 'user',
        parts: [{ text: msg.text }]
      }));

      const response = await axios.post(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`,
        { contents },
        { headers: { 'Content-Type': 'application/json' } }
      );

      const botReply = response.data.candidates[0].content.parts[0].text;
      setMessages(prev => [...prev, { role: 'model', text: botReply }]);
      
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'model', text: "❌ *Error communicating with AI. Please check your API key.*" }]);
    }
    setIsChatting(false);
  }

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-10">
          <div className="card p-5 shadow-lg glass-panel rounded-4">
            <h3 className="mb-3 fw-bold" style={{ color: '#38bdf8' }}><i className="bi bi-robot me-2"></i>Sentinel AI Report Assistant</h3>
            <p className="text-light opacity-75">Upload a lab report to translate medical jargon, then chat securely with the AI about your results.</p>
            
            <div className="row mt-4">
              <div className="col-md-12">
                <label className="fw-semibold text-info small text-uppercase">Upload Lab Report (PDF/PNG/JPG)</label>
                <div className="input-group mt-2 mb-3">
                  <span className="input-group-text bg-dark border-secondary text-info"><i className="bi bi-file-earmark-medical"></i></span>
                  <input type="file" className="form-control bg-dark text-light border-secondary p-3" accept=".pdf, image/*" onChange={e => setDocFile(e.target.files[0])} />
                </div>
              </div>
            </div>
            
            {messages.length === 0 && (
              <button className="btn w-100 fw-bold shadow-sm p-3 mt-3" style={{ backgroundColor: '#0ea5e9', color: 'white', borderRadius: '12px' }} onClick={summarizeDoc} disabled={loading}>
                {loading ? <span><span className="spinner-border spinner-border-sm me-2"></span>Reading & Translating...</span> : 'Analyze Report'}
              </button>
            )}

            {/* LIVE CHAT INTERFACE */}
            {messages.length > 0 && (
              <div className="mt-5">
                <h5 className="fw-bold text-info mb-3"><i className="bi bi-chat-dots me-2"></i>Real-Time Consultation</h5>
                
                <div className="chat-box p-4 border border-secondary rounded-4 shadow-inner" style={{ height: '500px', overflowY: 'auto', backgroundColor: '#0b1120' }}>
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`mb-4 d-flex ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
                      <div className={`p-3 rounded-4 shadow-sm ${msg.role === 'user' ? 'text-dark' : 'text-light'}`} style={{ maxWidth: '85%', backgroundColor: msg.role === 'user' ? '#38bdf8' : '#1e293b' }}>
                        <ReactMarkdown className="mb-0 markdown-body">{msg.text}</ReactMarkdown>
                      </div>
                    </div>
                  ))}
                  {isChatting && (
                    <div className="text-start mb-4">
                      <span className="badge bg-secondary p-2 shadow-sm text-light rounded-pill"><span className="spinner-grow spinner-grow-sm me-2"></span>Sentinel AI is typing...</span>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="input-group mt-4 shadow-sm rounded-pill overflow-hidden border border-secondary">
                  <input 
                    type="text" 
                    className="form-control bg-dark text-light border-0 p-3" 
                    placeholder="Ask a follow-up question..." 
                    value={chatInput} 
                    onChange={e => setChatInput(e.target.value)} 
                    onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
                  />
                  <button className="btn px-4 fw-bold" style={{ backgroundColor: '#0ea5e9', color: 'white' }} onClick={handleSendMessage} disabled={isChatting}>
                    <i className="bi bi-send-fill"></i>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// --- 3.5 PRESCRIPTION DECODER COMPONENT ---
const PrescriptionPage = () => {
    const GEMINI_API_KEY = "YOUR_API_KEY_HERE" // <-- PASTE YOUR KEY HERE
    
    const [imageFile, setImageFile] = useState(null)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
  
    // Helper function to convert the image file into the format Gemini requires
    const fileToGenerativePart = async (file) => {
      return new Promise((resolve) => {
        const reader = new FileReader()
        reader.onloadend = () => resolve({
          inlineData: { data: reader.result.split(',')[1], mimeType: file.type }
        })
        reader.readAsDataURL(file)
      })
    }
  
    const decodeHandwriting = async () => {
      if (!imageFile) return alert("Please upload an image of the prescription.")
      setLoading(true)
  
      try {
        const imagePart = await fileToGenerativePart(imageFile)
        
        const systemPrompt = `
        You are an expert pharmacist AI. Carefully read this handwritten doctor's prescription or medical note. 
        Format your response EXACTLY like this:
        
        ### 📝 Decoded Prescription
        - **Medication/Note 1:** [Name] - [Dosage/Instructions]
        - **Medication/Note 2:** [Name] - [Dosage/Instructions]
        
        ### ⚠️ Safety Warning
        Add a brief warning that AI can misread handwriting and the patient MUST verify this with a human pharmacist before taking any medication.
        `
  
        const payload = {
          contents: [{ parts: [{ text: systemPrompt }, imagePart] }]
        }
  
        const response = await axios.post(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`,
          payload,
          { headers: { 'Content-Type': 'application/json' } }
        )
  
        setResult(response.data.candidates[0].content.parts[0].text)
      } catch (error) {
        console.error(error)
        alert("Error deciphering the image. Ensure your API key is correct.")
      }
      setLoading(false)
    }
  
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8">
            <div className="card p-5 shadow-lg glass-panel rounded-4">
              <h3 className="mb-3 fw-bold" style={{ color: '#38bdf8' }}><i className="bi bi-pen me-2"></i>Handwriting Decoder</h3>
              <p className="text-light opacity-75">Upload a photo of a doctor's handwritten prescription or notes to digitize and translate it.</p>
              
              <input type="file" className="form-control bg-dark text-light border-secondary mt-3 mb-4 p-3" accept="image/*" onChange={e => setImageFile(e.target.files[0])} />
              
              <button className="btn w-100 fw-bold shadow-sm p-3" style={{ backgroundColor: '#0ea5e9', color: 'white', borderRadius: '12px' }} onClick={decodeHandwriting} disabled={loading}>
                {loading ? <span><span className="spinner-border spinner-border-sm me-2"></span>Deciphering Handwriting...</span> : 'Decode Prescription'}
              </button>
  
              {result && (
                <div className="mt-5 p-4 border border-secondary rounded-4 shadow-sm" style={{ backgroundColor: '#0b1120' }}>
                  <ReactMarkdown className="markdown-body text-light">{result}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

// --- 4. LANDING PAGE COMPONENT ---
const LandingPage = () => {
  return (
    <div className="container mt-5">
      <div className="text-center mb-5 pb-3">
        <div className="d-inline-block p-3 rounded-circle shadow-lg mb-3" style={{ background: 'linear-gradient(135deg, #0284c7, #38bdf8)' }}>
           <i className="bi bi-shield-plus text-white display-4" style={{ lineHeight: 0 }}></i>
        </div>
        <h1 className="display-4 fw-bold text-white tracking-tight">Sentinel Health</h1>
        <p className="lead mt-3 opacity-75 text-light" style={{ maxWidth: '600px', margin: '0 auto' }}>Proactive AI diagnostics and plain-English medical insights, designed for patients.</p>
      </div>

      <div className="row g-4 mt-2">
        {/* Core Tools */}
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-file-earmark-medical text-info display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Report Assistant</h5>
              <p className="card-text opacity-75 text-light small">Upload lab reports for plain-English translations and live AI consultation.</p>
              <Link to="/docs" className="btn btn-outline-info w-100 mt-auto fw-bold rounded-pill">Open Tool</Link>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-heart-pulse text-info display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Structured Vitals</h5>
              <p className="card-text opacity-75 text-light small">Input raw medical metrics to instantly check ranges and health risk levels.</p>
              <Link to="/vitals" className="btn btn-outline-info w-100 mt-auto fw-bold rounded-pill">Open Tool</Link>
            </div>
          </div>
        </div>
        
        {/* Scan Tools */}
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-x-diamond text-warning display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Brain Tumor MRI</h5>
              <p className="card-text opacity-75 text-light small">PyTorch-powered tumor detection using EfficientNet architectures.</p>
              <Link to="/scan/tumor" className="btn btn-outline-warning w-100 mt-auto fw-bold rounded-pill">Analyze MRI</Link>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-lungs text-warning display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Lung Cancer CT</h5>
              <p className="card-text opacity-75 text-light small">PyTorch-powered pulmonary nodule screening algorithms.</p>
              <Link to="/scan/lung-cancer" className="btn btn-outline-warning w-100 mt-auto fw-bold rounded-pill">Analyze CT</Link>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-person-bounding-box text-danger display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Alzheimer's MRI</h5>
              <p className="card-text opacity-75 text-light small">Keras classification with visual attention heatmaps.</p>
              <Link to="/scan/alzheimers" className="btn btn-outline-danger w-100 mt-auto fw-bold rounded-pill">Analyze MRI</Link>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 shadow-lg glass-panel hover-card rounded-4 p-2 border-0">
            <div className="card-body text-center d-flex flex-column">
              <i className="bi bi-bandaid text-danger display-4 mb-3"></i>
              <h5 className="card-title fw-bold text-white">Bone Fracture X-Ray</h5>
              <p className="card-text opacity-75 text-light small">Keras structural analysis with Grad-CAM explainability.</p>
              <Link to="/scan/fracture" className="btn btn-outline-danger w-100 mt-auto fw-bold rounded-pill">Analyze X-Ray</Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}



// --- 5. MAIN APP & ROUTING ---
function App() {
  return (
    <Router>
      <div className="min-vh-100 d-flex flex-column">
        {/* Navigation Bar */}
        <nav className="navbar navbar-expand-lg shadow-sm" style={{ background: 'rgba(15, 23, 42, 0.9)', backdropFilter: 'blur(10px)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="container py-2">
            <Link className="navbar-brand fw-bold d-flex align-items-center" to="/" style={{ color: '#f8fafc' }}>
              <i className="bi bi-shield-plus fs-4 me-2" style={{ color: '#38bdf8' }}></i>
              Sentinel Health
            </Link>
            <div className="navbar-nav ms-auto flex-row gap-4">
              <Link className="nav-link text-light fw-semibold" to="/">Dashboard</Link>
              <Link className="nav-link text-light fw-semibold" to="/docs">AI Chat</Link>
              <Link className="nav-link text-light fw-semibold" to="/vitals">Vitals</Link>
            </div>
          </div>
        </nav>

        {/* Route Configuration */}
        <div className="flex-grow-1">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/vitals" element={<VitalsPage />} />
            <Route path="/docs" element={<DocsPage />} />
            
            <Route path="/scan/tumor" element={<ScanPage title="Brain Tumor Detection" icon="bi-x-diamond" scanType="Brain MRI (Tumor)" description="Upload a Brain MRI to screen for tumor presence using PyTorch." />} />
            <Route path="/scan/lung-cancer" element={<ScanPage title="Lung Cancer Screening" icon="bi-lungs" scanType="Chest CT (Lung Cancer)" description="Upload a Chest CT scan to screen for malignant nodules." />} />
            <Route path="/scan/alzheimers" element={<ScanPage title="Alzheimer's Detection" icon="bi-person-bounding-box" scanType="Brain MRI (Alzheimer's)" description="Upload a Brain MRI to check for patterns associated with memory loss." />} />
            <Route path="/scan/fracture" element={<ScanPage title="Bone Fracture Detection" icon="bi-bandaid" scanType="Bone X-Ray (Fracture)" description="Upload a Bone X-Ray to check for structural breaks or cracks." />} />
          </Routes>
        </div>
        
        {/* Footer */}
        <footer className="text-center py-4 mt-auto border-top" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
           <p className="text-muted small mb-0"><i className="bi bi-exclamation-circle me-1"></i>Prototype built for Hackathon. Does not replace professional medical advice.</p>
        </footer>
      </div>
    </Router>
  )
}

export default App