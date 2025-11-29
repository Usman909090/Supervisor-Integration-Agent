"""
HTML/React frontend served by FastAPI. Uses CDN React + Babel to keep the demo
self-contained while providing a richer UI than plain HTML.
"""
from __future__ import annotations

import json
from typing import List
from fastapi.responses import HTMLResponse

from .models import AgentMetadata


STYLES = """
          :root {
            --bg: #0f172a;
            --panel: #0b1220;
            --card: #111827;
            --accent: #22d3ee;
            --muted: #94a3b8;
            --text: #e2e8f0;
            --border: #1f2937;
            --success: #22c55e;
            --error: #ef4444;
            --glow: rgba(34,211,238,0.25);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            background:
              radial-gradient(circle at 20% 20%, #1e293b, #0f172a 45%),
              radial-gradient(circle at 80% 0%, #0ea5e9, #0f172a 35%),
              var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
          }
          .shell {
            max-width: 1100px;
            margin: 0 auto;
            padding: 48px 24px 100px;
            position: relative;
          }
          /* Ambient sparkles */
          .sparkle {
            position: absolute;
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(34,211,238,0.8);
            box-shadow: 0 0 12px var(--glow);
            animation: float 6s ease-in-out infinite;
            opacity: 0.7;
          }
          @keyframes float {
            0% { transform: translateY(0px) translateX(0px); opacity: 0.7; }
            50% { transform: translateY(-10px) translateX(6px); opacity: 1; }
            100% { transform: translateY(0px) translateX(0px); opacity: 0.7; }
          }

          .hero {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 28px;
          }
          .badge {
            display: inline-flex; gap: 8px; align-items: center;
            padding: 6px 10px; border-radius: 999px;
            background: rgba(34,211,238,0.1);
            color: var(--accent); font-weight: 600; font-size: 13px;
            border: 1px solid rgba(34,211,238,0.2);
          }
          .panel {
            background: rgba(11,18,32,0.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
          }
          textarea {
            width: 100%;
            min-height: 140px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text);
            padding: 14px;
            resize: vertical;
            font-size: 15px;
          }
          .controls {
            display: flex; align-items: center; gap: 14px; margin-top: 12px; flex-wrap: wrap;
          }
          .checkbox { display: inline-flex; gap: 8px; align-items: center; color: var(--muted); font-size: 14px; }
          button.primary {
            background: linear-gradient(120deg, #22d3ee, #22c55e);
            color: #0b1220;
            border: none;
            padding: 12px 18px;
            font-weight: 700;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
          }
          button.primary:hover { transform: translateY(-1px); box-shadow: 0 10px 30px rgba(34,211,238,0.25); filter: brightness(1.05); }
          button.primary:active { transform: translateY(0px); }
          .status { font-size: 14px; color: var(--muted); display: inline-flex; align-items: center; gap: 8px; }
          .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--glow); animation: pulse 1.2s ease-in-out infinite; }
          @keyframes pulse { 0% { transform: scale(1); opacity: .9; } 50% { transform: scale(1.35); opacity: 1; } 100% { transform: scale(1); opacity: .9; } }

          .section-title { display: flex; align-items: center; gap: 8px; margin: 18px 0 8px; font-weight: 700; color: #fff; }
          .small { color: var(--muted); font-size: 14px; }
          .result-box {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            white-space: pre-wrap;
            min-height: 60px;
            position: relative;
          }
          .copy-btn {
            position: absolute; top: 10px; right: 10px;
            font-size: 12px; padding: 6px 10px; border-radius: 999px;
            border: 1px solid var(--border); background: var(--card); color: var(--text);
            cursor: pointer;
          }

          /* Orbit layout for agents */
          .orbit {
            position: relative;
            height: 420px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(17,24,39,0.8), rgba(17,24,39,0.5));
            border: 1px solid var(--border);
            overflow: auto; /* allow scrolling when agents overflow */
            scroll-padding: 12px;
          }
          .sun {
            position: absolute; left: 50%; top: 50%;
            transform: translate(-50%, -50%);
            background: radial-gradient(circle, #22d3ee 0%, #22c55e 60%);
            width: 120px; height: 120px; border-radius: 50%;
            box-shadow: 0 0 40px rgba(34,211,238,0.35);
            display: flex; align-items: center; justify-content: center;
            color: #0b1220; font-weight: 800;
          }
          .ring {
            position: absolute; left: 50%; top: 50%;
            transform: translate(-50%, -50%);
            border: 1px dashed rgba(148,163,184,0.25);
            border-radius: 50%;
            animation: rotate 24s linear infinite;
            pointer-events: none;
          }
          .ring.r1 { width: 220px; height: 220px; animation-duration: 26s; }
          .ring.r2 { width: 280px; height: 280px; animation-duration: 32s; }
          .ring.r3 { width: 340px; height: 340px; animation-duration: 40s; }
          @keyframes rotate { from { transform: translate(-50%, -50%) rotate(0deg); } to { transform: translate(-50%, -50%) rotate(360deg); } }

          .planet {
            position: absolute;
            padding: 8px 10px;
            border-radius: 12px;
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            width: 180px;
            transition: transform 150ms ease, box-shadow 150ms ease;
            cursor: pointer;
          }
          .planet:hover { transform: translateY(-2px); box-shadow: 0 12px 34px rgba(0,0,0,0.4); }
          .planet h4 { margin: 0; font-size: 13px; color: #fff; text-align: center; }
          .planet p { display: none; }
          .pill { display: inline-flex; padding: 4px 10px; border-radius: 999px; background: rgba(34,211,238,0.12); color: var(--text); font-size: 12px; border: 1px solid rgba(34,211,238,0.2); margin-right: 6px; margin-top: 4px; }

          /* Debug dock */
          .dock {
            margin-top: 18px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
          }
          .dock-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 12px; background: rgba(17,24,39,0.7); border-bottom: 1px solid var(--border);
          }
          .dock-content { padding: 12px; }
          .timeline {
            display: grid; gap: 10px;
          }
          .timeline-item {
            display: grid; grid-template-columns: 28px 1fr auto; gap: 10px;
            align-items: center; background: var(--card); border: 1px solid var(--border);
            padding: 10px; border-radius: 10px;
          }
          .dot { width: 10px; height: 10px; border-radius: 50%; }
          .dot.success { background: var(--success); box-shadow: 0 0 12px rgba(34,197,94,0.35); }
          .dot.error { background: var(--error); box-shadow: 0 0 12px rgba(239,68,68,0.35); }
          .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
          .json-box {
            background: #0d1524; padding: 10px; border-radius: 10px;
            border: 1px solid var(--border); overflow-x: auto;
          }

          footer { margin-top: 22px; color: var(--muted); font-size: 13px; text-align: center; }
"""

COMMON_REACT = """
          const { useState, useEffect, useMemo } = React;

          const Pill = ({ text }) => <span className="pill">{text}</span>;

          const PlanetCard = ({ agent, style }) => {
            const tooltip = `${agent.description || ''}` + (agent.intents?.length ? `\nIntents: ${agent.intents.join(', ')}` : '');
            return (
              <div className="planet" style={style} title={tooltip}>
                <h4>{agent.name}</h4>
              </div>
            );
          };

          const TimelineItem = ({ item, index }) => (
            <div className="timeline-item">
              <span className={`dot ${item.status === 'success' ? 'success' : 'error'}`}></span>
              <div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <strong>{item.name}</strong>
                  <span className="mono" style={{ color: 'var(--muted)' }}>{item.intent}</span>
                </div>
                {item.output && <div className="small" style={{ marginTop: 4, color: '#cbd5e1' }}>{String(item.output).slice(0, 160)}{String(item.output).length > 160 ? '…' : ''}</div>}
                {item.error && <div className="small" style={{ marginTop: 4, color: '#fca5a5' }}>{item.error}</div>}
              </div>
              <span className="mono" style={{ color: 'var(--muted)' }}>#{index+1}</span>
            </div>
          );
"""

def _render_page(title: str, script_body: str) -> HTMLResponse:
    html_content = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />
        <style>{STYLES}</style>
      </head>
      <body>
        <div class="shell">
          <div id="root"></div>
          <!-- ambient sparkles -->
          <div class="sparkle" style="left: 6%; top: 20%; animation-delay: .2s;"></div>
          <div class="sparkle" style="left: 92%; top: 12%; animation-delay: .6s;"></div>
          <div class="sparkle" style="left: 14%; top: 78%; animation-delay: 1.1s;"></div>
          <div class="sparkle" style="left: 70%; top: 65%; animation-delay: .9s;"></div>
        </div>
        <script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script type="text/babel">
          {COMMON_REACT}
          {script_body}
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def render_home() -> HTMLResponse:
    script = """
          const App = () => {
            const initialConv = window.localStorage.getItem('conversationId') || (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
            const [query, setQuery] = useState('Summarize our project status and flag any deadline risks.');
            const [debug, setDebug] = useState(false);
            const [agents, setAgents] = useState([]);
            const [answer, setAnswer] = useState('');
            const [usedAgents, setUsedAgents] = useState([]);
            const [intermediate, setIntermediate] = useState({});
            const [status, setStatus] = useState('');
            const [error, setError] = useState(null);
            const [openIntermediate, setOpenIntermediate] = useState(false);
            const [conversationId, setConversationId] = useState(initialConv);
            const [fileName, setFileName] = useState('');

            useEffect(() => {
              fetch('/api/agents').then((r) => r.json()).then(setAgents).catch(() => setAgents([]));
              window.localStorage.setItem('conversationId', initialConv);
            }, []);

            const handleSubmit = async () => {
              if (!query.trim()) return;
              setStatus('Working...');
              setAnswer('');
              setUsedAgents([]);
              setIntermediate({});
              setError(null);
              try {
                const resp = await fetch('/api/query', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ query, user_id: null, conversation_id: conversationId, options: { debug } })
                });
                const data = await resp.json();
                setStatus('');
                setAnswer(data.answer || '');
                setUsedAgents(data.used_agents || []);
                setIntermediate(data.intermediate_results || {});
                setError(data.error);
                // confetti
                try {
                  const s = document.createElement('span');
                  s.style.position='fixed'; s.style.left='50%'; s.style.top='12%';
                  s.style.transform='translateX(-50%)';
                  s.style.filter='drop-shadow(0 0 10px rgba(34,211,238,.6))';
                  s.textContent='✨';
                  document.body.appendChild(s);
                  setTimeout(() => s.remove(), 700);
                } catch {}
              } catch (err) {
                setStatus('');
                setError({ message: 'Network error', type: 'network_error' });
              }
            };

            const handleCopy = async () => {
              try {
                await navigator.clipboard.writeText(answer || '');
              } catch {}
            };

            const handleFileUpload = (e) => {
              const file = e.target.files[0];
              if (!file) return;
              
              setFileName(file.name);
              setStatus('Reading file...');
              
              const reader = new FileReader();
              reader.onload = (event) => {
                const fileContent = event.target.result;
                // If query is empty or just placeholder, replace it; otherwise append
                if (!query.trim() || query === 'Summarize our project status and flag any deadline risks.') {
                  setQuery(`Summarize this document:\n\n${fileContent}`);
                } else {
                  setQuery(`${query}\n\n--- Document Content ---\n\n${fileContent}`);
                }
                setStatus('');
              };
              reader.onerror = () => {
                setStatus('');
                setError({ message: 'Failed to read file', type: 'file_error' });
                setFileName('');
              };
              
              // Read as text for text files, or try to read as text for other files
              if (file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md') || 
                  file.name.endsWith('.json') || file.name.endsWith('.csv') || file.name.endsWith('.log')) {
                reader.readAsText(file);
              } else {
                // For binary files like PDF, DOCX, show a message
                setStatus('');
                setError({ message: `File type ${file.type || 'unknown'} not supported for direct text reading. Please use the document URL feature or paste the content manually.`, type: 'file_error' });
                setFileName('');
              }
            };

            // Compute orbit positions (percent-based radii to keep within bounds)
            const orbitPositions = useMemo(() => {
              const rings = [34, 42, 48]; // percent radii from center for three orbits
              const items = [];
              const count = Math.max(agents.length, 6);
              agents.forEach((agent, i) => {
                const ringIndex = i % rings.length;
                const angle = (i * (360 / count)) % 360;
                const r = rings[ringIndex];
                const x = 50 + Math.cos(angle * Math.PI/180) * r;
                const y = 50 + Math.sin(angle * Math.PI/180) * r;
                // clamp to avoid clipping due to card width/height
                const clampedX = Math.min(92, Math.max(8, x));
                const clampedY = Math.min(92, Math.max(8, y));
                items.push({ agent, style: { left: `${clampedX}%`, top: `${clampedY}%` } });
              });
              return items;
            }, [agents]);

            return (
              <div className="panel">
                <div className="hero">
                  <div className="badge">Supervisor · Multi-Agent Orchestrator</div>
                  <div>
                    <h1 style={{ margin: '4px 0 6px' }}>Ask once. Let the supervisor plan the rest.</h1>
                    <p className="small">The LLM planner reads your request, selects the right worker agents, and merges their answers.</p>
                  </div>
                </div>

                <div>
                  <label className="section-title">Your request</label>
                  <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type your question here or upload a file..." />
                  <div style={{ marginTop: 8, marginBottom: 8 }}>
                    <label style={{ display: 'inline-block', cursor: 'pointer', fontSize: '13px', color: 'var(--muted)' }}>
                      <input type="file" onChange={handleFileUpload} accept=".txt,.md,.json,.csv,.log,text/*" style={{ display: 'none' }} />
                      <span style={{ textDecoration: 'underline' }}>📎 Upload text file</span>
                      {fileName && <span style={{ marginLeft: 8, color: 'var(--accent)' }}>({fileName})</span>}
                    </label>
                  </div>
                  <div className="controls">
                    <label className="checkbox">
                      <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} /> Show debug
                    </label>
                    <button className="primary" onClick={handleSubmit}>Submit</button>
                    {status && <span className="status"><span className="status-dot"></span>{status}</span>}
                  </div>
                </div>

                <div style={{ marginTop: 18, position: 'relative' }}>
                  <label className="section-title">Answer</label>
                  <div className="result-box">
                    <button className="copy-btn" onClick={handleCopy}>Copy</button>
                    {answer || (status ? '…thinking…' : 'No answer yet.')}
                  </div>
                  {error && <div className="small" style={{ color: '#f87171', marginTop: 8 }}>Error: {error.message}</div>}
                </div>

                <div style={{ marginTop: 18 }}>
                  <label className="section-title">Worker agents</label>
                  <p className="small">Visualized in orbit around the supervisor. Hover to see details.</p>
                  <div className="orbit">
                    <div className="sun">Supervisor</div>
                    <div className="ring r1"></div>
                    <div className="ring r2"></div>
                    <div className="ring r3"></div>
                    {orbitPositions.map(({ agent, style }) => (
                      <PlanetCard key={agent.name} agent={agent} style={{ position: 'absolute', transform: 'translate(-50%, -50%)', ...style }} />
                    ))}
                  </div>
                </div>

                {debug && (
                  <div className="dock">
                    <div className="dock-header">
                      <strong>Debug</strong>
                      <span className="small">Planner calls & intermediate payloads</span>
                    </div>
                    <div className="dock-content">
                      <label className="section-title">Agent call timeline</label>
                      <div className="timeline">
                        {usedAgents.map((ua, idx) => <TimelineItem key={`${ua.name}-${ua.intent}-${idx}`} item={ua} index={idx} />)}
                        {usedAgents.length === 0 && <span className="small">No agents called yet.</span>}
                      </div>

                      <div style={{ marginTop: 14 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <label className="section-title" style={{ margin: 0 }}>Intermediate results</label>
                          <button className="copy-btn" onClick={() => navigator.clipboard.writeText(JSON.stringify(intermediate, null, 2))}>Copy JSON</button>
                        </div>
                        <div style={{ marginTop: 8 }}>
                          <button className="primary" style={{ padding: '8px 12px', fontWeight: 600 }} onClick={() => setOpenIntermediate(v => !v)}>
                            {openIntermediate ? 'Hide payload' : 'Show payload'}
                          </button>
                        </div>
                        {openIntermediate && (
                          <pre className="mono json-box" style={{ marginTop: 10 }}>{JSON.stringify(intermediate, null, 2)}</pre>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                <footer>Powered by FastAPI · React · LLM planner · Worker registry</footer>
              </div>
            );
          };

          ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    """
    return _render_page("Supervisor Agent Demo", script)

def render_agents_page(agents: List[AgentMetadata]) -> HTMLResponse:
    agents_json = json.dumps([a.dict() for a in agents])
    script = f"""
          const initialAgents = {agents_json};
          const App = () => {{
            return (
              <div className="panel">
                <div className="hero">
                  <div className="badge">Registry</div>
                  <div>
                    <h1 style={{ margin: '4px 0 6px' }}>Available Worker Agents</h1>
                    <p className="small">These agents are registered and available for the supervisor to call.</p>
                  </div>
                </div>
                 <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
                  {{initialAgents.map(agent => (
                    <PlanetCard
                      key={{agent.name}}
                      agent={{agent}}
                      style={{{{ position: 'relative', width: '100%', transform: 'none' }}}}
                    />
                  ))}}
                </div>
                <footer style={{ marginTop: 40 }}>
                  <a href="/" style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 600 }}>← Back to Dashboard</a>
                </footer>
              </div>
            );
          }};
          ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    """
    return _render_page("Agents - Supervisor", script)

def render_query_page() -> HTMLResponse:
    script = """
          const App = () => {
            const [query, setQuery] = useState('');
            const [answer, setAnswer] = useState('');
            const [status, setStatus] = useState('');
            const [error, setError] = useState(null);

            const handleSubmit = async () => {
              if (!query.trim()) return;
              setStatus('Working...');
              setAnswer('');
              setError(null);
              try {
                const resp = await fetch('/api/query', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ query, user_id: null, options: { debug: false } })
                });
                const data = await resp.json();
                setStatus('');
                setAnswer(data.answer || '');
                setError(data.error);
              } catch (err) {
                setStatus('');
                setError({ message: 'Network error', type: 'network_error' });
              }
            };

            return (
              <div className="panel">
                <div className="hero">
                  <div className="badge">Query Interface</div>
                  <div>
                    <h1 style={{ margin: '4px 0 6px' }}>Submit a Request</h1>
                    <p className="small">Direct query interface without the full dashboard.</p>
                  </div>
                </div>

                <div>
                  <label className="section-title">Your request</label>
                  <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type your question here..." />
                  <div className="controls">
                    <button className="primary" onClick={handleSubmit}>Submit</button>
                    {status && <span className="status"><span className="status-dot"></span>{status}</span>}
                  </div>
                </div>

                <div style={{ marginTop: 18 }}>
                  <label className="section-title">Answer</label>
                  <div className="result-box">
                    {answer || (status ? '…thinking…' : 'No answer yet.')}
                  </div>
                  {error && <div className="small" style={{ color: '#f87171', marginTop: 8 }}>Error: {error.message}</div>}
                </div>

                <footer style={{ marginTop: 40 }}>
                  <a href="/" style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 600 }}>← Back to Dashboard</a>
                </footer>
              </div>
            );
          };
          ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    """
    return _render_page("Query - Supervisor", script)
