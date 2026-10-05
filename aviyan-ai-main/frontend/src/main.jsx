import React from 'react';
import { createRoot } from 'react-dom/client';
import { Bot, Code2, Files, Globe2, Rocket, Search, Settings, Sparkles } from 'lucide-react';
import './styles.css';

function App(){
  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><div className="logo">✦</div><div><b>AVIYAN</b><small>AB DEV STUDIO</small></div></div>
      <button className="newchat">+ New chat</button>
      <nav>
        <a className="active"><Bot/> Chat</a><a><Code2/> Coding</a><a><Globe2/> Research</a><a><Files/> Files</a><a><Rocket/> Deploy</a>
      </nav>
      <div className="side-bottom"><a><Search/> Search</a><a><Settings/> Settings</a></div>
    </aside>
    <main>
      <header><span>AVIYAN Workspace</span><span className="status">● Online</span></header>
      <section className="hero"><div className="orb"><Sparkles/></div><h1>Build with AVIYAN</h1><p>Your AI workspace for reasoning, coding, research and deployment.</p></section>
      <div className="composer"><textarea placeholder="Ask AVIYAN anything..."/><div className="composer-row"><span>Attach · Tools · Model</span><button>Send ↗</button></div></div>
    </main>
  </div>
}
createRoot(document.getElementById('root')).render(<App/>);
