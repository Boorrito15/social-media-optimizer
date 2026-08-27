"""Real-Time Zero-Flicker Apple Design Live Dashboard.

Features:
- Pure Apple Human Interface Design (SF Pro, glassmorphism, vibrancy, dynamic badges)
- Real-time 30-download rolling window ETA algorithm (extremely accurate & reactive)
- 1000ms WebSocket-free poll with 60 FPS CSS spring physics
- Zero screen dimming, zero page reload
"""

from collections import deque
import http.server
import json
import socketserver
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import gcs_processed_bucket
from utils.gcs import list_existing_objects

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
    "tiktok": 2137,
    "youtube": 1552,
}

bucket = gcs_processed_bucket()

# Global thread-safe state cache
STATE = {
    "facebook": 0,
    "instagram": 0,
    "tiktok": 0,
    "youtube": 0,
    "recent": [],
    "last_updated": "Initializing...",
    "vpm": 0.0,
    "vph": 0,
    "eta_time": "--:--",
    "eta_sub": "Calibrating speed...",
    "window_size": 0,
}

# Rolling history: list of (timestamp, count) tuples
HISTORY = deque(maxlen=300)


def background_gcs_scanner():
    """Continuously scan GCS and compute rolling 30-download speed & ETA."""
    global STATE, HISTORY
    
    while True:
        try:
            objs = list_existing_objects(bucket, prefix="videos/")
            fb = sum(1 for o in objs if "videos/facebook/" in o)
            ig = sum(1 for o in objs if "videos/instagram/" in o)
            tt = sum(1 for o in objs if "videos/tiktok/" in o)
            yt = sum(1 for o in objs if "videos/youtube/" in o)
            
            recent = [o.replace("videos/", "") for o in sorted(objs, reverse=True) if o.endswith(".mp4")][:8]
            
            current_meta = fb + ig
            now = time.time()
            
            # Record history point
            if not HISTORY or HISTORY[-1][1] != current_meta:
                HISTORY.append((now, current_meta))
            
            # Rolling 30-download calculation:
            # Find the sample roughly 30 downloads ago (or oldest available if < 30)
            target_lookback_downloads = 30
            ref_time, ref_count = HISTORY[0]
            
            for t_hist, c_hist in reversed(HISTORY):
                if (current_meta - c_hist) >= target_lookback_downloads:
                    ref_time, ref_count = t_hist, c_hist
                    break
            
            delta_count = current_meta - ref_count
            delta_time = max(1.0, now - ref_time)
            
            target_meta = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
            meta_remaining = max(0, target_meta - current_meta)
            
            if delta_count > 0 and delta_time > 1.0:
                vps = delta_count / delta_time
                vpm = vps * 60.0
                vph = int(vps * 3600.0)
                
                seconds_left = meta_remaining / vps
                eta_dt = datetime.now() + timedelta(seconds=seconds_left)
                
                hours = int(seconds_left // 3600)
                minutes = int((seconds_left % 3600) // 60)
                
                eta_time = eta_dt.strftime("%H:%M")
                eta_sub = f"in ~{hours}h {minutes}m (last {delta_count} vids)"
            else:
                vpm = 0.0
                vph = 0
                eta_time = "--:--"
                eta_sub = "Sampling downloads..."
            
            STATE.update({
                "facebook": fb,
                "instagram": ig,
                "tiktok": tt,
                "youtube": yt,
                "recent": recent,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "vpm": round(vpm, 1),
                "vph": vph,
                "eta_time": eta_time,
                "eta_sub": eta_sub,
                "window_size": delta_count,
            })
        except Exception as exc:
            print(f"[web-dashboard] GCS scan error: {exc}", file=sys.stderr)
        
        time.sleep(2.0)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Media Ingestion — Apple Command Center</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
      background: #000000;
      background-image: 
        radial-gradient(at 0% 0%, rgba(41, 151, 255, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(225, 48, 108, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(175, 82, 222, 0.08) 0px, transparent 60%);
      color: #f5f5f7;
      min-height: 100vh;
      padding: 32px 24px;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
    }

    .glass-card {
      background: rgba(28, 28, 30, 0.65);
      backdrop-filter: blur(40px);
      -webkit-backdrop-filter: blur(40px);
      border: 1px solid rgba(255, 255, 255, 0.09);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
      transition: border-color 0.3s ease;
    }
    .glass-card:hover {
      border-color: rgba(255, 255, 255, 0.18);
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
    }
    .title-group h1 {
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -0.04em;
      color: #ffffff;
      margin-bottom: 4px;
    }
    .title-group p {
      font-size: 0.95rem;
      color: #86868b;
    }
    .branch-badge {
      color: #2997ff;
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 0.85rem;
      background: rgba(41, 151, 255, 0.12);
      padding: 2px 8px;
      border-radius: 6px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(48, 209, 88, 0.12);
      color: #30d158;
      border: 1px solid rgba(48, 209, 88, 0.25);
      padding: 8px 16px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #30d158;
      box-shadow: 0 0 12px #30d158;
      animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    .kpi-card {
      padding: 20px 22px;
    }
    .kpi-label {
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #86868b;
      margin-bottom: 6px;
    }
    .kpi-value {
      font-size: 2.1rem;
      font-weight: 700;
      letter-spacing: -0.04em;
      color: #ffffff;
      line-height: 1.1;
    }
    .kpi-sub {
      font-size: 0.82rem;
      color: #86868b;
      margin-top: 6px;
    }

    .progress-banner {
      margin-bottom: 20px;
    }
    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      font-weight: 600;
    }
    .progress-track {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
      position: relative;
    }
    .progress-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .meta-fill {
      background: linear-gradient(90deg, #2997ff 0%, #af52de 50%, #ff2d55 100%);
      box-shadow: 0 0 20px rgba(175, 82, 222, 0.4);
    }

    .platform-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    .platform-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .platform-icon-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .platform-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 1.1rem;
    }
    .fb-icon { background: rgba(24, 119, 242, 0.18); color: #2997ff; }
    .ig-icon { background: rgba(225, 48, 108, 0.18); color: #ff2d55; }
    .fb-fill { background: #1877F2; box-shadow: 0 0 16px rgba(24, 119, 242, 0.4); }
    .ig-fill { background: linear-gradient(90deg, #af52de, #ff2d55); box-shadow: 0 0 16px rgba(225, 48, 108, 0.4); }

    .table-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    .table-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: #86868b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th {
      text-align: left;
      padding: 8px 12px;
      color: #86868b;
      font-weight: 500;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #f5f5f7;
    }
    tr:last-child td { border-bottom: none; }

    .footer {
      text-align: center;
      color: #86868b;
      font-size: 0.8rem;
      margin-top: 24px;
    }
  </style>
</head>
<body>
  <div class="container">
    
    <div class="header">
      <div class="title-group">
        <h1>Pipeline Command Center</h1>
        <p>Short-form video scraping, 480p H.264 transcode & GCS sync on <span class="branch-badge">meta_layer_scr</span></p>
      </div>
      <div>
        <div class="status-pill">
          <div class="pulse-dot"></div>
          <span>2 Workers Live (10 Threads)</span>
        </div>
      </div>
    </div>

    <!-- 4 KPI Cards -->
    <div class="kpi-grid">
      <div class="glass-card kpi-card">
        <div class="kpi-label">Total Meta Ingestion</div>
        <div class="kpi-value" id="kpi-meta-total">0</div>
        <div class="kpi-sub" id="kpi-meta-sub">0.0% of 9,453 videos</div>
      </div>
      <div class="glass-card kpi-card">
        <div class="kpi-label">Current Speed</div>
        <div class="kpi-value" style="color: #2997ff;" id="kpi-speed">0.0 <span style="font-size: 1.1rem; font-weight: 500;">/min</span></div>
        <div class="kpi-sub" id="kpi-speed-hour">~0 videos / hour</div>
      </div>
      <div class="glass-card kpi-card">
        <div class="kpi-label">ETA (Last 30 Vids)</div>
        <div class="kpi-value" style="color: #30d158; font-size: 1.85rem;" id="kpi-eta">--:--</div>
        <div class="kpi-sub" id="kpi-eta-sub">Calculating...</div>
      </div>
      <div class="glass-card kpi-card">
        <div class="kpi-label">GCS Target Bucket</div>
        <div class="kpi-value" style="font-size: 1.2rem; color: #af52de; word-break: break-all;">""" + bucket + """</div>
        <div class="kpi-sub">asia-southeast2 • standard</div>
      </div>
    </div>

    <!-- Overall Meta Progress Card -->
    <div class="glass-card progress-banner">
      <div class="progress-header">
        <span style="font-size: 1.05rem; letter-spacing: -0.02em;">Overall Meta Layer Progress</span>
        <span style="color: #2997ff; font-weight: 700;" id="meta-banner-stat">0 / 9,453 (0.0%)</span>
      </div>
      <div class="progress-track" style="height: 12px;">
        <div class="progress-fill meta-fill" id="meta-progress-bar" style="width: 0%;"></div>
      </div>
    </div>

    <!-- Platform Cards (FB & IG) -->
    <div class="platform-grid">
      
      <!-- Facebook Card -->
      <div class="glass-card">
        <div class="platform-header">
          <div class="platform-icon-title">
            <div class="platform-icon fb-icon">f</div>
            <div>
              <div style="font-weight: 600; font-size: 1.05rem;">Facebook Reels</div>
              <div style="font-size: 0.8rem; color: #86868b;">5 Parallel Concurrency Threads</div>
            </div>
          </div>
          <div style="font-weight: 700; font-size: 1.3rem; color: #2997ff;" id="fb-pct-badge">0.0%</div>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 2px;" id="fb-count-display">
          0 <span style="font-size: 0.95rem; color: #86868b; font-weight: 400;">/ 4,578</span>
        </div>
        <div style="font-size: 0.82rem; color: #86868b; margin-bottom: 14px;" id="fb-remaining">4,578 videos remaining</div>
        <div class="progress-track">
          <div class="progress-fill fb-fill" id="fb-progress-bar" style="width: 0%;"></div>
        </div>
      </div>

      <!-- Instagram Card -->
      <div class="glass-card">
        <div class="platform-header">
          <div class="platform-icon-title">
            <div class="platform-icon ig-icon">📸</div>
            <div>
              <div style="font-weight: 600; font-size: 1.05rem;">Instagram Reels</div>
              <div style="font-size: 0.8rem; color: #86868b;">5 Parallel Concurrency Threads</div>
            </div>
          </div>
          <div style="font-weight: 700; font-size: 1.3rem; color: #ff2d55;" id="ig-pct-badge">0.0%</div>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 2px;" id="ig-count-display">
          0 <span style="font-size: 0.95rem; color: #86868b; font-weight: 400;">/ 4,875</span>
        </div>
        <div style="font-size: 0.82rem; color: #86868b; margin-bottom: 14px;" id="ig-remaining">4,875 videos remaining</div>
        <div class="progress-track">
          <div class="progress-fill ig-fill" id="ig-progress-bar" style="width: 0%;"></div>
        </div>
      </div>

    </div>

    <!-- Status Breakdown & Recent Artifacts -->
    <div class="table-grid">
      <div class="glass-card">
        <div class="table-title">All Platforms Status</div>
        <table>
          <thead>
            <tr><th>Platform</th><th>Uploaded</th><th>Target</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>🔵 Facebook</td><td id="t-fb-up">0</td><td>4,578</td><td id="t-fb-pct">0.0%</td></tr>
            <tr><td>🟣 Instagram</td><td id="t-ig-up">0</td><td>4,875</td><td id="t-ig-pct">0.0%</td></tr>
            <tr><td>🎵 TikTok</td><td id="t-tt-up">1,995</td><td>2,137</td><td>93.4%</td></tr>
            <tr><td>🔴 YouTube</td><td id="t-yt-up">1,534</td><td>1,552</td><td>98.8%</td></tr>
          </tbody>
        </table>
      </div>

      <div class="glass-card">
        <div class="table-title">Recent GCS 480p Deliverables</div>
        <table>
          <thead>
            <tr><th>GCS Object Key</th><th>Format</th></tr>
          </thead>
          <tbody id="recent-table-body">
            <tr><td colspan="2" style="color: #86868b;">Loading recent uploads...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="footer">
      Ultra-responsive Rolling 30-Download ETA Algorithm • Zero-Reload Fluid Updates • Last Ping: <span id="last-ping-time">--:--:--</span>
    </div>

  </div>

  <script>
    const TOTAL_FB = 4578;
    const TOTAL_IG = 4875;
    const TOTAL_META = TOTAL_FB + TOTAL_IG;

    async function pollStats() {
      try {
        const res = await fetch('/api/stats');
        if (!res.ok) return;
        const data = await res.json();

        const fb = data.facebook;
        const ig = data.instagram;
        const meta = fb + ig;
        const fbPct = ((fb / TOTAL_FB) * 100).toFixed(1);
        const igPct = ((ig / TOTAL_IG) * 100).toFixed(1);
        const metaPct = ((meta / TOTAL_META) * 100).toFixed(1);

        document.getElementById('kpi-meta-total').innerText = meta.toLocaleString();
        document.getElementById('kpi-meta-sub').innerText = `${metaPct}% of ${TOTAL_META.toLocaleString()} videos`;
        document.getElementById('kpi-speed').innerHTML = `${data.vpm.toFixed(1)} <span style="font-size: 1.1rem; font-weight: 500;">/min</span>`;
        document.getElementById('kpi-speed-hour').innerText = `~${data.vph.toLocaleString()} videos / hour`;

        document.getElementById('kpi-eta').innerText = data.eta_time;
        document.getElementById('kpi-eta-sub').innerText = data.eta_sub;

        document.getElementById('meta-banner-stat').innerText = `${meta.toLocaleString()} / ${TOTAL_META.toLocaleString()} (${metaPct}%)`;
        document.getElementById('meta-progress-bar').style.width = `${Math.min(100, metaPct)}%`;

        document.getElementById('fb-pct-badge').innerText = `${fbPct}%`;
        document.getElementById('fb-count-display').innerHTML = `${fb.toLocaleString()} <span style="font-size: 0.95rem; color: #86868b; font-weight: 400;">/ ${TOTAL_FB.toLocaleString()}</span>`;
        document.getElementById('fb-remaining').innerText = `${(TOTAL_FB - fb).toLocaleString()} videos remaining`;
        document.getElementById('fb-progress-bar').style.width = `${Math.min(100, fbPct)}%`;

        document.getElementById('ig-pct-badge').innerText = `${igPct}%`;
        document.getElementById('ig-count-display').innerHTML = `${ig.toLocaleString()} <span style="font-size: 0.95rem; color: #86868b; font-weight: 400;">/ ${TOTAL_IG.toLocaleString()}</span>`;
        document.getElementById('ig-remaining').innerText = `${(TOTAL_IG - ig).toLocaleString()} videos remaining`;
        document.getElementById('ig-progress-bar').style.width = `${Math.min(100, igPct)}%`;

        document.getElementById('t-fb-up').innerText = fb.toLocaleString();
        document.getElementById('t-fb-pct').innerText = `${fbPct}%`;
        document.getElementById('t-ig-up').innerText = ig.toLocaleString();
        document.getElementById('t-ig-pct').innerText = `${igPct}%`;
        document.getElementById('t-tt-up').innerText = data.tiktok.toLocaleString();
        document.getElementById('t-yt-up').innerText = data.youtube.toLocaleString();

        if (data.recent && data.recent.length > 0) {
          const tbody = document.getElementById('recent-table-body');
          tbody.innerHTML = data.recent.map(r => `
            <tr>
              <td style="font-family: ui-monospace, monospace; color: #86868b; font-size: 0.82rem;">${r}</td>
              <td><span style="background: rgba(48, 209, 88, 0.12); color: #30d158; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">480p Ready</span></td>
            </tr>
          `).join('');
        }

        document.getElementById('last-ping-time').innerText = data.last_updated;
      } catch (err) {
        console.error('Polling error:', err);
      }
    }

    // Fast 1s polling interval - instant millisecond updates with zero DOM flicker
    setInterval(pollStats, 1000);
    pollStats();
  </script>
</body>
</html>
"""

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(STATE).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))


def run_server(port: int = 8505):
    t = threading.Thread(target=background_gcs_scanner, daemon=True)
    t.start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"\n🍏 Apple Live Dashboard running at: http://localhost:{port}\n")
        httpd.serve_forever()


if __name__ == "__main__":
    port = 8505
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
