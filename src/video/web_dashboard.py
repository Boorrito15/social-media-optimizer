"""Real-Time Zero-Flicker MinionsScout Editorial Live Dashboard with EMA Smoothing.

Brand Identity:
- Name: MinionsScout
- Theme: Editorial Cream (#FAF8F5 / #F4F1EA), Obsidian (#111111), Dragonfruit Crimson (#E02424, #FF3B30, #991B1B)
- Ultra-smooth Exponential Moving Average (EMA) stabilized Speed & ETA backend
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

HISTORY = deque(maxlen=500)
SMOOTH_VPM = None
SMOOTH_SECONDS_LEFT = None


def background_gcs_scanner():
    global STATE, HISTORY, SMOOTH_VPM, SMOOTH_SECONDS_LEFT
    
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
            
            # Record every scan timestamp and count
            HISTORY.append((now, current_meta))
            
            # Look back across the last 60-180 seconds or last 40 downloads for a robust baseline
            lookback_seconds = 180.0
            min_downloads = 25
            
            ref_time, ref_count = HISTORY[0]
            for t_hist, c_hist in HISTORY:
                if (now - t_hist) <= lookback_seconds and (current_meta - c_hist) >= min_downloads:
                    ref_time, ref_count = t_hist, c_hist
                    break
            
            delta_count = current_meta - ref_count
            delta_time = max(1.0, now - ref_time)
            
            target_meta = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
            meta_remaining = max(0, target_meta - current_meta)
            
            if delta_count > 0 and delta_time > 5.0:
                raw_vps = delta_count / delta_time
                raw_vpm = raw_vps * 60.0
                
                # Exponential Moving Average (EMA) smoothing filter (alpha = 0.15)
                alpha = 0.15
                if SMOOTH_VPM is None:
                    SMOOTH_VPM = raw_vpm
                else:
                    SMOOTH_VPM = (alpha * raw_vpm) + ((1.0 - alpha) * SMOOTH_VPM)
                
                smooth_vps = max(0.01, SMOOTH_VPM / 60.0)
                raw_seconds_left = meta_remaining / smooth_vps
                
                if SMOOTH_SECONDS_LEFT is None:
                    SMOOTH_SECONDS_LEFT = raw_seconds_left
                else:
                    SMOOTH_SECONDS_LEFT = (0.10 * raw_seconds_left) + (0.90 * SMOOTH_SECONDS_LEFT)
                
                eta_dt = datetime.now() + timedelta(seconds=SMOOTH_SECONDS_LEFT)
                
                hours = int(SMOOTH_SECONDS_LEFT // 3600)
                minutes = int((SMOOTH_SECONDS_LEFT % 3600) // 60)
                
                eta_time = eta_dt.strftime("%H:%M")
                eta_sub = f"in ~{hours}h {minutes}m (EMA stabilized)"
                vpm_display = round(SMOOTH_VPM, 1)
                vph_display = int(SMOOTH_VPM * 60.0)
            else:
                vpm_display = 0.0
                vph_display = 0
                eta_time = "--:--"
                eta_sub = "Sampling downloads..."
            
            STATE.update({
                "facebook": fb,
                "instagram": ig,
                "tiktok": tt,
                "youtube": yt,
                "recent": recent,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "vpm": vpm_display,
                "vph": vph_display,
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
  <title>MinionsScout — Live Pipeline Center</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      background-color: #F7F5F0;
      color: #111111;
      min-height: 100vh;
      padding: 32px 24px;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1160px;
      margin: 0 auto;
    }

    .hero-banner {
      background: linear-gradient(135deg, #E02424 0%, #B91C1C 40%, #7F1D1D 100%);
      border-radius: 28px;
      padding: 36px 40px;
      color: #FFFFFF;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(224, 36, 36, 0.25);
    }
    .hero-banner::after {
      content: '';
      position: absolute;
      top: -50%;
      right: -10%;
      width: 400px;
      height: 400px;
      background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, transparent 70%);
      pointer-events: none;
    }

    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }
    .logo-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      width: 38px;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .brand-name {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .hero-tag {
      background: rgba(255, 255, 255, 0.2);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.3);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }

    .hero-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2.5rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 1.1;
      margin-bottom: 8px;
    }
    .hero-subtitle {
      font-size: 1rem;
      color: rgba(255, 255, 255, 0.85);
      max-width: 650px;
    }

    .editorial-card {
      background: #FFFFFF;
      border: 1px solid rgba(17, 17, 17, 0.08);
      border-radius: 24px;
      padding: 24px 26px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .editorial-card:hover {
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    .kpi-label {
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #737373;
      margin-bottom: 8px;
    }
    .kpi-value {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: #111111;
      line-height: 1.1;
    }
    .kpi-value.crimson {
      color: #E02424;
    }
    .kpi-sub {
      font-size: 0.82rem;
      color: #737373;
      margin-top: 6px;
      font-weight: 500;
    }

    .progress-banner {
      margin-bottom: 20px;
      background: #111111;
      color: #FFFFFF;
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
      background: rgba(255, 255, 255, 0.15);
      border-radius: 9999px;
      overflow: hidden;
      position: relative;
    }
    .progress-fill {
      height: 100%;
      border-radius: 9999px;
      background: linear-gradient(90deg, #E02424 0%, #FF5A5F 100%);
      box-shadow: 0 0 16px rgba(224, 36, 36, 0.6);
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
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
    .platform-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.2rem;
      font-weight: 700;
    }
    .platform-badge {
      font-size: 0.8rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      background: rgba(224, 36, 36, 0.1);
      color: #E02424;
    }

    .table-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }
    .table-title {
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #737373;
      margin-bottom: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th {
      text-align: left;
      padding: 8px 10px;
      color: #737373;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid rgba(17, 17, 17, 0.08);
    }
    td {
      padding: 12px 10px;
      border-bottom: 1px solid rgba(17, 17, 17, 0.05);
      color: #111111;
      font-weight: 500;
    }
    tr:last-child td { border-bottom: none; }

    .tag-pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(224, 36, 36, 0.1);
      color: #E02424;
    }

    .footer {
      text-align: center;
      color: #888888;
      font-size: 0.8rem;
      font-weight: 500;
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    
    <div class="hero-banner">
      <div class="hero-top">
        <div class="logo-group">
          <svg class="logo-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="10" fill="#FFFFFF"/>
            <circle cx="50" cy="22" r="6" fill="#FFFFFF"/>
            <circle cx="50" cy="78" r="6" fill="#FFFFFF"/>
            <circle cx="22" cy="50" r="6" fill="#FFFFFF"/>
            <circle cx="78" cy="50" r="6" fill="#FFFFFF"/>
            <circle cx="30" cy="30" r="5" fill="#FFFFFF"/>
            <circle cx="70" cy="30" r="5" fill="#FFFFFF"/>
            <circle cx="30" cy="70" r="5" fill="#FFFFFF"/>
            <circle cx="70" cy="70" r="5" fill="#FFFFFF"/>
          </svg>
          <span class="brand-name">MinionsScout</span>
        </div>
        <div class="hero-tag">
          ● 2 WORKERS LIVE (10 THREADS)
        </div>
      </div>
      
      <div class="hero-title">Make an impact.</div>
      <div class="hero-subtitle">
        Short-form video scraping, 480p H.264 transcode & GCS persistence on <span style="font-family: monospace; background: rgba(0,0,0,0.25); padding: 2px 8px; border-radius: 6px;">meta_layer_scr</span>
      </div>
    </div>

    <!-- 4 KPI Cards -->
    <div class="kpi-grid">
      <div class="editorial-card">
        <div class="kpi-label">Total Meta Ingestion</div>
        <div class="kpi-value crimson" id="kpi-meta-total">0</div>
        <div class="kpi-sub" id="kpi-meta-sub">0.0% of 9,453 videos</div>
      </div>

      <div class="editorial-card">
        <div class="kpi-label">Stabilized Speed</div>
        <div class="kpi-value" id="kpi-speed">0.0 <span style="font-size: 1.1rem; font-weight: 500; color: #737373;">/min</span></div>
        <div class="kpi-sub" id="kpi-speed-hour">~0 videos / hour</div>
      </div>

      <div class="editorial-card">
        <div class="kpi-label">EMA Smoothed ETA</div>
        <div class="kpi-value crimson" style="font-size: 1.9rem;" id="kpi-eta">--:--</div>
        <div class="kpi-sub" id="kpi-eta-sub">Calculating...</div>
      </div>

      <div class="editorial-card">
        <div class="kpi-label">GCS Target Bucket</div>
        <div class="kpi-value" style="font-size: 1.25rem; font-weight: 700; word-break: break-all;">""" + bucket + """</div>
        <div class="kpi-sub">asia-southeast2 • standard</div>
      </div>
    </div>

    <!-- Overall Meta Progress Card -->
    <div class="editorial-card progress-banner">
      <div class="progress-header">
        <span style="font-size: 1.1rem; font-family: 'Space Grotesk', sans-serif; font-weight: 700;">Overall Meta Layer Progress</span>
        <span style="color: #FF5A5F; font-weight: 700; font-family: 'Space Grotesk', sans-serif;" id="meta-banner-stat">0 / 9,453 (0.0%)</span>
      </div>
      <div class="progress-track" style="height: 12px; background: rgba(255, 255, 255, 0.1);">
        <div class="progress-fill" id="meta-progress-bar" style="width: 0%;"></div>
      </div>
    </div>

    <!-- Platform Cards (FB & IG) -->
    <div class="platform-grid">
      
      <div class="editorial-card">
        <div class="platform-header">
          <div>
            <div class="platform-title">Facebook Reels</div>
            <div style="font-size: 0.8rem; color: #737373; margin-top: 2px;">5 Parallel Concurrency Threads</div>
          </div>
          <span class="platform-badge" id="fb-pct-badge">0.0%</span>
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; margin-bottom: 2px;" id="fb-count-display">
          0 <span style="font-size: 1rem; color: #737373; font-weight: 500;">/ 4,578</span>
        </div>
        <div style="font-size: 0.85rem; color: #737373; margin-bottom: 14px;" id="fb-remaining">4,578 videos remaining</div>
        <div class="progress-track" style="background: rgba(17, 17, 17, 0.08); height: 8px;">
          <div class="progress-fill" id="fb-progress-bar" style="width: 0%; background: #111111;"></div>
        </div>
      </div>

      <div class="editorial-card">
        <div class="platform-header">
          <div>
            <div class="platform-title">Instagram Reels</div>
            <div style="font-size: 0.8rem; color: #737373; margin-top: 2px;">5 Parallel Concurrency Threads</div>
          </div>
          <span class="platform-badge" id="ig-pct-badge">0.0%</span>
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; margin-bottom: 2px;" id="ig-count-display">
          0 <span style="font-size: 1rem; color: #737373; font-weight: 500;">/ 4,875</span>
        </div>
        <div style="font-size: 0.85rem; color: #737373; margin-bottom: 14px;" id="ig-remaining">4,875 videos remaining</div>
        <div class="progress-track" style="background: rgba(17, 17, 17, 0.08); height: 8px;">
          <div class="progress-fill" id="ig-progress-bar" style="width: 0%;"></div>
        </div>
      </div>

    </div>

    <!-- Status Breakdown & Recent Artifacts -->
    <div class="table-grid">
      <div class="editorial-card">
        <div class="table-title">All Platforms Status</div>
        <table>
          <thead>
            <tr><th>Platform</th><th>Uploaded</th><th>Target</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>Facebook</td><td id="t-fb-up" style="font-weight: 700;">0</td><td>4,578</td><td id="t-fb-pct" style="color: #E02424; font-weight: 700;">0.0%</td></tr>
            <tr><td>Instagram</td><td id="t-ig-up" style="font-weight: 700;">0</td><td>4,875</td><td id="t-ig-pct" style="color: #E02424; font-weight: 700;">0.0%</td></tr>
            <tr><td>TikTok</td><td id="t-tt-up" style="font-weight: 700;">1,995</td><td>2,137</td><td style="color: #111111; font-weight: 700;">93.4%</td></tr>
            <tr><td>YouTube</td><td id="t-yt-up" style="font-weight: 700;">1,534</td><td>1,552</td><td style="color: #111111; font-weight: 700;">98.8%</td></tr>
          </tbody>
        </table>
      </div>

      <div class="editorial-card">
        <div class="table-title">Recent GCS 480p Deliverables</div>
        <table>
          <thead>
            <tr><th>GCS Object Key</th><th style="text-align: right;">Format</th></tr>
          </thead>
          <tbody id="recent-table-body">
            <tr><td colspan="2" style="color: #737373;">Loading recent uploads...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="footer">
      MinionsScout • Exponential Moving Average (EMA) Stabilized Speed & ETA • Last Ping: <span id="last-ping-time">--:--:--</span>
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
        document.getElementById('kpi-speed').innerHTML = `${data.vpm.toFixed(1)} <span style="font-size: 1.1rem; font-weight: 500; color: #737373;">/min</span>`;
        document.getElementById('kpi-speed-hour').innerText = `~${data.vph.toLocaleString()} videos / hour`;

        document.getElementById('kpi-eta').innerText = data.eta_time;
        document.getElementById('kpi-eta-sub').innerText = data.eta_sub;

        document.getElementById('meta-banner-stat').innerText = `${meta.toLocaleString()} / ${TOTAL_META.toLocaleString()} (${metaPct}%)`;
        document.getElementById('meta-progress-bar').style.width = `${Math.min(100, metaPct)}%`;

        document.getElementById('fb-pct-badge').innerText = `${fbPct}%`;
        document.getElementById('fb-count-display').innerHTML = `${fb.toLocaleString()} <span style="font-size: 1rem; color: #737373; font-weight: 500;">/ ${TOTAL_FB.toLocaleString()}</span>`;
        document.getElementById('fb-remaining').innerText = `${(TOTAL_FB - fb).toLocaleString()} videos remaining`;
        document.getElementById('fb-progress-bar').style.width = `${Math.min(100, fbPct)}%`;

        document.getElementById('ig-pct-badge').innerText = `${igPct}%`;
        document.getElementById('ig-count-display').innerHTML = `${ig.toLocaleString()} <span style="font-size: 1rem; color: #737373; font-weight: 500;">/ ${TOTAL_IG.toLocaleString()}</span>`;
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
              <td style="font-family: monospace; font-size: 0.82rem; color: #444444;">${r}</td>
              <td style="text-align: right;"><span class="tag-pill">480p Ready</span></td>
            </tr>
          `).join('');
        }

        document.getElementById('last-ping-time').innerText = data.last_updated;
      } catch (err) {
        console.error('Polling error:', err);
      }
    }

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
        print(f"\n🔥 MinionsScout Dashboard running at: http://localhost:{port}\n")
        httpd.serve_forever()


if __name__ == "__main__":
    port = 8505
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
