"""Background progress monitor & notifier for the video scraping pipeline.

Supports:
- Telegram Bot notifications (via TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID)
- Lark / Feishu Webhook notifications (via LARK_WEBHOOK_URL)
- Live console / text status with ETA and speed calculation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import gcs_processed_bucket, get_optional
from utils.gcs import list_existing_objects

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
    "tiktok": 2137,
    "youtube": 1552,
}


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send a notification message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[monitor] Telegram send failed: {exc}", file=sys.stderr)
        return False


def send_lark(webhook_url: str, title: str, text: str) -> bool:
    """Send a card/message to a Lark / Feishu custom bot webhook."""
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": text},
                },
                {
                    "tag": "hr",
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"Updated at {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}",
                        }
                    ],
                },
            ],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[monitor] Lark send failed: {exc}", file=sys.stderr)
        return False


def format_status_report(
    counts: dict[str, int],
    start_counts: dict[str, int],
    start_time: float,
) -> tuple[str, str]:
    """Generate a formatted plain text / HTML summary with speed & ETA."""
    now = time.time()
    elapsed_sec = max(1.0, now - start_time)
    
    meta_now = counts.get("facebook", 0) + counts.get("instagram", 0)
    meta_start = start_counts.get("facebook", 0) + start_counts.get("instagram", 0)
    meta_delta = meta_now - meta_start
    
    meta_target = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
    meta_remaining = max(0, meta_target - meta_now)
    
    vpm = (meta_delta / elapsed_sec) * 60.0
    vph = vpm * 60.0
    
    if vpm > 0.01:
        minutes_left = meta_remaining / vpm
        eta_time = datetime.now() + timedelta(minutes=minutes_left)
        eta_str = f"{eta_time.strftime("%H:%M")} (in ~{int(minutes_left // 60)}h {int(minutes_left % 60)}m)"
    else:
        eta_str = "Calculating..."

    fb = counts.get("facebook", 0)
    ig = counts.get("instagram", 0)
    tt = counts.get("tiktok", 0)
    yt = counts.get("youtube", 0)

    fb_pct = (fb / TOTAL_TARGETS["facebook"]) * 100
    ig_pct = (ig / TOTAL_TARGETS["instagram"]) * 100
    meta_pct = (meta_now / meta_target) * 100

    text_msg = (
        f"🚀 <b>Social Media Optimizer — Scraper Progress</b>\n\n"
        f"🔵 <b>Facebook</b>: <code>{fb:,} / {TOTAL_TARGETS["facebook"]:,}</code> ({fb_pct:.1f}%)\n"
        f"🟣 <b>Instagram</b>: <code>{ig:,} / {TOTAL_TARGETS["instagram"]:,}</code> ({ig_pct:.1f}%)\n"
        f"📊 <b>Total Meta</b>: <code>{meta_now:,} / {meta_target:,}</code> ({meta_pct:.1f}%)\n"
        f"🎵 TikTok: {tt:,} | 🔴 YouTube: {yt:,}\n\n"
        f"⚡ <b>Speed</b>: <b>{vpm:.1f} videos/min</b> (~{int(vph)} / hour)\n"
        f"⏳ <b>ETA</b>: <b>{eta_str}</b>\n"
        f"📦 <b>Remaining</b>: {meta_remaining:,} videos\n"
        f"⏱ <b>Session</b>: {int(elapsed_sec // 60)} min"
    )

    lark_md = (
        f"**🔵 Facebook**: `{fb:,} / {TOTAL_TARGETS["facebook"]:,}` ({fb_pct:.1f}%)\n"
        f"**🟣 Instagram**: `{ig:,} / {TOTAL_TARGETS["instagram"]:,}` ({ig_pct:.1f}%)\n"
        f"**📊 Total Meta (IG+FB)**: `{meta_now:,} / {meta_target:,}` ({meta_pct:.1f}%)\n\n"
        f"⚡ **Speed**: `{vpm:.1f} videos/min` (~{int(vph)} videos/h)\n"
        f"⏳ **ETA**: **{eta_str}**\n"
        f"📦 **Remaining**: `{meta_remaining:,}` videos"
    )

    return text_msg, lark_md


def monitor_loop(
    interval_seconds: int = 300,
    telegram_token: str | None = None,
    telegram_chat_id: str | None = None,
    lark_webhook: str | None = None,
) -> None:
    bucket = gcs_processed_bucket()
    print(f"[monitor] Starting monitor on bucket: {bucket}")
    print(f"[monitor] Update interval: {interval_seconds} seconds")
    
    telegram_token = telegram_token or get_optional("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = telegram_chat_id or get_optional("TELEGRAM_CHAT_ID")
    lark_webhook = lark_webhook or get_optional("LARK_WEBHOOK_URL")

    if telegram_token and telegram_chat_id:
        print("[monitor] Telegram notifications ENABLED")
    if lark_webhook:
        print("[monitor] Lark notifications ENABLED")

    start_time = time.time()
    
    objs = list_existing_objects(bucket, prefix="videos/")
    start_counts = {
        "facebook": sum(1 for o in objs if "videos/facebook/" in o),
        "instagram": sum(1 for o in objs if "videos/instagram/" in o),
        "tiktok": sum(1 for o in objs if "videos/tiktok/" in o),
        "youtube": sum(1 for o in objs if "videos/youtube/" in o),
    }

    msg_html, msg_lark = format_status_report(start_counts, start_counts, start_time)
    print("\n" + msg_html.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "") + "\n")

    if telegram_token and telegram_chat_id:
        send_telegram(telegram_token, telegram_chat_id, "🤖 <b>Scraper Monitor Started</b>\n\n" + msg_html)
    if lark_webhook:
        send_lark(lark_webhook, "🤖 Scraper Monitor Started", msg_lark)

    while True:
        time.sleep(interval_seconds)
        try:
            objs = list_existing_objects(bucket, prefix="videos/")
            counts = {
                "facebook": sum(1 for o in objs if "videos/facebook/" in o),
                "instagram": sum(1 for o in objs if "videos/instagram/" in o),
                "tiktok": sum(1 for o in objs if "videos/tiktok/" in o),
                "youtube": sum(1 for o in objs if "videos/youtube/" in o),
            }
            msg_html, msg_lark = format_status_report(counts, start_counts, start_time)
            print(f"[{datetime.now().strftime("%H:%M:%S")}] FB: {counts["facebook"]} | IG: {counts["instagram"]}")

            if telegram_token and telegram_chat_id:
                send_telegram(telegram_token, telegram_chat_id, msg_html)
            if lark_webhook:
                send_lark(lark_webhook, "📊 Scraper Progress Update", msg_lark)

            if counts["facebook"] >= TOTAL_TARGETS["facebook"] and counts["instagram"] >= TOTAL_TARGETS["instagram"]:
                done_msg = "🎉 <b>ALL Instagram & Facebook videos have been completely scraped!</b>"
                if telegram_token and telegram_chat_id:
                    send_telegram(telegram_token, telegram_chat_id, done_msg)
                if lark_webhook:
                    send_lark(lark_webhook, "🎉 Scraping Complete", done_msg)
                print(done_msg)
                break

        except Exception as e:
            print(f"[monitor] Error fetching GCS objects: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor video scraping progress")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--telegram-token", type=str, default=None)
    parser.add_argument("--telegram-chat-id", type=str, default=None)
    parser.add_argument("--lark-webhook", type=str, default=None)
    args = parser.parse_args()

    monitor_loop(
        interval_seconds=args.interval,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id,
        lark_webhook=args.lark_webhook,
    )
