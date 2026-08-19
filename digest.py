import os, smtplib, feedparser
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import quote

KEYWORDS = ['"데이터 분석가" 채용', "커머스 AI 도입", "AX 기획자"]
FEED = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"

def collect(hours=24):
    cutoff, seen, items = datetime.now(timezone.utc) - timedelta(hours=hours), set(), []
    for kw in KEYWORDS:
        for e in feedparser.parse(FEED.format(q=quote(kw))).entries:
            pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if pub < cutoff or e.title in seen:
                continue
            seen.add(e.title)
            items.append({"title": e.title, "link": e.link, "kw": kw})
    return items

def send(subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = os.environ["MAIL_USER"]
    msg["To"]      = os.environ["MAIL_TO"]
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(os.environ["MAIL_USER"], os.environ["MAIL_APP_PASSWORD"])
        s.send_message(msg)