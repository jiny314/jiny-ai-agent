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

if __name__ == "__main__":
    items = collect()
    if items:
        # 뉴스 항목들을 메일 본문 텍스트로 변환
        body = "\n\n".join([f"[{item['kw']}] {item['title']}\n{item['link']}" for item in items])
        send("오늘의 뉴스 다이제스트", body)
        print(f"성공: {len(items)}개의 뉴스를 메일로 발송했습니다.")
    else:
        print("수집된 최신 뉴스가 없습니다.")