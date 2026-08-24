import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수 불러오기 (.env 파일 읽기)
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# 2. 구글 Gemini API 서버를 바라보도록 클라이언트 설정 (핵심 포인트!)
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 3. 구글의 무료 가성비 모델인 'gemini-2.5-flash'를 지정
def get_chat_completion(prompt, model="gemini-3.6-flash"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 친절하고 도움이 되는 AI 비서입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# 4. 실행 부분
if __name__ == "__main__":
    user_prompt = input("AI에게 물어볼 질문을 입력하세요: ")
    response = get_chat_completion(user_prompt)
    print("\nAI 응답:")
    print(response)