import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수(.env) 불러오기
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# 2. 구글 Gemini API 서버를 바라보도록 OpenAI 클라이언트 설정
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_responses(prompt, model="gemini-3.6-flash"):
    # 3. 구글 Gemini 호환 Chat Completions API 사용
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 주어진 문서를 읽고 핵심 내용을 요약해 주는 친절한 AI 비서입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    # 4. 응답 구조에서 텍스트 꺼내기
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = """
    OpenAI의 Responses API에 대해 공부하고 있습니다. 
    이 API의 특징과 주요 기능이 무엇인지 알기 쉽게 요약 정리해 주세요.
    """
    
    output = get_responses(prompt)
    print("\n--- AI 응답 ---")
    print(output)