from openai import OpenAI
import rich

client = OpenAI()

default_model = "gpt-5-mini"

def stream_chat_completion(prompt, model):
    strem = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="")

def stream_response(prompt, model):
    with client.responses.stream(model=model, input=prompt) as stream:
        for event in stream:
            if "output_text" in event.type:
                rich.print(event)
    rich.print(stream.get_final_response())

if __name__ == "__main__":
    stream_chat_completion("스트리밍이 뭔가요?", default_model)
    stream_response("점심 메뉴 추천 해주세요.", default_model)