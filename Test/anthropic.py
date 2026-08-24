import Test.anthropic as anthropic

client = anthropic.Anthropic()

conversation = []

conversation.append({"role": "system", "content": "안녕 나는 승귤이야"})

response = client.messages.create(
    model="claude-3.0",
    max_tokens=100,
    messages=conversation
)

assistant_message = reponse.content[0].text
print(assistant_message)
conversation.append({"role": "assistant", "content": assistant_message})

conversation.append({"role": "user", "content": "내 이름이 뭐라고?"})

reponse = client.messages.create(
    model="claude-3.0",
    max_tokens=100,
    messages=conversation
)

print(response.content[0].text)