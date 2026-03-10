from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

response = llm.invoke("Explain artificial intelligence")

print(response)