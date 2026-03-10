from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device=-1
)

def generate_answer(query: str, context: str) -> str:
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"
    response = generator(prompt, max_new_tokens=300, do_sample=False)
    return response[0]["generated_text"].strip()