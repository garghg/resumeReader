import fitz
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def compare(resume, job):
    doc = fitz.open(resume)
    for page in doc:
        resume_text = page.get_text()
    text = [resume_text, job]
    embeddings = model.encode(text)
    similarities = model.similarity(embeddings, embeddings)
    score = round(similarities.tolist()[0][1], 4)*100
    print('score: '+ str(score))
    return str(score)