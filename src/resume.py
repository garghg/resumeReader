#import necessary modules
import fitz
from sentence_transformers import SentenceTransformer

#get sentence transformers model to use
model = SentenceTransformer("all-MiniLM-L6-v2")


def compare(resume, job):
    #open resume as a readable pymupdf doc
    doc = fitz.open(resume)
    #iterate over doc and get text from each page
    for page in doc:
        resume_text = page.get_text()
    #pass resume text and job text in model as a list
    text = [resume_text, job]
    #create embeddings to check similarities
    embeddings = model.encode(text)
    similarities = model.similarity(embeddings, embeddings)
    #convert similarity score to a percentage
    score = round(similarities.tolist()[0][1], 4)*100
    #return score as a string
    return str(score)