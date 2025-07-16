#import necessary modules
import fitz
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load('resume_model')

def getSkills(texts):
    for text in texts:
        doc = nlp(text)
        print('Entities: ', [(ent.text, ent.label_) for ent in doc.ents])
        print()

#get sentence transformers model to use
model = SentenceTransformer("all-MiniLM-L6-v2")


def compare(resume, job):
    #open resume as a readable pymupdf doc
    doc = fitz.open(resume)
    #iterate over doc and get text from each page
    for page in doc:
        resume_text = page.get_text()
    #pass resume text and job text in model as a list
    texts = [resume_text, job]
    #pass texts into NLP model
    getSkills(texts)
    #create embeddings to check similarities
    embeddings = model.encode(texts)
    similarities = model.similarity(embeddings, embeddings)
    #convert similarity score to a percentage
    score = round(similarities.tolist()[0][1], 4)*100
    #return score as a string
    return str(score)