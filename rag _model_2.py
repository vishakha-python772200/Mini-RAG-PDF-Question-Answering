
# Using Sentence Transformers (Mini BERT)
# =========================

from pypdf import PdfReader # this library used for pdf reading
from sentence_transformers import SentenceTransformer # senternce transform navacha AI model ghete he library
from sklearn.metrics.pairwise import cosine_similarity # dont text madhe similarity kiti te baghte he library
import re # sentence madhe divided karte
import numpy as np # index shodyla no chya kamsathi used hote 

# -------- STEP 1: READ PDF --------
reader = PdfReader(r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Rag model_2\notes.pdf")
text = "" # empty string create hoto 

for page in reader.pages:
    text += page.extract_text() # add all text in pdf

print("pdf Loaded successfully")

# -------- STEP 2: Split into sentences --------
sentences = re.split(r'(?<=[.!?]) +', text) # yala sentence madhe divide karto

# Remove very small sentences
sentences = [s.strip() for s in sentences if len(s) > 20] # je unnecessary words removed karta ex chapter 1,chapter 2 etc

print("Text Split into Sentences")

# -------- STEP 3: Load AI Model --------
model = SentenceTransformer('all-MiniLM-L6-v2') # he model load karte mini bert navacha sentence ghete meaning ghete tyla convert into no
#("all-MiniLM-L6-V2") # model cha for (EX.vivo y20) 
print("al model loaded")

# -------- STEP 4: Convert Sentences to Vectors --------
sentence_embeddings = model.encode(sentences) # convert sentence into no vector create karto

print(" Sentences Converted to embeddings")

# -------- STEP 5: Ask Question --------
while True: # repetead question vicharu shakto apan 
    question = input("\n ask your question (type exit to stop) : ",)

    if question.lower() == "exit": # exit takla stop hun jata
        break

    question_embedding = model.encode([question]) # question also convert in no formate

    similarities = cosine_similarity(question_embedding, sentence_embeddings) # similarites check karto user chya questions and sentence madhe

    best_index = np.argmax(similarities)# jya sentence similarity high ti gheto
    best_score = similarities[0][best_index] # best score of same sentence

    if best_score < 0.40:
        print("\n sorry, relevant answer not found in PDF.") # ar score ha tya sentence cha less than 0.40 then not found ans deto 
    else:
        print("\n  best answer:")
        print(sentences[best_index]) # match conditon print karto 
        print(f"\n  similarity score: {best_score:.2f}") # decimal madhe 2 no dkahv mahnje score 0 adn 1 madhe yeto 0.872930 as na dakhvt
        # 2 decimal madhe digit dakhv fakt(value round-up kar)
        
