import re
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))
from pyresparser import ResumeParser
import os
from docx import Document
import skills_extraction as skills_extraction

# Load dataset:
jd_df=pd.read_csv(r'C:\Users\soham\Desktop\HACK AI\47_Phoenix_3\jd_structured_data.csv')

# Load the extracted resume skills:
file_path=r'C:\Users\soham\Desktop\HACK AI\47_Phoenix_3\Resume\CV.pdf'
skills=[]
skills.append("python","scala","shell")


vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(skills)

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
jd_test = (jd_df['Processed_JD'].values.astype('U'))

def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices

distances, indices = getNearestN(jd_test)
test = list(jd_test) 
matches = []

for i,j in enumerate(indices):
    dist=round(distances[i][0],2)
  
    temp = [dist]
    matches.append(temp)
    
matches = pd.DataFrame(matches, columns=['Match confidence'])

# Following recommends Top 5 Jobs based on candidate resume:
jd_df['match']=matches['Match confidence']
jd_df.head(5).sort_values('match')