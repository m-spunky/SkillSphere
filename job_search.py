import streamlit as st
import pandas as pd
import PyPDF2
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))



# Function to process the resume and recommend jobs
def process_resume(skills=['Deep  Learning', 'Python', 'Object  Detection', '3D Reconstruction', 'Pytorch3d', 'OpenCV', 'Python', 'Web  Service', 'Containerization', 'Image  Processing', 'C, C++', 'Generative  AI', '.Net Programming']):
    stopw  = set(stopwords.words('english'))

    jd_df=pd.read_csv(r'C:\Users\soham\Desktop\HACK AI\47_Phoenix_3\jd_structured_data.csv')

    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices


    
    # Feature Engineering:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([skills])

    
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (jd_df['Processed_JD'].values.astype('U'))

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
    
    return jd_df.head(5).sort_values('match')

# Streamlit app
def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name
        df_jobs = process_resume()

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        st.dataframe(df_jobs[['Job Title','Company Name','Location','Industry','Sector','Average Salary']])

# Run the Streamlit app
if __name__ == '__main__':
    main()