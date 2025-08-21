import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document
import PyPDF2

# Load the trained model and TF-IDF vectorizer
try:
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    dtree = pickle.load(open('dtree.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or TF-IDF vectorizer file not found. Please ensure 'tfidf.pkl' and 'dtree.pkl' are in the same directory.")
    st.stop()

# Define the CleanResume function
def CleanResume(txt):
    Cleantxt = re.sub('https\\S+\\s', ' ', txt)
    Cleantxt = re.sub('\\s+', ' ', Cleantxt)
    Cleantxt = re.sub('#\\S+\\s', ' ', Cleantxt)
    Cleantxt = re.sub('@\\S+\\s', ' ', Cleantxt)
    Cleantxt = re.sub('[%s]' % re.escape("""!"&#$@^()':<>?|*{/}][,._+=-~`'"""), ' ', Cleantxt)
    Cleantxt = re.sub('RT|cc', ' ', Cleantxt)
    Cleantxt = Cleantxt.lower()
    return Cleantxt

# Define the category mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Streamlit app title and description
st.title("Resume Category Predictor")
st.write("Upload your resume or paste the text to get a predicted category.")

# Input method selection
input_method = st.radio("Choose input method:", ("Paste Resume Text", "Upload Resume File"))

resume_text = ""

if input_method == "Paste Resume Text":
    resume_text = st.text_area("Paste your resume text here:")
elif input_method == "Upload Resume File":
    uploaded_file = st.file_uploader("Upload a resume file (PDF, TXT, or DOCX)", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            try:
                resume_text = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                try:
                    resume_text = uploaded_file.getvalue().decode("latin-1")
                except Exception as e:
                     st.error(f"Error reading text file: {e}")

        elif file_extension == 'docx':
            try:
                doc = Document(uploaded_file)
                resume_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading docx file: {e}")
        elif file_extension == 'pdf':
             try:
                reader = PyPDF2.PdfReader(uploaded_file)
                resume_text = ""
                for page_num in range(len(reader.pages)):
                    resume_text += reader.pages[page_num].extract_text()
             except Exception as e:
                st.error(f"Error reading PDF file: {e}")


if st.button("Predict Category", key="predict_button_1") and resume_text:
    # Clean the input resume
    cleaned_resume = CleanResume(resume_text)

    # Transform the cleaned resume using the trained TfidfVectorizer
    input_features = tfidf.transform([cleaned_resume])

    # Make the prediction using the loaded classifier
    prediction_id = dtree.predict(input_features)[0]

    # Map category ID to category name
    category_name = category_mapping.get(prediction_id, "Unknown")

    st.subheader("Predicted Category:")
    st.write(category_name)
elif st.button("Predict Category", key="predict_button_2") and not resume_text:
    st.warning("Please enter or upload a resume to get a prediction.")