import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import PyPDF2
import docx
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_txt(file):
    return file.read().decode('utf-8')

def read_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def read_pdf(file):
    reader= PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def read_csv(file):
    return pd.read_csv(file)

def read_excel(file):
    return pd.read_excel(file)

def read_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)


def get_response(prompt):
    model= genai.GenerativeModel('gemini-2.0-flash')
    response=model.generate_content(prompt,generation_config={"temperature": 0.7})
    return response.text

def generate_chart(data,col1,col2=None, chart_type='bar'):
    fig,ax= plt.subplots()
    if chart_type == 'bar':
        data[col1].value_counts().plot(kind='bar', ax=ax)
    elif chart_type == 'line' and col2:
        data.plot(x=col1, y=col2, kind='line', ax=ax)
    elif chart_type == 'scatter' and col2:
        data.plot(x=col1, y=col2, kind='scatter', ax=ax)
    
    
    return fig

st.set_page_config(page_title="Data Analyst Assistant")
st.title("Data Analyst Assistant")

file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf", "csv", "xlsx", "png", "jpg"])
if file is not None:
    filetype= file.name.lower()
    data=None
    if filetype.endswith('.txt'):
        content = read_txt(file)
    elif filetype.endswith('.docx'):
        content = read_docx(file)
    elif filetype.endswith('.pdf'):
        content = read_pdf(file)
    elif filetype.endswith('.csv'):
        data = read_csv(file)
        st.write(data.head())
        content = data.head().to_string()
    elif filetype.endswith('.xlsx'):
        data = read_excel(file)
        st.write(data.head())
        content = data.head().to_string()
    elif filetype.endswith(('.png', '.jpg', '.jpeg')):
        content = read_image(file)
    else:
        st.error("Unsupported file type")
        st.stop()
    
    question= st.text_input("Ask a question about the data:")
    if question:
        prompt =f"""
        You are a data analyst agent powered by Llama-4 Maverick. 
        Your task is to analyze the provided document and answer questions based on its content. 
        You can also generate visualizations based on the data in the document.
        Please provide a detailed and accurate answer to the question based on the document content. 
        If the question requires data analysis or visualization, generate the necessary code and describe the results clearly.
        Document Content:\n{content}\n
        \nQuestion: {question}"""
        with st.spinner("Generating response..."):
            response = get_response(prompt)
        st.success("Answer:")
        st.write(response)
    
    if data is not None:
        st.markdown("Data Visualization Options:")
        st.subheader("Generate Visualization")
        col=data.columns.tolist()
        col1 = st.selectbox("Select X-axis",col)
        col2 = st.selectbox("Select Y-axis (optional)", [None] + col)
        chart_type = st.selectbox("Select Chart Type", ['bar', 'line', 'scatter'])
        if st.button("Generate Chart"):
            fig= generate_chart(data, col1, col2, chart_type)
            st.pyplot(fig)
    


