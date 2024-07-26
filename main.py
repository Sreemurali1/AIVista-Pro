# Import necessary libraries
import io
import os
import logging

from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from typing import Optional
from fastapi.responses import JSONResponse
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Initialize FastAPI app
app = FastAPI()

#Mount static files and templates
app.mount("/static", StaticFiles(directory="template/static"), name="static")
templates = Jinja2Templates(directory="template")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Replace with your frontend's domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create generative models
model = genai.GenerativeModel("gemini-1.5-flash")
model1 = genai.GenerativeModel("gemini-pro")


# get_response function
async def get_response(question: str, image_bytes: bytes):
    try:


        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
        else:
            img = None

        if question and img:
            response = model.generate_content([question, img])
        elif not img:
            response = model1.generate_content(question)
        elif not question:
            response = model.generate_content(img)
        else:
            raise HTTPException(status_code=400, detail="Invalid input")


    except Exception as e:
         print(f"Error generating response: {e}")
         response = "I'm having trouble generating a response right now. Please try again later."


    return response.text


# Function to embedd the pdf into faiss index
def get_pdf_text(pdf):
    text = ""
    with io.BytesIO(pdf) as pdf_buffer:
        pdf_reader = PdfReader(pdf_buffer)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Context:
    {context} 
    
    Question: {question}

    **Instructions:** 

    * consider you as an expert in reading the pdf files and understand what is context.
    * Carefully consider the provided context.
    * If the answer is directly found in the context, provide the answer and cite the relevant section.
    * If the answer requires external knowledge, access the knowledge base.
    * If the question is unclear, ask for clarification.
    * Identify the sections of the context that are most relevant to answering the question.
    * Provide a detailed answer based on the relevant context.
    * loop all over the text and give the perfect answer. 
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Route to render index.html
@app.get("/")
async def ask_gemini(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

   
#Generate content based on input like text, image or pdf   
@app.post("/process_file_and_get_answer/")
async def process_file_and_get_answer(question: Optional[str] = Form(None), image: UploadFile = File(None), file: UploadFile = File(None)):
    try:
        response_text = ""
        if file:
            pdf_doc = await file.read()
            raw_text = get_pdf_text(pdf_doc)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
            docs = new_db.similarity_search(question, k=10)  # Retrieve top 3 relevant chunks

            chain = get_conversational_chain()
            response_text = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)

            return {"message": "File processed and question answered successfully", "response_text": response_text["output_text"]}
        elif question and image:
            image_bytes = await image.read() if image else None
            response_text = await get_response(question, image_bytes)
            return JSONResponse(content={"response_text": response_text})
        elif image:
            image_bytes = await image.read()
            response_text = await get_response(question,image_bytes)
            return JSONResponse(content={"response_text": response_text})
        
        elif question:
            #image_bytes = await image.read()
            response_text = await get_response(question,image)
            return JSONResponse(content={"response_text": response_text})
        else:
            raise HTTPException(status_code=400, detail="Either question or file must be provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












    




