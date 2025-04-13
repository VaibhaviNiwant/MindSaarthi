from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_groq import ChatGroq
from flask_cors import CORS
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)

# Initialize LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with your actual API key
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Setup Vector Database
def create_vector_db():
    loader = DirectoryLoader("./", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

# Check if VectorDB exists
db_path = "./chroma_db"
llm = initialize_llm()

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Setup QA Chain without keyword check
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    
    # Use a detailed prompt
    prompt_templates = """You are a compassionate and professional mental health chatbot. Respond to the following question thoughtfully, based on the context provided:

    {context}
    User: {question}
    Chatbot:"""
    
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = setup_qa_chain(vector_db, llm)

@app.route("/chat", methods=["POST", "GET"])
def chatbot_response():
    if request.method == "GET":
        return jsonify({"message": "Use POST method to communicate with the chatbot."})

    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"role": "assistant", "content": "Please provide a valid input."})

    # Get the assistant's response directly from the vector database and LLM
    assistant_response = qa_chain.run(user_input)
    return jsonify({"role": "assistant", "content": assistant_response})

# Home Route
@app.route("/")
def home():
    return "Mental Health Chatbot API is running!"

if __name__ == "__main__":
    app.run(debug=True)
