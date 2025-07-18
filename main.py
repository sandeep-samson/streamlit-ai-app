# Import necessary libraries
import ollama
import tempfile
import streamlit as st
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# Set the page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model selection
if "model" not in st.session_state:
    st.session_state['model'] = None

# Fetch available models from Ollama
models = [model[1]['model'] for model in enumerate(ollama.list()["models"])]
st.session_state['model'] = st.selectbox("Chose your model", models)

# Initialize the Ollama LLM
llm = OllamaLLM(model=st.session_state["model"])

# Define the prompt template
template = """ You are a helpful assistant. Answer the following question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Create a chain
chain = prompt | llm

# Streamlit app layout
st.title(":robot: AI Assistant")
st.sidebar.header("Ask me anything or upload a document to get answers!")
question = st.sidebar.text_input("Enter your question:", "")

# Document Loader
st.sidebar.header("Document Loader")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])

# If a file is uploaded, process it else answer the question directly
if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load the document
    loader = TextLoader(tmp_file_path, encoding="utf-8")
    documents = loader.load()
    st.sidebar.success("Document loaded successfully!")
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
        )
    
    if st.sidebar.button("Ask Document"):
        if question:
            with st.spinner("Searching document..."):
                result = qa.invoke({"query": question})
                st.success("Answer found!")
                st.write(result['result'])
        else:
            st.error("Please enter a question to ask the document.")
else:
    input_ = {"question": question}
    if st.sidebar.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                response = chain.invoke(input_)
                st.success("Answer generated successfully!")
                st.write(response)
        else:
            st.error("Please enter a question to get an answer.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Powered by [Ollama](https://ollama.com) and [LangChain](https://langchain.com).")
st.sidebar.write("Model: " + st.session_state['model'])
st.sidebar.write("Version: 1.0.0")
