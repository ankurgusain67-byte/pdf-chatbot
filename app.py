import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tempfile

load_dotenv()

# PAGE SETUP
st.title('PDF Question Answering Chatbot')
st.write('Upload a PDF and ask any question about it!')

# SESSION STATE SETUP
# Streamlit reruns the whole file on every interaction
# session_state saves data between those reruns
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

# PDF UPLOAD SECTION
uploaded_file = st.file_uploader('Upload your PDF here', type='pdf')

if uploaded_file is not None and st.session_state.vectordb is None:
    with st.spinner('Reading and processing your PDF...'):
        
        # Save uploaded file temporarily to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # STEP 1: LOAD
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # STEP 2: SPLIT
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)
        
        # STEP 3: EMBED + STORE
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        st.session_state.vectordb = Chroma.from_documents(chunks, embeddings)
        
    st.success('PDF processed! You can now ask questions.')

# SHOW CHAT HISTORY
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message('user'):
            st.write(msg.content)
    else:
        with st.chat_message('assistant'):
            st.write(msg.content)

# CHAT INPUT
if prompt := st.chat_input('Ask a question about your PDF...'):
    
    if st.session_state.vectordb is None:
        st.warning('Please upload a PDF first!')
    else:
        # Show user message
        with st.chat_message('user'):
            st.write(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Get relevant chunks
        relevant_docs = st.session_state.vectordb.similarity_search(prompt, k=3)
        context = '\n\n'.join([doc.page_content for doc in relevant_docs])
        
        # Ask LLM
        llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0)
        system_msg = SystemMessage(content='Answer only from the PDF context given. If the answer is not in the context, say I could not find that in the PDF.')
        full_question = f'Context:\n{context}\n\nQuestion: {prompt}'
        
        response = llm.invoke([system_msg] + [HumanMessage(content=full_question)])
        
        # Show AI reply
        with st.chat_message('assistant'):
            st.write(response.content)
        st.session_state.chat_history.append(AIMessage(content=response.content))