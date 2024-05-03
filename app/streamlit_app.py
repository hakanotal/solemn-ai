from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv('.env', override=True)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


@st.cache_resource
def initialization():
    st.write(f"Initialization started.")

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a VectorStore
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    st.write("Knowledge Graph created.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

    # Prompt template to query Gemini
    llm_prompt_template = """You are a legal-assistant for question-answering tasks. 
    Use the following context to form your answer for the question. 
    Use five sentences minimum and keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    # Combine data from documents to readable string format.
    def format_docs(docs):
        # print(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    # Create stuff documents chain using LCEL.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    
    st.write("RAG Chain created.")

    return rag_chain


rag_chain = initialization()

def main():
    st.title("Solemn AI")
    st.subheader("EU AI ACT - RAG Chain")

    # Get user input
    user_query = st.text_input("Ask a question:")

    if user_query:
        # Process user query and get the answer
        answer = rag_chain.invoke(user_query)

        # Display the answer
        st.write(f"Answer:\n> {answer}")

if __name__ == "__main__":
    main()