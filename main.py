from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import PyPDF2
from langchain.document_loaders import UnstructuredURLLoader
import streamlit as st
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import os
from langchain.schema import Document  # Import the required Document class

def main():
    st.title("KLU Chat GPT")

    # Inputs
    text_file = st.file_uploader("Upload a Text file", type=["txt"])
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    '''web_address = st.text_input("Enter a web address (URL):", placeholder="https://example.com")'''
    user_query = st.text_area("Enter your query:")

    # Initialize variables
    text_content = ""
    all_documents = []

    # Process inputs


    if st.button("Process"):
        if text_file:
            try:
                text_content = text_file.read().decode("utf-8")
                all_documents.append(Document(page_content=text_content))
            except Exception as e:
                st.error(f"Failed to process the text file: {e}")

        if pdf_file:
            try:
                pdf_reader = PdfReader(pdf_file)
                pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
                all_documents.append(Document(page_content=pdf_text))
            except Exception as e:
                st.error(f"Failed to process the PDF file: {e}")

        '''if web_address:
            try:
                loader = UnstructuredURLLoader(urls=[web_address])
                documents = loader.load()
                all_documents.extend(documents)  # Ensure loader returns a valid list of Document objects
            except Exception as e:
                st.error(f"Failed to process the web address: {e}")

        # Check the contents of all_documents
        st.write("Documents to process:", all_documents)'''

        if not all_documents:
            st.warning("No valid input provided. Please upload a file or enter a URL.")
            return

        # Splitting Text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        try:
            split_texts = splitter.split_documents(all_documents)
            st.success("Text successfully split!")
        except Exception as e:
            st.error(f"Failed to split documents: {e}")
            return

        # Embedding and ChromaDB
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv('API_KEY'))
            persist_directory = "E:/Chroma_db"
            db = Chroma.from_documents(split_texts, embeddings, persist_directory=persist_directory)
        except Exception as e:
            st.error(f"Failed to create embeddings or ChromaDB: {e}")
            return

        # Define Prompt and LLM
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.

        {context}

        Question: {question}streamlit 
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        try:
            llm = ChatOpenAI(temperature=0.8, openai_api_key=os.getenv('API_KEY'))
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Get Answer
            if user_query:
                docs = db.similarity_search(user_query)
                context = "\n".join([doc.page_content for doc in docs])
                response = llm_chain.run({"context": context, "question": user_query})
                st.write("Your Answer is:", response)
            else:
                st.warning("Please enter a query.")
        except Exception as e:
            st.error(f"Failed to generate response: {e}")

if __name__ == "__main__":
    main()
# This is a sample Python script.

