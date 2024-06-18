import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from chains import load_embedding_model, load_llm

import os
import dotenv

dotenv.load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    st.header("📄 Chat with your PDF file")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Load embeddings using OpenAI
        embeddings, dimension = load_embedding_model(
            embedding_model_name="openai"
        )

        # Store chunks in Neo4j
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,
        )

        # Load LLM using OpenAI
        llm = load_llm(llm_name="gpt-4")

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()