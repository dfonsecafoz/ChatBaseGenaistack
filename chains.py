from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from typing import List, Any
from utils import BaseLogger, extract_title_and_question
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
import dotenv

dotenv.load_dotenv()

def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config.get("ollama_base_url"), model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    elif embedding_model_name == "google-genai-embedding-001":        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        dimension = 768
        logger.info("Embedding: Using Google Generative AI Embeddings")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using Sentence Transformer")

    return embeddings, dimension

def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "ollama":
        llm = ChatOllama(base_url=config.get("ollama_base_url"), model="llama2")
        logger.info("LLM: Using Ollama")
    elif llm_name == "openai":
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("LLM: Using OpenAI")
    elif llm_name == "aws":
        llm = BedrockChat()
        logger.info("LLM: Using AWS")
    else:
        raise ValueError(f"Unknown LLM name: {llm_name}")
    return llm
