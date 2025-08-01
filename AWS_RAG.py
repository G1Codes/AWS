import json
import os
import boto3
import streamlit as st


from dotenv import load_dotenv
load_dotenv()

#Embedding
from langchain_aws import BedrockEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.llms import Bedrock

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


from langchain_community.vectorstores import FAISS

#LLMs
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

#Bedrock Client

st.set_page_config(page_title="RAG Chatbot with AWS Bedrock", layout="wide")

bedrock = boto3.client("bedrock-runtime")

# Embeddings
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",region_name="us-east-1")
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


embedding_options = ["Bedrock Embeddings", "Google Embeddings"]

selected_embedding = st.selectbox("Please select the embedding model", embedding_options)


if selected_embedding == "Bedrock Embeddings":
    embeddings = bedrock_embeddings
else:
    embeddings = google_embeddings

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("AWS_App/data")
    documents = loader.load()

    #split the data
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 400)
    splitted_docs = text_splitter.split_documents(documents)
    return splitted_docs

# Vector embeddings & Vector store

def get_vector_store(splitted_docs):
    faiss_vectorestore = FAISS.from_documents(splitted_docs, embeddings)
    #Save Locally
    faiss_vectorestore.save_local("AWS_App/faiss_index")
    return faiss_vectorestore

#LLMs

def mistral():
    llm = Bedrock(model_id= "mistral.mistral-7b-instruct-v0:2", 
                  client = bedrock,
                  model_kwargs= {"max_tokens": 250})
    return llm

def llama():
    llm= Bedrock(model_id= "meta.llama3-8b-instruct-v1:0", 
                  client = bedrock,
                  model_kwargs= {"max_gen_len": 250})
    return llm

# Prompt Template
prompt_template = """
Human: Based on the following context, provide a concise summary that directly answers the question.
Your summary should be approximately 250 words and include key details from the context.
If the answer cannot be found in the provided context, state that you do not know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""
prompt = PromptTemplate(template=prompt_template,
                        input_variables= ["context", "question"])

#Response
def get_response(llm, retriever, query):
    # Wrap retriever in a RunnableLambda so that it can become part of a chain
    retriever_runnable = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))

    # Create the chain using the correct format
    chain = create_retrieval_chain(retriever_runnable, prompt | llm | StrOutputParser())

    # Now invoke the chain
    response = chain.invoke({"question": query})
    # return response["answer"]  # assuming your final output has a key 'answer'
    return response['answer']

# Create Streamlit App

def main():

    st.title("RAG Chatbot with AWS Bedrock")

    st.write("Ask a question about your documents:")
    user_question = st.text_area("Your Question:", height=70)

    if st.button("Vectorize"):
        with st.spinner("Creating Vector Database..."):
            splitted_docs = data_ingestion()
            get_vector_store(splitted_docs)
            st.success("Vector Database Created")
    
    
    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("AWS_App/faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()
            llm = mistral()
            st.success(get_response(llm, faiss_index, user_question)) 
        
    if st.button ("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("AWS_App/faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()
            llm = llama()
            st.success(get_response(llm,faiss_index,user_question))

if __name__ == "__main__":
    main()            