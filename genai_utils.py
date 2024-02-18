import subprocess
import tiktoken
import pandas as pd
import os
import csv
import json
import time
import re
import transformers
import torch
import numpy as np
from datetime import datetime

#We will use langchanin to create a vector store to retrieve stronger negatives
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


import openai

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')


#let's split the document into chunks chunk size 128 and overlap size 64
def create_and_get_retriever(docs, emb_model_name, chunk_size = 128, chunk_overlap = 64, top_k=8, add_start_index = True, 
                             separators = ["\n\n\n","\n\n", "\n", " ", ""], context_header = [], faiss_dir = "./data/faiss"):
    print(f"Original # of docs {len(docs)}")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, chunk_overlap  = chunk_overlap, add_start_index = add_start_index,
                                                                         separators=separators)
    docs_split = text_splitter.split_documents(docs)
    print(f"Split # of docs {len(docs_split)}")

    embedding_function = HuggingFaceEmbeddings(
            model_name=emb_model_name,
            cache_folder="./models/sentencetransformers"
        )
    if len(context_header) > 0: #if adding metadata to the page content to aide retrieval
        for doc in docs_split:
            header_str = ""
            for header in context_header:
                if header in doc.metadata.keys():
                    header_str += str(header)+ ": " + doc.metadata[header] + "; " 
            doc.page_content = header_str +"\n"+ doc.page_content
            doc.metadata = {"context_header": context_header}

    db = FAISS.from_documents(docs_split, embedding_function)
    db.save_local(faiss_dir)
    return VectorStoreRetriever(vectorstore=db, search_kwargs={"k": top_k})


def get_retriever(emb_model_name, top_k=8, faiss_dir = "./data/faiss"):
    embedding_function = HuggingFaceEmbeddings(model_name=emb_model_name,cache_folder="./models/sentencetransformers")
    db = FAISS.load_local(faiss_dir, embedding_function)
    return VectorStoreRetriever(vectorstore=db, search_kwargs={"k": top_k})


def generate_response(prompt, model, system_prompt="", temperature=0):
    """
    Generate a response to a prompt using the given model.
    """
    try:
        response_full = openai.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt }],temperature=temperature)
    except Exception as e:
        st.warning("OpenAI API call failed. Waiting 5 seconds and trying again.")
        time.sleep(5)
        response_full = openai.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt }],temperature=temperature)
    except:
        return "Exception: OpenAI API call failed. Please check code or try again later."
    response = response_full.choices[0].message.content
    return response


def generate_kb_response(prompt, model, retriever, system_prompt="",template=None, temperature=0, include_source=False):
    """
    Generate a response to a prompt using the given model and the knowledge base retriever.
    Args:
    prompt: The prompt to generate a response to.
    model: OpenAI model to use to generate the response (can change to other models if needed)
    retriever: The knowledge base retriever to use to retrieve relevant documents.
    system_prompt: The system prompt to use for the OpenAI model.
    template: The template to use for the prompt. If None, a default template will be used. Please use {prompt} and {context} as placeholders for the prompt and context.
    temperature: The temperature to use for the OpenAI model.
    include_source: Whether to include the source documents metadata as part of context for LLM. Useful if you want the LLM to include the source documents in the response.
    """
    relevant_docs = retriever.get_relevant_documents(prompt)

    relevant_docs_str = ""
    docs_with_source = ""
    for doc in relevant_docs:
        if include_source:
            docs_with_source += doc.page_content + "\n" + "Source: " + str(doc.metadata) + "\n\n"
        else:
            relevant_docs_str += doc.page_content + "\n\n"
            docs_with_source += doc.page_content + "\n" + "Source: " + str(doc.metadata) + "\n\n"
    if include_source:
        relevant_docs_str = docs_with_source

    if template is None:
        prompt_full = f"""Answer based on the following context

        {relevant_docs_str}

        Question: {prompt}"""
    else:
        prompt_full = template.format(prompt=prompt, context=relevant_docs_str)

    try:
        response_full = openai.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt_full }],temperature=temperature)
    except Exception as e:
        print("OpenAI API call failed. Waiting 5 seconds and trying again.")
        time.sleep(5)
        response_full = openai.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt_full }],temperature=temperature)
    except Exception as e:
        return {'answer':"Exception: OpenAI API call failed. Please check code or try again later.", 'source_documents':docs_with_source} 
    
    #response = response_full['choices'][0]['message']['content']
    response = response_full.choices[0].message.content
    
    return {'answer':response, 'source_documents':docs_with_source}