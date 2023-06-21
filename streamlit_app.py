

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import streamlit as st

OPENAI_API_KEY = st.text_input("API Key:")
pdf_files = st.file_uploader("Upload pdf files", type=["pdf"],
                               accept_multiple_files=False)

if pdf_files is not None:
  pdf_reader = PdfReader(pdf_files)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()

  # split into chunks
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
  chunks = text_splitter.split_text(text)
  filtered_chunks = [chunk for chunk in chunks if "indemnification" in chunk.lower() or "indemnify" in chunk.lower()]

  prompt_template = """  
  Context: {context}
  
  Question: {question}
  Answer:"""
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context","question"]
  )

  user_question = "Identify if the document one-sided or a mutual indemnification agreement? Provide examples of clauses to justify your answer."
  user_context = "Indemnity clauses may be structured as mutual indemnification, where both parties agree to indemnify each other for specific types of losses, or they may be one-sided, where only one party agrees to indemnify the other."
  # create embeddings
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  knowledge_base = FAISS.from_texts(filtered_chunks, embeddings)

  # show user input
  if user_question:
    docs = knowledge_base.similarity_search(user_question)
  
  st.write(docs)
  
  llm = OpenAI(openai_api_key=OPENAI_API_KEY)
  chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
  with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=user_question, context=user_context)
    st.write(response)
  
