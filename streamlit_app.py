

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

  prompt_template = """

  One sided indemnification agreements are where the Supplier shall bear full responsibility for indemnifying the Customer against any losses or damages caused by the Supplier's unauthorized use, disclosure, or misappropriation of the Customer's confidential information. 
  Two sided indemnification agreements are where the Customer and Supplier mutually agree to indemnify and defend each other against any claims, damages, liabilities, losses, costs, and expenses arising out of third-party intellectual property infringement claims related to their respective products, services, or deliverables provided under this Agreement.
  
  
  {context}
  
  Question: {question}
  Answer:"""
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context","question"]
  )

  user_question = "Using the following information, identify if the document is a one sided or two sided indemnification agreement. Provide examples of clauses from the below to justify your answer."

  # create embeddings
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
  knowledge_base = FAISS.from_texts(chunks, embeddings)

   st.write(knowledge_base)

  # show user input
  if user_question:
    docs = knowledge_base.similarity_search(user_question)

  llm = OpenAI(openai_api_key=OPENAI_API_KEY)
  chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
  with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=user_question)
    st.write(response)
  
