# to run this: streamlit run <filename>
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
import joblib

from dotenv import load_dotenv

import os

import json

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI

load_dotenv()

usage_model = joblib.load('usage_model_class.joblib')
data_type_model = joblib.load('data_type_model_class.joblib')

embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

st.title("Is there hidden devil lurking in the details of your contract?")


def make_db_from_input(input_text):
    import datetime

    current_datetime = datetime.datetime.now()
    current_datetime_string = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    tmp_path = f'{current_datetime_string}.txt'
    with open(tmp_path, 'w') as f:
        f.write(input_text)

    raw_documents = TextLoader(tmp_path).load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="<br>"
    )
    documents = text_splitter.split_documents(raw_documents)

    db = FAISS.from_documents(documents, embeddings)
    return db


def run_check():
    text = st.session_state["contract_text"]
    db = make_db_from_input(text)

    tab_models = [(tab2, data_type_model), (tab3, usage_model)]
    for tab, model in tab_models:
        with tab:
            clusters, tokens = model.predict(db)[:2]
            c1, c2 = st.columns(2)
            with c1:
                st.header('Usual elements')
            with c2:
                st.header('Unusual elements')

            for cluster, token in zip(clusters, tokens):
                if cluster != -1:
                    with c1:
                        st.text(token)
                else:
                    with c2:
                        st.text(token)


tab1, tab2, tab3 = st.tabs(["Contract", "Data Collected", "Data Usage"])

with tab1, st.form("config_form"):
    st.text("Make sure each pargraphs are separated by <br><br>")
    st.text_area("Enter contract here:", key="contract_text")
    st.form_submit_button(label="Check!", on_click=run_check)