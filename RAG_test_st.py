#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
import os


# In[2]:


BasePath = "./faiss_db_test"


# In[3]:


from langchain_community.embeddings import OllamaEmbeddings


# In[4]:


embeddings = (
    OllamaEmbeddings(model='all-minilm')
)


# In[5]:


import ollama
from langchain_community.chat_models import ChatOllama


# In[6]:


llm = ChatOllama(model="tinyllama", temperature=0) #, format="json"


# In[7]:


retriever = FAISS.load_local(BasePath, embeddings,allow_dangerous_deserialization =True).as_retriever()


# In[8]:


from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# In[10]:


def answer_question_using_rag(question):
    
    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "keep the answer concise. "
    "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    response = chain.invoke({"input": question})

    final_ans = response['answer']#[10:-2]
    
    
    return final_ans


# In[11]:


st.title("RAG on Python PDF Book")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
# React to user input
if prompt := st.chat_input("Please ask your questions regarding Python language"):
        # display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        result = answer_question_using_rag(prompt)
        response = f"Bot: {result}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            
        # add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



# In[ ]:




