import streamlit as st 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import SpiderLoader, PyPDFLoader, WebBaseLoader, DirectoryLoader, CSVLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from dotenv import load_dotenv
import os
from pymongo import MongoClient

#note: run using python -m streamlit run app.py instead 
#to resolve import of local package add path of folder to python.analysis.extraPaths cmd+, to search it in settings

load_dotenv()
FAISS_PATH = "faiss_data"

os.environ['OPENAI_API_KEY'] = st.secrets["API_KEY_JO"]

#setup mongodb 
connection_string = st.secrets["MONGO_AUTH"]
client = MongoClient(connection_string)
collection = client.caicamp.qa_eval

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

st.set_page_config(page_title="SMEGPT", page_icon="🤖")

st.title("SMEsGPT")

#streamlit conversation 
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

#get response
def get_response(query, chat_history, context):
    template = """
    You are a helpful customer assistance bot, helping SMEs who want to partner with 7-11 with their questions. 
    If the user question is in Thai, reply in Thai. If the user question is in English, reply in English.
    If no context is provided, say so and do not attempt to hallucinate a response. 
    Answer the following questions in detail using the following context and chat history:
    
    Context: {context}
    
    Chat history: {chat_history}
    
    User question: {user_question}
    """
    # prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    
    # print(template.format(context=context, chat_history=chat_history, user_question=query))  
    
    return llm.stream(template.format(context=context, chat_history=chat_history, user_question=query))

def stream_response(response):
    for chunk in response:
        yield chunk.content
        
#user input
user_query = st.chat_input("Your question")
if user_query is not None and user_query != "":
    # load_db = FAISS.load_local(FAISS_PATH, CohereEmbeddings(cohere_api_key=st.secrets["COHERE_API_KEY"]), allow_dangerous_deserialization=True)
    load_db = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    results = load_db.similarity_search_with_relevance_scores(user_query, k=3)
    context_list = []
    scores_list = []
    for context, scores in results:
        context_list.append(context)
        scores_list.append(scores)
    # st.write(context_list)
    # st.write(scores_list)
    context_pruned = []
    for idx, score in enumerate(scores_list):
        if score > 0.70:
            context_pruned.append(context_list[idx])
    # st.write(context_pruned)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_pruned])
    print(context_text)
    print(scores_list)
    # top_source = [doc.metadata["source"] for doc in context_pruned][0]
    
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.user_query = user_query
        
    with st.chat_message("AI"):
        ai_response = st.write_stream(stream_response(get_response(user_query, st.session_state.chat_history, context_text)))
        # st.write(f"source: {top_source}")
    st.session_state.chat_history.append(AIMessage(ai_response))

sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
# Display feedback buttons
selected = st.feedback("thumbs")
if selected is not None:
    if user_query is None and st.session_state.chat_history != []:
        collection.insert_one({"question": st.session_state.user_query, "answer": st.session_state.chat_history[-1].content, "rating": sentiment_mapping[selected]})
    
    

