import openai
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter  # Importe esta linha
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import tempfile
import urllib.request
import requests


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou seu assistente."]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("question:", placeholder="Me pergunte sobre o(s) conjunto(s) de dados pré-carregados", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                st.write(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                st.write(st.session_state["generated"][i], key=str(i))

def create_conversational_chain(vector_store):

    openai_api_key = os.getenv("OPENAI_API_KEY")  # Acessando a variável de ambiente OPENAI_API_KEY
    llm = OpenAI(
        model="gpt-3.5-turbo",
        language="pt-BR",
        callbacks=[StreamingStdOutCallbackHandler()],
        api_key=openai_api_key,
        input={"max_tokens": 1500}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain


def main():

    # Initialize session state
    initialize_session_state()
    st.title('[Versão 3.0] GPT Chatbot.')
    # URL direta para a imagem hospedada no GitHub
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/main/fluxo%20atual%201-%20GPT.jpg'
    # Exibir a imagem usando a URL direta
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Modelo GPT (gpt-3.5-turbo) com refinamento de parâmetros;  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) Processamento dos dados carregados (em B.) com uso da biblioteca Langchain.')
    # Carrega o arquivo diretamente (substitua o caminho do arquivo conforme necessário)

    # Carrega o primeiro arquivo diretamente
    file_path1 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(urllib.request.urlopen(file_path1).read())
        temp_file_path1 = temp_file1.name

    text1 = []
    loader1 = TextLoader(temp_file_path1)
    text1.extend(loader1.load())
    os.remove(temp_file_path1)
    
    # Carrega o segundo arquivo diretamente
    file_path2 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(urllib.request.urlopen(file_path2).read())
        temp_file_path2 = temp_file2.name

    text2 = []
    loader2 = TextLoader(temp_file_path2)
    text2.extend(loader2.load())
    os.remove(temp_file_path2)
    
    # Combina os textos carregados dos dois arquivos
    text = text1 + text2

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
