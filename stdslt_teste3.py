import streamlit as st
import urllib.request
import os
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader


# Configurações do ChatBot
st.title("Este é o ChatBot desenvolvido por Pedro Sampaio Amorim. Inclua um texto para debater com o bot!")

# Carrega o texto diretamente de um link
file_path1 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt"
with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
    temp_file1.write(urllib.request.urlopen(file_path1).read())
    temp_file_path1 = temp_file1.name

text1 = []
loader1 = TextLoader(temp_file_path1)
text1.extend(loader1.load())
os.remove(temp_file_path1)

# Exibe o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        if message.get("tipo") == "tipo_1":
            # Mensagens do Tipo 1 (contexto) não são exibidas na tela
            continue
        with st.chat_message("user_tipo_2"):
            st.markdown(f"**Usuário:** {message['content']}")

# Dividir o texto em chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=950, length_function=len)
text_chunks = text_splitter.split_documents(text1)

# Criar embeddings para os chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Criar vector store com os embeddings
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# Criar a cadeia conversacional
chain = ConversationalRetrievalChain(vector_store)

# Recebe a entrada do usuário (Tipo 2)
prompt_tipo_2 = st.text_input("Enviou o texto? Se sim, o que você gostaria de discutir sobre ele? Caso não queira falar sobre texto, do que deseja falar?")

if prompt_tipo_2:
    st.session_state.messages.append({"role": "user", "content": prompt_tipo_2, "tipo": "tipo_2"})

    # Gera a resposta do ChatBot
    with st.chat_message("assistant"):
        response = chain.get_response(prompt_tipo_2)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
