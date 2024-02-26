import openai
import streamlit as st
import pandas as pd
import urllib.request

# Configurações do ChatBot
st.title("Este é o ChatBot desenvolvido por Pedro Sampaio Amorim. Inclua um texto para debater com o bot!")

openai.api_key = st.secrets['OPENAI_API_KEY']

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Carrega o texto diretamente de um link
file_path1 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/PSA"
conteudo = urllib.request.urlopen(file_path1).read().decode('utf-8')

# Recebe a entrada do usuário do arquivo enviado (Tipo 1)
prompt_tipo_1 = conteudo
st.session_state.messages.append({"role": "user", "content": prompt_tipo_1, "tipo": "tipo_1"})

# Exibe o histórico de mensagens
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

# Recebe a entrada do usuário (Tipo 2)
if prompt_tipo_2 := st.text_input("Enviou o texto? Se sim, o que você gostaria de discutir sobre ele? Caso não queira falar sobre texto, do que deseja falar?"):
    st.session_state.messages.append({"role": "user", "content": prompt_tipo_2, "tipo": "tipo_2"})

    # Gera a resposta do ChatBot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Cria uma lista de mensagens para enviar ao modelo de geração de texto
        chat_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        # Envia as mensagens para o modelo e recebe uma resposta
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=chat_messages,
            stream=True,
        ):
            # Atualiza a resposta completa com a parte nova da resposta
            full_response += response.choices[0].delta.get("content", "")
            # Exibe a resposta parcial
            message_placeholder.markdown(full_response + "▌")
        # Exibe a resposta completa
        message_placeholder.markdown(full_response)
    # Registra a resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})
