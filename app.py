import streamlit as st
import pandas as pd
import ollama 

st.set_page_config(page_title="Chatbot Titanic")
st.title("🚢 Chatbot: Análise do Titanic")
st.markdown("Faça perguntas sobre os dados do arquivo titanic.parquet ")

DATA_FILE = "titanic.parquet" 

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        return df
    except FileNotFoundError:
        st.error(f"🚨 ERRO: Arquivo de dados não encontrado: {file_path}")
        st.stop()

df_titanic = load_data(DATA_FILE)

if df_titanic is None:
    st.stop()

DATA_SUMMARY = f"""
Dados do Titanic:
- Total de linhas: {len(df_titanic)}
- Colunas e tipos:
{df_titanic.dtypes.to_frame('Tipo').to_markdown()}
- Resumo Estatístico (descrição):
{df_titanic.describe(include='all').to_markdown()}
"""

SYSTEM_PROMPT = f"""
Você é um assistente de IA especialista em responder perguntas sobre os dados do Titanic.
Sua tarefa é analisar o contexto estatístico fornecido e responder à pergunta do usuário.
Use apenas as informações estatísticas e de estrutura de dados abaixo para formular sua resposta.
Responda sempre em português.

CONTEXTO DE DADOS:
{DATA_SUMMARY}
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pergunte sobre os dados do Titanic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando e analisando o resumo dos dados..."):
            try:
                response = ollama.chat(
                    model='llama3',
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                assistant_response = response['message']['content']
                st.markdown(assistant_response)
            except Exception as e:
                assistant_response = (
                    f"❌ Erro ao conectar com o LLama/Ollama: {e}. "
                    f"Verifique se o Ollama está em execução (`ollama serve`) e se o modelo 'llama3' está instalado."
                )
                st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})