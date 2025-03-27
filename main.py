# from variables import OPENAI_KEY, PINECONE_API
from fastapi import FastAPI, HTTPException
# from typing import Optional

from fastapi.middleware.cors import CORSMiddleware
# This import for async chat
from fastapi.responses import StreamingResponse
# import asyncio
from pydantic import BaseModel
from typing import Optional
# from openai import OpenAI

# import os

# This is for Assistant
# import openai


import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
from typing import Dict, Any
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Esto es para Streaming (extra desde Claude)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from typing import AsyncGenerator



from dotenv import load_dotenv
# Cargar las variables del archivo .env
load_dotenv()


# # # Get the OPENAI_KEY from environment in Render
# # OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# client = OpenAI(api_key=OPENAI_KEY)

OPENAI_KEY = os.getenv("OPENAI_KEY")
PINECONE_API = os.getenv("PINECONE_API")

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://super-assistants.web.app/"],  # Adjust as needed for your app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Configuración de variables de entorno
# OPENAI_KEY = os.getenv("OPENAI_KEY")
# PINECONE_API = os.getenv("PINECONE_API")

INDEX_NAME = "totalpdf"

# Modelo para la solicitud de chat
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

# Modelo para la respuesta
class ChatResponse(BaseModel):
    response: str


# # Crear prompt para QA
# qa_prompt = ChatPromptTemplate.from_messages([
#     ('system', system_prompt),
#     ('system', 'Contexto: {context}'),
#     MessagesPlaceholder('chat_history'),
#     ('human', '{input}')
# ])

# # Crear cadena de respuesta
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# # Crear cadena RAG completa
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Configurar historial de conversación



# Endpoint para evitar la caca del favicon
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return {}

# Endpoint de verificación de salud
@app.get("/health")
async def health_check():
    return {"status": "healthy"}




store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# # Crear cadena RAG conversacional
# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key='input',
#     history_messages_key='chat_history',
#     output_messages_key='answer'
# )


# Modelo para la solicitud de chat con streaming
class ChatoRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    system_prompt_text: Optional[str] = None  # Campo opcional


async def stream_rag_response(query: str, session_id: str = "default_session", system_prompt_text: Optional[str] = None):
    """Genera respuestas en streaming desde el sistema RAG"""        
        
    # Inicialización de modelos y vector store
    embeddings_model = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=OPENAI_KEY
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.6,
        max_tokens=1250,
        api_key=OPENAI_KEY,
    )

    # Inicializar Pinecone
    pc = Pinecone(api_key=PINECONE_API)
    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings_model
    )
    retriever = vector_store.as_retriever()
    
    DEFAULT_SYSTEM_PROMPT = 'Eres un asistente que responde unicamente usando la informacion de los PDFs que tienes en las vectorstore'

    system_prompt_text_to_system_prompt = system_prompt_text or DEFAULT_SYSTEM_PROMPT
    # Definir prompts
    system_prompt = (
       system_prompt_text_to_system_prompt
    )

    contextualize_q_system_prompt = (
        "Responde segun el historial del chat y la última pregunta del usuario "
        "Si no está en el historial del chat o en el contexto no debes responder a la pregunta. Debes indicar que esa información no esta disponible y que preguntes a Martin E Mendez al respecto."
        "Ademas siempre responde de manera simpatica. Hasta graciosa. Puedes usar emojis."
    )

    # Crear prompt para contextualizar la pregunta
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    # Crear retriever consciente del historial
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    
    # Preparar el retriever con awareness de historial
    documents = await history_aware_retriever.ainvoke({
        "chat_history": get_session_history(session_id).messages,
        "input": query
    })
    
    # Preparamos el contexto con los documentos recuperados
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Creamos un prompt específico para esta consulta
    streaming_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('system', f'Contexto: {context}'),
        *get_session_history(session_id).messages,
        ('human', query)
    ])
    
    # Creamos una nueva cadena de streaming
    streaming_chain = streaming_prompt | llm | StrOutputParser()
    
    # Guardamos el mensaje en el historial para futuras consultas
    get_session_history(session_id).add_message(HumanMessage(content=query))
    
    # Transmitimos la respuesta por chunks
    response_text = ""
    # Aquí está la corrección - pasar un diccionario vacío como input
    async for chunk in streaming_chain.astream({}):
        response_text += chunk
        yield chunk
    
    # Guardamos la respuesta completa en el historial
    from langchain_core.messages import AIMessage
    get_session_history(session_id).add_message(AIMessage(content=response_text))


# Endpoint para chatear con streaming
@app.post("/stream_chat")
async def stream_chat(request: ChatoRequest):
    """Endpoint que devuelve la respuesta en streaming"""
    print(request)
    return StreamingResponse(
        stream_rag_response(request.message, request.session_id, request.system_prompt_text),
        media_type="text/plain"
    )