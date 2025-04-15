from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional


import os
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
    allow_origins=["http://localhost:4200", "https://super-assistants.web.app", "https://trainer-teacher.web.app"],  # Adjust as needed for your app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_NAME = "totalpdf"

INDEX_NAME_2 = "ethic-teacher"

# Modelo para la solicitud de chat
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

# Modelo para la respuesta
class ChatResponse(BaseModel):
    response: str


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

 
# Inicialización de modelos y vector store
embeddings_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_KEY
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=400,
    api_key=OPENAI_KEY,
)

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API)
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings_model
)


# Inicializar Pinecone2
pc2 = Pinecone(api_key=PINECONE_API)
index_2 = pc2.Index(INDEX_NAME_2)
vector_store_2 = PineconeVectorStore(
    index=index_2,
    embedding=embeddings_model
)


# Modelo para la solicitud de chat con streaming
class ChatoRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    system_prompt_text: Optional[str] = None  # Campo opcional


async def stream_rag_response(query: str, session_id: str = "default_session", system_prompt_text: Optional[str] = None):
    """Genera respuestas en streaming desde el sistema RAG"""        
    
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
    # print(request)
    return StreamingResponse(
        stream_rag_response(request.message, request.session_id, request.system_prompt_text),
        media_type="text/plain"
    )
    
    
    








######## TEST para PARAMETROS ########






# Modelo para la solicitud de chat con streaming
class TeacherChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    system_prompt_text: Optional[str] = None  # Campo opcional
    pages: Optional[list] = None  # Campo opcional
    doc_path: Optional[str] = None  # Campo opcional
    



async def stream_rag_response_test(query: str, session_id: str = "default_session", system_prompt_text: Optional[str] = None, pages: Optional[list] = None, doc_path: Optional[str] = None):
    """Genera respuestas en streaming desde el sistema RAG""" 
    
    print(doc_path)
        
    retriever = vector_store_2.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20,  # Devolver los 3 documentos más relevantes
            "filter": {
                # "source": "pdfs/ethics-supervisors.pdf",  # Solo documentos de esta fuente
                "source": doc_path,  # Solo documentos de esta fuente
                # "title": "Ethics Management",
                # "page": {"$in": [2, 3, 4, 5, 6, 7, 8]}  # Buscar en las páginas 3, 4, 5, 6, 7, 8
                "page": {"$in": pages}  # Buscar en las páginas 8
            }
        }
    )
    
    # Probar la búsqueda
    # query = "ethical decisions at work"
    
    # docs = retriever.invoke(query)
    # for doc in docs:
    #     print(doc.page_content , '\n\n')
    
    # DEFAULT_SYSTEM_PROMPT = 'Eres un asistente que responde unicamente usando la informacion de los PDFs que tienes en las vectorstore'
    DEFAULT_SYSTEM_PROMPT = """
    You are an AI teacher for Ethics Management at work, answering questions and teaching based on vectorstore documents.

    **Content**:
    - Use only vectorstore documents, focusing on the specified section (e.g., "Defining Business Ethics") or the full course if no section is selected.

    **Special Inputs**:
    - "Just ask me 2 serious questions...": Ask two challenging questions from documents, give feedback, and explain answers if needed.
    - "Please, just ask me 1 easy question...": Ask one simple question from documents and provide feedback.
    - "In the docs you will find one starting with 'Section '...": Explain the section in ≤110 words, then ask if the user wants to continue or repeat.
    - "Can you explain 'Ethics Management for'...": Give a general course overview using all documents. Start with no more than 110 words. Then ask the user if they'd like to continue or hear the explanation again. Teach in the most helpful way possible.

    **Other Questions**:
    - Answer clearly using documents, asking for clarification if vague.
    - If a question is too broad or unclear, politely ask the user to rephrase or offer a list of specific topics they can choose from.
    - If the requested information isn't found in the documents, let the user know that your answers are limited to course content and suggest they contact Martin E Mendez for more details.

    **Tone**:
    - Professional, educational, conversational.

    **Constraints**:
    - Stay within ethics management; use only vectorstore content.

    **Date**: April 10, 2025.
    """

    system_prompt_text_to_system_prompt = system_prompt_text or DEFAULT_SYSTEM_PROMPT
    # Definir prompts
    system_prompt = (
       system_prompt_text_to_system_prompt
    )

    contextualize_q_system_prompt = (
        "Respond based on the chat history and the user's last question. "
        "If the information is not found in the chat history or context, you must not answer the question. "
        "Let the user know the information is unavailable and suggest asking Martin E Mendez about it. "
        "Also, always reply in a friendly — even humorous — tone. Feel free to use emojis."
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
@app.post("/stream_chat_test")
async def stream_chat_test(request: TeacherChatRequest):
    """Endpoint que devuelve la respuesta en streaming"""
    print(request)
    return StreamingResponse(
        stream_rag_response_test(request.message, request.session_id, request.system_prompt_text, request.pages, request.doc_path),
        media_type="text/plain"
    )
    
    
    
    
    
    
@app.get("/store")
async def getStore():
    """Endpoint que devuelve lo que tenga de store"""
    # print(store)
    return store


def generate_prompt_en(role, task, tone, format_out, restrictions=None, context=None):
    prompt = f"You are {role}. Your task is to {task}.\n"
    
    if context:
        prompt += f"Additional context:\n{context}\n\n"
    
    prompt += f"Respond in a {tone} tone and use the following output format: {format_out}.\n"
    
    if restrictions:
        prompt += f"Restrictions:\n{restrictions}\n"
    
    prompt += "\nIf you don't know the answer, say so honestly."
    return prompt


generate_prompt_en(
    role="an expert in business ethics",
    task="answer questions using only the information from the provided PDF documents",
    tone="friendly, even a bit humorous",
    format_out="bulleted lists with emojis at the start of each item",
    restrictions="Do not make up information. If it's not in the document, say you don't know.",
    context="The document is titled 'ethics-supervisors.pdf' and discusses ethical dilemmas in the workplace."
)    