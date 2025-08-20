from fastapi import FastAPI, HTTPException
import httpx
import io
from PyPDF2 import PdfReader
from docx import Document # Para manejar .docx


from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import numpy as np


import os
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Esto es para Streaming (extra desde Claude)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
# from typing import AsyncGenerator

from dotenv import load_dotenv
# Cargar las variables del archivo .env
load_dotenv()

# For Redis DB
import redis
import json
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

# Conexión a Redis (Upstash)
REDIS_URL = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

KEY_PREFIX_CHAT = "chat_history:"


# # For Redis DB
# # import redis
# # import json
# from redis.asyncio import Redis
# import json

# REDIS_URL = os.getenv("REDIS_URL")

# # # Redis client
# # redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)




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
    allow_origins=["http://localhost:4200", "https://super-assistants.web.app", "https://trainer-teacher.web.app", "https://central-ats.web.app", "https://bridgetoworks.com"],  # Adjust as needed for your app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_NAME = "ethic-teacher"

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




store = {'perroe': "jamon", 'juancito': "Alejandro"}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def get_session_history(session_id: str) -> ChatMessageHistory:
    stored = redis_client.get(f"{KEY_PREFIX_CHAT}{session_id}")
    if stored:
        return ChatMessageHistory(messages=messages_from_dict(json.loads(stored)))
    return ChatMessageHistory()


def save_session_history(session_id: str, history: ChatMessageHistory):
    redis_client.set(f"{KEY_PREFIX_CHAT}{session_id}", json.dumps(messages_to_dict(history.messages)))
    

 
# Inicialización de modelos y vector store
embeddings_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_KEY
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=280,
    api_key=OPENAI_KEY,
)

llmResume = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=2000,
    api_key=OPENAI_KEY,
)

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API)
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(
    index=index,
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
    
    # DEFAULT_SYSTEM_PROMPT = 'Eres un asistente que responde unicamente usando la informacion de los PDFs que tienes en las vectorstore'
    DEFAULT_SYSTEM_PROMPT = 'You are an assistant that responds solely using the information from the PDFs you have in the vectorstore'

    system_prompt_text_to_system_prompt = system_prompt_text or DEFAULT_SYSTEM_PROMPT
    # Definir prompts
    system_prompt = (
       system_prompt_text_to_system_prompt
    )

    contextualize_q_system_prompt = (
        "Answer based on the chat history and the user's latest question"
        "If it's not in the chat history or context, you shouldn't answer the question. Just say that info isn't available and tell them to ask Martin E. Mendez about it"
        "Also, always reply in a friendly way. Even a bit funny"
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
    # get_session_history(session_id).add_message(HumanMessage(content=query))
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=query))
    
    # Transmitimos la respuesta por chunks
    response_text = ""
    # Aquí está la corrección - pasar un diccionario vacío como input
    async for chunk in streaming_chain.astream({}):
        response_text += chunk
        yield chunk
    
    # Guardamos la respuesta completa en el historial
    from langchain_core.messages import AIMessage
    # get_session_history(session_id).add_message(AIMessage(content=response_text))
    history.add_message(AIMessage(content=response_text))
    save_session_history(session_id, history)

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
        
    retriever = vector_store.as_retriever(
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
    You are an AI teacher answering questions and teaching based on vectorstore documents.

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
    - Use only vectorstore content.

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
    # get_session_history(session_id).add_message(HumanMessage(content=query))
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=query))
    
    # Transmitimos la respuesta por chunks
    response_text = ""
    # Aquí está la corrección - pasar un diccionario vacío como input
    async for chunk in streaming_chain.astream({}):
        response_text += chunk
        yield chunk
    
    # Guardamos la respuesta completa en el historial
    from langchain_core.messages import AIMessage
    # get_session_history(session_id).add_message(AIMessage(content=response_text))
    history.add_message(AIMessage(content=response_text))
    save_session_history(session_id, history)


# Endpoint para chatear con streaming
@app.post("/stream_chat_test")
async def stream_chat_test(request: TeacherChatRequest):
    """Endpoint que devuelve la respuesta en streaming"""
    # print(request)
    return StreamingResponse(
        stream_rag_response_test(request.message, request.session_id, request.system_prompt_text, request.pages, request.doc_path),
        media_type="text/plain"
    )
    
    
    
    
    
    
@app.get("/store")
async def getStore():
    """Endpoint que devuelve lo que tenga de store"""
    print(store)
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


## Redis test

@app.get("/redis/{session_id}")
async def get_session(session_id: str):
    redis_client.get(f"{KEY_PREFIX_CHAT}{session_id}")
    return {"status": "get", "session": session_id}

@app.delete("/redis/{session_id}")
async def delete_session(session_id: str):
    redis_client.delete(f"{KEY_PREFIX_CHAT}{session_id}")
    return {"status": "deleted", "session": session_id}

@app.get("/redis/sessions")
async def list_sessions():
    keys = redis_client.keys(f"{KEY_PREFIX_CHAT}*")
    return {"sessions": keys}


## RESUMEs and CVs

# Modelo para la solicitud de procesamiento de CV
class ProcessResumeRequest(BaseModel):
    resume_url: str
    user_id: str
    file_type: str # 

    
# Nuevo Endpoint modificado
@app.post("/process_resume_content")
async def process_resume_content(request: ProcessResumeRequest):
    """
    Recibe la URL de un CV, lo descarga, extrae el texto (basado en file_type),
    lo procesa con IA para estructurar la información y devuelve el JSON estructurado.
    """
    print(f"Received request to process resume for user {request.user_id} from URL: {request.resume_url}")
    print(f"File type received: {request.file_type}") # Para depuración
    
    try:
        # 1. Descargar el archivo del CV
        async with httpx.AsyncClient() as client:
            response = await client.get(request.resume_url)
            response.raise_for_status() # Lanza una excepción para errores HTTP (4xx o 5xx)
            
        file_content_stream = io.BytesIO(response.content)
        
        # 2. Extraer Texto del CV basado en el 'file_type'
        extracted_text = ""
        
        if request.file_type == 'application/pdf':
            try:
                reader = PdfReader(file_content_stream)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            except Exception as pdf_error:
                raise HTTPException(status_code=422, detail=f"Failed to read PDF content: {str(pdf_error)}")
        elif request.file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': # .docx
            try:
                doc = Document(file_content_stream)
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            except Exception as docx_error:
                raise HTTPException(status_code=422, detail=f"Failed to read DOCX content: {str(docx_error)}")
        elif request.file_type == 'application/msword': # .doc (más complejo, no se recomienda)
            raise HTTPException(
                status_code=400, 
                detail="Legacy .doc files are not directly supported. Please upload a PDF or .docx file."
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {request.file_type}. Only PDF and DOCX are supported."
            )

        if not extracted_text.strip():
            raise ValueError("Could not extract any meaningful text from the document.")

        print(f"Successfully extracted {len(extracted_text)} characters from the document.")
        
        # 3. Procesar el texto extraído con LangChain y OpenAI para estructurar la información
        # Define el prompt para extraer información
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in extracting key information from resumes.
            Extract the following details from the provided resume text:
            - **Name**
            - **Email**
            - **Phone Number** (standardized format if possible)
            - **Postal Code**
            - **City**
            - **Summary/Objective** (if present, max 6 sentences)
            - **Skills** (list main technical and soft skills, comma-separated)
            - **Work Experience** (list job titles, companies, dates, and brief descriptions)
            - **Certification** (list certificates, issuing organizations, years)
            - **Education** (list degrees, institutions, and graduation years)
            
            Format the output as a JSON object. If a field is not found, use `null`.
            Ensure the output is a valid JSON string, without any additional text or markdown outside the JSON object.
            """),
            ("user", "Here is the resume text:\n\n{resume_text}")
        ])

        # Crear la cadena de procesamiento
        # Asegúrate de que tu variable 'llmResume' (ChatOpenAI) esté inicializada globalmente en tu script.
        extraction_chain = extraction_prompt | llmResume | StrOutputParser()

        # Invocar la cadena para obtener la información estructurada
        # Ten en cuenta los límites de tokens del LLM para el tamaño del CV.
        raw_llm_response = await extraction_chain.ainvoke({"resume_text": extracted_text})
        
        # Intentar parsear la respuesta como JSON, manejando posibles envolturas de markdown
        parsed_resume_data = {}
        try:
            if raw_llm_response.strip().startswith("```json") and raw_llm_response.strip().endswith("```"):
                json_string = raw_llm_response.strip()[7:-3].strip()
            else:
                json_string = raw_llm_response.strip()
            
            parsed_resume_data = json.loads(json_string)
        except json.JSONDecodeError as json_err:
            print(f"LLM did not return valid JSON. Error: {json_err}. Raw response: {raw_llm_response}")
            raise HTTPException(
                status_code=500, 
                detail=f"AI failed to produce valid JSON from resume. Raw AI response: {raw_llm_response[:200]}..."
            )
        
        print("Parsed resume data (structured JSON):", parsed_resume_data)

        # 4. Devolver el JSON estructurado
        return JSONResponse(content=parsed_resume_data)

    except httpx.HTTPStatusError as e:
        # Manejo de errores durante la descarga del CV
        status_code = e.response.status_code if e.response else 500
        detail_msg = f"Error downloading resume from {request.resume_url}: HTTP {status_code} - {e.response.text if e.response else 'Unknown error'}"
        print(detail_msg)
        raise HTTPException(status_code=status_code, detail=detail_msg)
    except Exception as e:
        # Manejo de cualquier otro error inesperado
        print(f"An unexpected error occurred during resume processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    




# Modelo para la solicitud de chat con streaming
class ChatoRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    system_prompt_text: Optional[str] = None  # Campo opcional


async def stream_rag_response(query: str, session_id: str = "default_session", system_prompt_text: Optional[str] = None):
    """Genera respuestas en streaming desde el sistema RAG"""   

@app.post("/process_resume_content_test")
async def process_resume_content_test(request: ProcessResumeRequest):
    try:
        # Intentar parsear la respuesta como JSON, manejando posibles envolturas de markdown
        parsed_resume_data = {}
        
        parsed_resume_data = {
            "Name": "Martin E. Mendez",
            "Email": "info@mart-in.us",
            "Phone Number": "(347) 876-6500",
            "City": "New York",
            "Zipcode": "12345",
            "Summary/Objective": "Full-stack developer with 5+ years of experience in scalable web applications. Recently focused on integrating AI (LLMs, chatbots) to enhance user interaction.",
            "Skills": "Angular, Vue.js, TypeScript, Python, HTML5, CSS3, Sass, Tailwind CSS, Bootstrap, Node.js, Express, FastAPI, RESTful APIs, MongoDB, Cloud Firestore, Firebase, Pinecone, CI/CD pipelines, Git, Docker, Figma, Stripe API, QR code generation, Project management, UX/UI design, cross-functional collaboration, problem-solving",
            "Work Experience": [
                {
                "Job Title": "Software Developer",
                "Company": "Tupungato Wine Co.",
                "Dates": "May 2024 – Present",
                "Description": "Developed and launched a customer acquisition platform using Angular, TypeScript, and Tailwind CSS, ensuring brand positioning before the official launch. Built a secure RESTful API with Node.js and Express, integrating Stripe for payment processing during exclusive sales events, resulting in a 15% increase in transaction efficiency."
                },
                {
                "Job Title": "Web Developer – Project Manager",
                "Company": "AgroFull",
                "Dates": "March 2022 – August 2024",
                "Description": "Resolved project stagnation by creating high-fidelity Figma prototypes, providing clear UX/UI guidance that accelerated development by 30%. Enhanced the application’s frontend using Vue.js and Bootstrap, leveraging Git for version control to ensure seamless design integration and consistent UI across platforms."
                },
                {
                "Job Title": "Software Developer",
                "Company": "Victor Iermito Luthier",
                "Dates": "April 2021 – December 2021",
                "Description": "Developed a single-page application (SPA) with Angular and TypeScript to automate content management, replacing an outdated system and reducing manual updates by 40%. Implemented a self-service CRUD system for clients to manage guitar inventory and generate QR codes, cutting operational costs by 90% eliminating developer dependency."
                }
            ],
            "Education": [
                {
                "Degree": "Master of Music",
                "Institution": "Superior Conservatory of Paris",
                "Graduation Year": 2008
                },
                {
                "Degree": "Bachelor of Music",
                "Institution": "Conservatory of Colombes",
                "Graduation Year": 2006
                },
                {
                "Degree": "High School Diploma, Specialized in Exact Sciences",
                "Institution": "Buenos Aires, Argentina",
                "Graduation Year": 1998
                }
            ]
        }
        
        print("Parsed resume data (structured JSON):", parsed_resume_data)

        # 4. Devolver el JSON estructurado
        return JSONResponse(content=parsed_resume_data)

    except httpx.HTTPStatusError as e:
        # Manejo de errores durante la descarga del CV
        status_code = e.response.status_code if e.response else 500
        detail_msg = f"Error downloading resume from {request.resume_url}: HTTP {status_code} - {e.response.text if e.response else 'Unknown error'}"
        print(detail_msg)
        raise HTTPException(status_code=status_code, detail=detail_msg)
    except Exception as e:
        # Manejo de cualquier otro error inesperado
        print(f"An unexpected error occurred during resume processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    



# Modelo de solicitud simplificado
class EmbeddingScoreRequest(BaseModel):
    candidate_summary: str
    candidate_experience: str
    candidate_skills: str
    job_description: str

def calculate_cosine_similarity(vec1, vec2):
    """Calcula la similitud de coseno entre dos vectores."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def realistic_score(score: int) -> int:
    """
    Transforma un score de 0-1000 usando una función que:
    - Aumenta más los valores pequeños
    - Aumenta menos los valores grandes
    - Siempre devuelve un valor mayor o igual al de entrada
    - Nunca excede 1000
    - El 1000 devuelve 1000
    """
    if score >= 1000:
        return 1000
    
    if score <= 0:
        return score
    
    # Normalizamos al rango [0, 1]
    normalized_score = score / 1000.0
    
    # Usamos una función que garantiza crecimiento monotónico
    # La fórmula es: score + (1000 - score) * factor
    # donde factor va de ~0.5 para valores pequeños a ~0 para valores grandes
    
    remaining_space = 1000 - score  # Espacio disponible para crecer
    
    # Factor que decrece suavemente de 0.6 a 0
    # Usamos una función sigmoide invertida para suavidad
    growth_factor = 0.3 * (1 - normalized_score**2)
    
    enhanced_score = score + remaining_space * growth_factor
    
    return int(np.round(enhanced_score))


@app.post("/calculate_embedding_score")
async def calculate_embedding_score(request: EmbeddingScoreRequest):
    """
    Calcula un score de compatibilidad por similitud de embeddings.
    """
    try:
        # 1. Combinar el texto del candidato en una sola cadena
        # candidate_text = f"{request.candidate_summary} {request.candidate_experience}"
        candidate_text = f"{request.candidate_summary} {request.candidate_experience} {request.candidate_skills}"
        
        # 2. Generar embeddings para el candidato y la descripción del puesto
        candidate_embedding = embeddings_model.embed_documents([candidate_text])[0]
        job_embedding = embeddings_model.embed_documents([request.job_description])[0]
        
        # 3. Convertir a arrays de numpy para el cálculo
        candidate_vector = np.array(candidate_embedding)
        job_vector = np.array(job_embedding)
        
        # 4. Calcular la similitud de coseno y escalar a 100
        similarity_score = calculate_cosine_similarity(candidate_vector, job_vector)
        compatibility_score = int(similarity_score * 1000)
        
        # 5. Metemos acercador al 1000
        final_score = realistic_score(compatibility_score)
        
        print('text candidate \n\n')
        print(candidate_text + '\n\n')
        print('job description \n\n')
        print(request.job_description)
        
        return {"compatibility_score": final_score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular el score: {str(e)}")



# TEST para Score del Embedding
class ScoreTest(BaseModel):
    score: int
    
@app.post("/test_score_human_score")
async def calculate_embedding_score(request: ScoreTest):
    score = request.score
    final_score = realistic_score(score)
    return {"compatibility_score": final_score}
    
    
