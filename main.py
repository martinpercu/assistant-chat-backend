from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Configurar LangChain con OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_KEY"))


app = FastAPI()

@app.get("/")
def read_root():
    return {"mensaje": "¡Hola! Este es mi backend con FastAPI"}

@app.get("/saludo/{nombre}")
def generar_saludo(nombre: str):
    prompt = PromptTemplate(
        input_variables=["nombre"],
        template="Escribe un saludo creativo para {nombre}."
    )
    respuesta = llm.invoke(prompt.format(nombre=nombre))
    return {"saludo": respuesta.content}