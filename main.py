import os
import io
import logging
import uuid # Para manejar UUIDs
from datetime import datetime, timezone # Para 'updated_at' timestamp

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client # Importación correcta
from deepface import DeepFace
from pydantic import BaseModel, Field # Para modelos de API claros
from typing import List, Optional, Dict

# --- Configuración de Logging Detallado ---
# Define un formato más informativo
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__) # Logger específico para este módulo

# --- Cargar Variables de Entorno ---
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY") # ¡Usar SERVICE_ROLE_KEY!

if not supabase_url or not supabase_key:
    logger.critical("FATAL: Las variables de entorno SUPABASE_URL y SUPABASE_KEY no están definidas en .env")
    raise ValueError("Supabase URL y Key deben estar definidas en el archivo .env")

# --- Inicializar Cliente Supabase ---
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Cliente Supabase inicializado correctamente.")
    # Opcional: Hacer una pequeña prueba de conexión si es necesario
    # test_connection = supabase.table('patients').select('id').limit(1).execute()
    # logger.info(f"Prueba de conexión a Supabase: {'Exitosa' if not hasattr(test_connection, 'error') or not test_connection.error else 'Fallida'}")
except Exception as e:
    logger.exception(f"Error CRÍTICO al inicializar el cliente Supabase: {e}")
    raise # Detener la aplicación si no se puede conectar a la DB

# --- Modelos Pydantic para la API (Esquema de Datos) ---
class LinkFaceResponse(BaseModel): # Respuesta para la vinculación
    success: bool
    message: str
    patient_id: Optional[str] = None # UUID como string
    patient_name: Optional[str] = None

class EmotionResult(BaseModel):
    dominant_emotion: str

class IdentificationResultData(BaseModel): # Datos del paciente identificado
    id: str # UUID como string
    nombre_completo: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0) # Similitud entre 0 y 1
    emocion_registro: Optional[str] = None

class IdentifyResponse(BaseModel): # Respuesta para la identificación
    found: bool
    patient: Optional[IdentificationResultData] = None
    current_emotion: EmotionResult

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="CynosureFace - API Backend",
    description="API para vincular rostros a pacientes existentes (por Surecode) e identificar rostros.",
    version="0.6.0" # Incrementar versión por cambio de funcionalidad
)

# --- Configuración de CORS ---
origins = [
    "http://localhost:5173", # Vite
    "http://127.0.0.1:5173",
    # Añadir orígenes de producción si es necesario
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Restringir a métodos usados si se prefiere
    allow_headers=["*"],
)

# --- Funciones Auxiliares ---
async def process_image_for_deepface(image_bytes: bytes) -> np.ndarray:
    """Convierte bytes de imagen a array NumPy (BGR) y valida."""
    logger.debug(f"Procesando imagen ({len(image_bytes)} bytes)...")
    if not image_bytes:
        logger.warning("process_image_for_deepface recibió bytes vacíos.")
        raise HTTPException(status_code=400, detail="Imagen recibida está vacía.")
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Leer como BGR
        if img_np is None:
            logger.error("cv2.imdecode falló. ¿Formato de imagen inválido?")
            raise ValueError("Formato de imagen inválido o archivo corrupto.")
        logger.info(f"Imagen decodificada. Dimensiones: {img_np.shape}")
        return img_np
    except ValueError as ve: # Capturar error de formato inválido
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Error inesperado en process_image_for_deepface: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la imagen.")

# --- Endpoints de la API ---

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz para verificar funcionamiento."""
    logger.info("Acceso al endpoint raíz '/'")
    return {"message": "API CynosureFace v0.6 - Funcionando."}

# Endpoint para VINCULAR rostro (usa la ruta /register por compatibilidad con frontend anterior)
@app.post("/register", response_model=LinkFaceResponse, tags=["Vinculación Facial"])
async def link_face_to_patient_by_surecode(
    surecode: str = Form(..., description="Código único (Surecode) del paciente existente."),
    image: UploadFile = File(..., description="Imagen del rostro a vincular.")
):
    """
    Busca un paciente por Surecode, extrae datos faciales de la imagen y
    ACTUALIZA el registro del paciente encontrado con el vector y emoción.
    NO crea nuevos pacientes.
    """
    request_id = uuid.uuid4() # ID único para trazar esta petición en los logs
    logger.info(f"[Req ID: {request_id}] Solicitud de vinculación iniciada para Surecode: '{surecode}'")

    try:
        # 1. Buscar Paciente por Surecode
        logger.info(f"[Req ID: {request_id}] Buscando paciente con Surecode '{surecode}'...")
        # Seleccionar campos necesarios: id es vital, nombre es útil para respuesta
        lookup_response = supabase.table("patients").select("id, name, nombre_completo").eq("surecode", surecode).limit(1).execute()

        if not lookup_response.data:
            logger.warning(f"[Req ID: {request_id}] Paciente con Surecode '{surecode}' NO encontrado.")
            # Devolver 404 Not Found si el surecode no existe
            raise HTTPException(status_code=404, detail=f"No se encontró paciente con Surecode '{surecode}'. Verifique el código.")

        patient_info = lookup_response.data[0]
        patient_id = patient_info['id']
        patient_name = patient_info.get('nombre_completo') or patient_info.get('name', f"ID:{patient_id[:8]}")
        logger.info(f"[Req ID: {request_id}] Paciente encontrado: ID={patient_id}, Nombre='{patient_name}'.")

        # 2. Procesar Imagen con DeepFace
        logger.info(f"[Req ID: {request_id}] Procesando imagen adjunta...")
        image_bytes = await image.read()
        img_np = await process_image_for_deepface(image_bytes) # Reutiliza la función auxiliar

        logger.info(f"[Req ID: {request_id}] Extrayendo vector facial (ArcFace)...")
        # Asegurarse de que DeepFace detecte una cara
        try:
            embedding_objs = DeepFace.represent(img_path=img_np, model_name='ArcFace', enforce_detection=True, detector_backend='mediapipe')
        except ValueError as deepface_err:
             # Capturar error específico si DeepFace no detecta cara
             logger.warning(f"[Req ID: {request_id}] DeepFace no detectó un rostro en la imagen: {deepface_err}")
             raise HTTPException(status_code=400, detail="No se detectó un rostro claro en la imagen proporcionada.")

        if not embedding_objs or 'embedding' not in embedding_objs[0]:
             logger.error(f"[Req ID: {request_id}] DeepFace.represent no devolvió embedding.")
             raise HTTPException(status_code=500, detail="Error al extraer datos faciales (represent).")
        embedding = embedding_objs[0]["embedding"]
        logger.info(f"[Req ID: {request_id}] Vector facial extraído.")

        logger.info(f"[Req ID: {request_id}] Analizando emoción...")
        try:
            analysis = DeepFace.analyze(img_path=img_np, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', silent=True)
            dominant_emotion_registro = analysis[0]["dominant_emotion"] if (analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]) else "desconocida"
        except Exception as emotion_err:
             logger.warning(f"[Req ID: {request_id}] Error analizando emoción: {emotion_err}. Usando 'desconocida'.")
             dominant_emotion_registro = "desconocida"
        logger.info(f"[Req ID: {request_id}] Emoción detectada: {dominant_emotion_registro}")

        # 3. Actualizar Paciente en Supabase
        vector_list = np.array(embedding).tolist()
        data_to_update = {
            "embedding_facial": vector_list,
            "emocion_registro": dominant_emotion_registro,
            "updated_at": datetime.now(timezone.utc).isoformat() # Actualizar timestamp
        }

        logger.info(f"[Req ID: {request_id}] Actualizando paciente ID {patient_id} en Supabase...")
        update_response = supabase.table("patients").update(data_to_update).eq("id", patient_id).execute()

        # Verificar errores de Supabase en la actualización
        if hasattr(update_response, 'error') and update_response.error:
             error_msg = update_response.error.get('message', str(update_response.error))
             logger.error(f"[Req ID: {request_id}] Error Supabase al actualizar: {error_msg}")
             raise HTTPException(status_code=500, detail=f"Error DB al actualizar: {error_msg}")
        # Verificar si se actualizó alguna fila (opcional, puede depender de RLS)
        # count = update_response.count if hasattr(update_response, 'count') else len(update_response.data) if hasattr(update_response, 'data') else None
        # logger.info(f"[Req ID: {request_id}] Filas actualizadas: {count}")

        logger.info(f"[Req ID: {request_id}] Vinculación facial completada para Surecode '{surecode}'.")
        return LinkFaceResponse(
            success=True,
            message=f"Rostro vinculado exitosamente a: {patient_name}",
            patient_id=str(patient_id),
            patient_name=patient_name
        )

    except HTTPException as http_exc:
        # Re-lanzar excepciones HTTP ya manejadas (ej: 404 de paciente no encontrado)
        logger.warning(f"[Req ID: {request_id}] Excepción HTTP manejada: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except ValueError as ve:
        # Errores de validación o de DeepFace (no detectó cara)
        logger.warning(f"[Req ID: {request_id}] Error de Valor: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Captura de errores inesperados
        logger.exception(f"[Req ID: {request_id}] Error inesperado en vinculación para Surecode '{surecode}': {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor durante la vinculación.")


@app.post("/identify", response_model=IdentifyResponse, tags=["Identificación Facial"])
async def identify_patient_by_face(
    # Recibe el umbral, pero usa el default 0.7 si no se envía
    threshold: float = Form(0.7, ge=0.4, le=1.0, description="Umbral mínimo de similitud (0.4 a 1.0). Default: 0.7"),
    image: UploadFile = File(..., description="Imagen del rostro a identificar.")
):
    """
    Identifica un paciente a partir de una imagen facial.
    Busca en la base de datos el rostro más similar que supere el umbral.
    """
    request_id = uuid.uuid4()
    logger.info(f"[Req ID: {request_id}] Solicitud de identificación iniciada. Umbral efectivo: {threshold:.2f}")

    try:
        # 1. Procesar Imagen
        logger.info(f"[Req ID: {request_id}] Procesando imagen para identificación...")
        image_bytes = await image.read()
        img_np = await process_image_for_deepface(image_bytes)

        # 2. Extraer Vector Facial de Consulta
        logger.info(f"[Req ID: {request_id}] Extrayendo vector facial de consulta (ArcFace)...")
        try:
            embedding_objs = DeepFace.represent(img_path=img_np, model_name='ArcFace', enforce_detection=True, detector_backend='mediapipe')
        except ValueError as deepface_err:
             logger.warning(f"[Req ID: {request_id}] DeepFace no detectó rostro en consulta: {deepface_err}")
             raise HTTPException(status_code=400, detail="No se detectó un rostro claro en la imagen.")

        if not embedding_objs or 'embedding' not in embedding_objs[0]:
             logger.error(f"[Req ID: {request_id}] DeepFace.represent no devolvió embedding para consulta.")
             raise HTTPException(status_code=500, detail="Error al extraer datos faciales (consulta).")
        query_embedding = embedding_objs[0]["embedding"]
        logger.info(f"[Req ID: {request_id}] Vector de consulta extraído.")

        # 3. Analizar Emoción de Consulta
        logger.info(f"[Req ID: {request_id}] Analizando emoción de consulta...")
        try:
            analysis = DeepFace.analyze(img_path=img_np, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', silent=True)
            current_dominant_emotion = analysis[0]["dominant_emotion"] if (analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]) else "desconocida"
        except Exception as emotion_err:
             logger.warning(f"[Req ID: {request_id}] Error analizando emoción consulta: {emotion_err}. Usando 'desconocida'.")
             current_dominant_emotion = "desconocida"
        logger.info(f"[Req ID: {request_id}] Emoción consulta: {current_dominant_emotion}")
        current_emotion_details = EmotionResult(dominant_emotion=current_dominant_emotion)

        # 4. Buscar Coincidencia en Supabase (usando función RPC)
        vector_list = np.array(query_embedding).tolist()
        logger.info(f"[Req ID: {request_id}] Llamando a RPC 'match_patients' con umbral {threshold:.2f}...")
        rpc_response = supabase.rpc(
            'match_patients',
            {'query_embedding': vector_list, 'match_threshold': threshold}
        ).execute()

        # Verificar errores de la llamada RPC
        if hasattr(rpc_response, 'error') and rpc_response.error:
            error_msg = rpc_response.error.get('message', str(rpc_response.error))
            logger.error(f"[Req ID: {request_id}] Error Supabase RPC (match_patients): {error_msg}")
            raise HTTPException(status_code=500, detail=f"Error DB durante búsqueda: {error_msg}")

        # 5. Procesar Resultado de la Búsqueda
        if rpc_response.data:
            # Éxito: Se encontró al menos una coincidencia
            match = rpc_response.data[0] # La función SQL devuelve el mejor match
            patient_data = IdentificationResultData(
                id=str(match['id']),
                nombre_completo=match.get('nombre_completo'),
                similarity=match['similarity'],
                emocion_registro=match.get('emocion_registro')
            )
            logger.info(f"[Req ID: {request_id}] Coincidencia encontrada: ID={patient_data.id}, Nombre='{patient_data.nombre_completo}', Sim={patient_data.similarity:.4f}")
            return IdentifyResponse(found=True, patient=patient_data, current_emotion=current_emotion_details)
        else:
            # No se encontraron coincidencias por encima del umbral
            logger.info(f"[Req ID: {request_id}] No se encontraron coincidencias válidas.")
            return IdentifyResponse(found=False, patient=None, current_emotion=current_emotion_details)

    except HTTPException as http_exc:
        logger.warning(f"[Req ID: {request_id}] Excepción HTTP manejada: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except ValueError as ve:
        logger.warning(f"[Req ID: {request_id}] Error de Valor: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"[Req ID: {request_id}] Error inesperado en identificación: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor durante la identificación.")


# --- Ejecución Local ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn FastAPI en http://0.0.0.0:8000 (Modo Desarrollo con Recarga)")
    uvicorn.run(
        "main:app",         # Nombre del archivo y la instancia de FastAPI
        host="0.0.0.0",     # Escuchar en todas las interfaces de red
        port=8000,          # Puerto estándar
        reload=True,        # Reiniciar automáticamente con cambios en el código
        log_level="info"    # Nivel de log para Uvicorn
    )