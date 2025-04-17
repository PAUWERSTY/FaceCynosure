import os
import io
import logging
import uuid # Para manejar UUIDs
from datetime import datetime, timezone # Para 'updated_at' timestamp

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Importar Middleware
from supabase import create_client, Client # Importación correcta
from deepface import DeepFace
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- Configuración de Logging Detallado ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Cargar Variables de Entorno ---
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("FATAL: Variables SUPABASE_URL y SUPABASE_KEY no definidas.")
    raise ValueError("Supabase URL y Key deben estar definidas en .env")

# --- Inicializar Cliente Supabase ---
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Cliente Supabase inicializado correctamente.")
except Exception as e:
    logger.exception(f"Error CRÍTICO al inicializar Supabase: {e}")
    raise

# --- Modelos Pydantic ---
class LinkFaceResponse(BaseModel):
    success: bool; message: str; patient_id: Optional[str] = None; patient_name: Optional[str] = None
class EmotionResult(BaseModel):
    dominant_emotion: str
class IdentificationResultData(BaseModel):
    id: str; nombre_completo: Optional[str] = None; similarity: float = Field(..., ge=0.0, le=1.0); emocion_registro: Optional[str] = None
class IdentifyResponse(BaseModel):
    found: bool; patient: Optional[IdentificationResultData] = None; current_emotion: EmotionResult

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="CynosureFace - API Backend",
    description="API para vincular y reconocer rostros.",
    version="0.6.1" # Versión corregida
)

# --- Configuración de CORS (CORREGIDA) ---
origins = [
    "http://localhost:5173",  # Permitir frontend local Vite <--- DESCOMENTADO
    "http://127.0.0.1:5173", # Permitir otra IP local       <--- DESCOMENTADO
    # Si despliegas tu frontend a producción, añade su URL HTTPS aquí:
    # "https://tu-frontend-app.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Usar la lista correcta de orígenes
    allow_credentials=True,     # Generalmente bueno tenerlo
    allow_methods=["POST", "GET"],# Métodos permitidos (ajusta si necesitas otros)
    allow_headers=["*"],        # Permitir todas las cabeceras comunes
)
# --- FIN DE CORRECCIÓN CORS ---


# --- Funciones Auxiliares ---
async def process_image_for_deepface(image_bytes: bytes) -> np.ndarray:
    # ... (código igual que antes) ...
    logger.debug(f"Procesando imagen ({len(image_bytes)} bytes)...")
    if not image_bytes: logger.warning("..."); raise HTTPException(status_code=400, detail="Imagen vacía.")
    try:
        nparr = np.frombuffer(image_bytes, np.uint8); img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None: logger.error("..."); raise ValueError("Formato inválido.")
        logger.info(f"Imagen decodificada: {img_np.shape}"); return img_np
    except ValueError as ve: raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e: logger.exception("..."); raise HTTPException(status_code=500, detail="Error procesando imagen.")

# --- Endpoints de la API ---

@app.get("/", tags=["General"])
async def root():
    # ... (código igual que antes) ...
    logger.info("Acceso al endpoint raíz '/'"); return {"message": "API CynosureFace v0.6.1 - Funcionando."}

@app.post("/register", response_model=LinkFaceResponse, tags=["Vinculación Facial"])
async def link_face_to_patient_by_surecode(surecode: str = Form(...), image: UploadFile = File(...)):
    # ... (código igual que antes) ...
    request_id = uuid.uuid4(); logger.info(f"[Req ID: {request_id}] Vinculación iniciada para Surecode: '{surecode}'")
    try:
        logger.info(f"[Req ID: {request_id}] Buscando Surecode '{surecode}'..."); lookup_response = supabase.table("patients").select("id, name, nombre_completo").eq("surecode", surecode).limit(1).execute()
        if not lookup_response.data: logger.warning(f"[Req ID: {request_id}] Surecode '{surecode}' NO encontrado."); raise HTTPException(status_code=404, detail=f"Surecode '{surecode}' no encontrado.")
        patient_info = lookup_response.data[0]; patient_id = patient_info['id']; patient_name = patient_info.get('nombre_completo') or patient_info.get('name', f"ID:{patient_id[:8]}"); logger.info(f"[Req ID: {request_id}] Paciente encontrado: ID={patient_id}, Nombre='{patient_name}'.")
        logger.info(f"[Req ID: {request_id}] Procesando imagen..."); image_bytes = await image.read(); img_np = await process_image_for_deepface(image_bytes)
        logger.info(f"[Req ID: {request_id}] Extrayendo vector (ArcFace)...");
        try: embedding_objs = DeepFace.represent(img_path=img_np, model_name='ArcFace', enforce_detection=True, detector_backend='mediapipe')
        except ValueError as deepface_err: logger.warning(f"[Req ID: {request_id}] DeepFace no detectó rostro: {deepface_err}"); raise HTTPException(status_code=400, detail="No se detectó rostro claro.")
        if not embedding_objs or 'embedding' not in embedding_objs[0]: logger.error(f"[Req ID: {request_id}] Represent no devolvió embedding."); raise HTTPException(status_code=500, detail="Error extracción facial (represent).")
        embedding = embedding_objs[0]["embedding"]; logger.info(f"[Req ID: {request_id}] Vector extraído.")
        logger.info(f"[Req ID: {request_id}] Analizando emoción...");
        try: analysis = DeepFace.analyze(img_path=img_np, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', silent=True); dominant_emotion_registro = analysis[0]["dominant_emotion"] if (analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]) else "desconocida"
        except Exception as emotion_err: logger.warning(f"[Req ID: {request_id}] Error emoción: {emotion_err}."); dominant_emotion_registro = "desconocida"
        logger.info(f"[Req ID: {request_id}] Emoción: {dominant_emotion_registro}")
        vector_list = np.array(embedding).tolist(); data_to_update = {"embedding_facial": vector_list, "emocion_registro": dominant_emotion_registro, "updated_at": datetime.now(timezone.utc).isoformat()}
        logger.info(f"[Req ID: {request_id}] Actualizando paciente ID {patient_id}..."); update_response = supabase.table("patients").update(data_to_update).eq("id", patient_id).execute()
        if hasattr(update_response, 'error') and update_response.error: error_msg = update_response.error.get('message', str(update_response.error)); logger.error(f"[Req ID: {request_id}] Error Supabase al actualizar: {error_msg}"); raise HTTPException(status_code=500, detail=f"Error DB al actualizar: {error_msg}")
        logger.info(f"[Req ID: {request_id}] Vinculación completada para Surecode '{surecode}'.")
        return LinkFaceResponse(success=True, message=f"Rostro vinculado a: {patient_name}", patient_id=str(patient_id), patient_name=patient_name)
    except HTTPException as http_exc: logger.warning(f"[Req ID: {request_id}] HTTP Exc: {http_exc.status_code} - {http_exc.detail}"); raise http_exc
    except ValueError as ve: logger.warning(f"[Req ID: {request_id}] Value Err: {ve}"); raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e: logger.exception(f"[Req ID: {request_id}] Error inesperado vinculación '{surecode}': {e}"); raise HTTPException(status_code=500, detail="Error interno servidor (vinculación).")

@app.post("/identify", response_model=IdentifyResponse, tags=["Identificación Facial"])
async def identify_patient_by_face(threshold: float = Form(0.7, ge=0.4, le=1.0), image: UploadFile = File(...)):
    # ... (código igual que antes) ...
    request_id = uuid.uuid4(); logger.info(f"[Req ID: {request_id}] Identificación iniciada. Umbral: {threshold:.2f}")
    try:
        logger.info(f"[Req ID: {request_id}] Procesando imagen..."); image_bytes = await image.read(); img_np = await process_image_for_deepface(image_bytes)
        logger.info(f"[Req ID: {request_id}] Extrayendo vector consulta (ArcFace)...");
        try: embedding_objs = DeepFace.represent(img_path=img_np, model_name='ArcFace', enforce_detection=True, detector_backend='mediapipe')
        except ValueError as deepface_err: logger.warning(f"[Req ID: {request_id}] DeepFace no detectó rostro consulta: {deepface_err}"); raise HTTPException(status_code=400, detail="No se detectó rostro claro.")
        if not embedding_objs or 'embedding' not in embedding_objs[0]: logger.error(f"[Req ID: {request_id}] Represent consulta no devolvió embedding."); raise HTTPException(status_code=500, detail="Error extracción facial (consulta).")
        query_embedding = embedding_objs[0]["embedding"]; logger.info(f"[Req ID: {request_id}] Vector consulta extraído.")
        logger.info(f"[Req ID: {request_id}] Analizando emoción consulta...");
        try: analysis = DeepFace.analyze(img_path=img_np, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe', silent=True); current_dominant_emotion = analysis[0]["dominant_emotion"] if (analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]) else "desconocida"
        except Exception as emotion_err: logger.warning(f"[Req ID: {request_id}] Error emoción consulta: {emotion_err}."); current_dominant_emotion = "desconocida"
        logger.info(f"[Req ID: {request_id}] Emoción consulta: {current_dominant_emotion}"); current_emotion_details = EmotionResult(dominant_emotion=current_dominant_emotion)
        vector_list = np.array(query_embedding).tolist(); logger.info(f"[Req ID: {request_id}] Llamando RPC 'match_patients' umbral {threshold:.2f}..."); rpc_response = supabase.rpc('match_patients', {'query_embedding': vector_list, 'match_threshold': threshold}).execute()
        if hasattr(rpc_response, 'error') and rpc_response.error: error_msg = rpc_response.error.get('message', str(rpc_response.error)); logger.error(f"[Req ID: {request_id}] Error Supabase RPC: {error_msg}"); raise HTTPException(status_code=500, detail=f"Error DB búsqueda: {error_msg}")
        if rpc_response.data:
            match = rpc_response.data[0]; patient_data = IdentificationResultData(id=str(match['id']), nombre_completo=match.get('nombre_completo'), similarity=match['similarity'], emocion_registro=match.get('emocion_registro')); logger.info(f"[Req ID: {request_id}] Coincidencia: ID={patient_data.id}, Nombre='{patient_data.nombre_completo}', Sim={patient_data.similarity:.4f}"); return IdentifyResponse(found=True, patient=patient_data, current_emotion=current_emotion_details)
        else: logger.info(f"[Req ID: {request_id}] No se encontraron coincidencias."); return IdentifyResponse(found=False, patient=None, current_emotion=current_emotion_details)
    except HTTPException as http_exc: logger.warning(f"[Req ID: {request_id}] HTTP Exc: {http_exc.status_code} - {http_exc.detail}"); raise http_exc
    except ValueError as ve: logger.warning(f"[Req ID: {request_id}] Value Err: {ve}"); raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e: logger.exception(f"[Req ID: {request_id}] Error inesperado identificación: {e}"); raise HTTPException(status_code=500, detail="Error interno servidor (identificación).")

# --- Ejecución Local ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn FastAPI en http://0.0.0.0:8000 (Modo Desarrollo con Recarga)")
    uvicorn.run( "main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
