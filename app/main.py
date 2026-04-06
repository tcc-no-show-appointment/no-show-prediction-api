from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes.prediction_routes import router as prediction_router
from app.routes.appointment_routes import router as appointment_router
from app.services.model_manager import model_manager
from app.utils.logger import get_logger
from app.config import config

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    
    STARTUP: Downloads model and config from blob storage (ONCE)
    SHUTDOWN: Cleanup resources
    """
    # Startup: Load model and config into memory
    logger.info("🚀 APPLICATION STARTUP - Loading model and config...")
    try:
        await model_manager.load_model_and_config()
        logger.info("✅ Startup complete - API ready to accept requests")
    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    logger.info("🛑 APPLICATION SHUTDOWN - Cleaning up resources...")
    await model_manager.cleanup()
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="No-Show Prediction API",
    description="API for predicting whether patients will show up for their medical appointments",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
    root_path=config.ROOT_PATH
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS, POST, GET, etc.
    allow_headers=["*"],  # Allows all headers
)

app.include_router(prediction_router, tags=["predictions"])
app.include_router(appointment_router, tags=["appointments"])

@app.get("", tags=["health"])
@app.get("/", tags=["health"], include_in_schema=False)
async def root():
    return {"status": "ok", "message": "No-Show Prediction API is running"}


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint that verifies models are loaded.
    """
    try:
        effective = model_manager.get_effective_models()
        config = model_manager.get_config()
        thresholds = model_manager.get_thresholds()
        default = model_manager.get_default_model()
        return {
            "status": "healthy",
            "effective_models": len(effective),
            "specialties": list(effective.keys()),
            "dedicated_specialty_models": list(model_manager.get_models().keys()),
            "outras_especialidades_fallback": default is not None,
            "config_loaded": config is not None,
            "thresholds": thresholds,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }