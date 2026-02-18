from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes.prediction_routes import router as prediction_router
from app.routes.appointment_routes import router as appointment_router
from app.services.model_manager import model_manager
from app.utils.logger import get_logger

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
    redirect_slashes=False
)

app.include_router(prediction_router, tags=["predictions"])
app.include_router(appointment_router, tags=["appointments"])

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "message": "No-Show Prediction API is running"}


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint that verifies model is loaded.
    """
    try:
        model = model_manager.get_model()
        config = model_manager.get_config()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "config_loaded": config is not None,
            "model_type": type(model).__name__ if model else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }