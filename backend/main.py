"""
Cerebrum - AI-native second brain backend
Main FastAPI application entry point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from typing import Dict, Any
from routes import memory, cognitive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("cerebrum")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    logger.info("🧠 Cerebrum backend starting up...")
    
    # Initialize memory store on startup
    try:
        await memory.initialize_memory_store()
        logger.info("✅ Memory store initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory store: {e}")
        raise
    
    yield
    
    logger.info("🧠 Cerebrum backend shutting down...")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Cerebrum API",
    description="AI-native second brain with long-term memory capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Alternative local port
        "https://*.vercel.app",   # Vercel deployments
        "https://*.cerebrum.ai",  # Custom domain (if applicable)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """Root health check endpoint"""
    return {
        "status": "healthy",
        "service": "cerebrum-backend",
        "version": "1.0.0",
        "message": "🧠 Cerebrum AI Memory System Online"
    }

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "memory_store": "operational",
        "cognitive_engine": "operational"
    }

# Include routers
app.include_router(
    memory.router,
    prefix="/memory",
    tags=["Memory"]
)

app.include_router(
    cognitive.router,
    prefix="/cognitive",
    tags=["Cognitive"]
)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )
