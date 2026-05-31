import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.utils.logging_config import setup_logging
from app.routes.analysis import router as analysis_router

# 1. Initialize logging system
setup_logging()
logger = logging.getLogger(__name__)

# 2. Guarantee OpenAI environment configuration for sub-components
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI startup and teardown events."""
    logger.info("Initializing Agentic Data Analyst API service...")
    yield
    logger.info("Shutting down Agentic Data Analyst API service...")


# 3. Instantiate FastAPI application
app = FastAPI(
    title="Agentic Data Analyst API",
    description="Enterprise-grade FastAPI service orchestrating LangGraph data analysis agents.",
    version="1.0.0",
    lifespan=lifespan
)

# 4. Standard CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Route registration
# Support both the root endpoints (for backwards compatibility) and a modular path
app.include_router(analysis_router)


# Supporting legacy route path directly at root level
@app.get("/")
def legacy_health_check() -> dict:
    logger.info("Root endpoint health check hit.")
    return {"Message": "API is healthy"}
