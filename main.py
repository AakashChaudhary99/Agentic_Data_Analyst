import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Launch refactored FastAPI app using configured host, port, and reload options
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development
    )
