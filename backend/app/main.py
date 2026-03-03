import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import diagnosis, health

app = FastAPI(
    title="Malaria CDSS API",
    description="AI-powered malaria clinical decision support system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(diagnosis.router, prefix="/api/v1")

# Serve React SPA in production (Docker / HF Spaces)
BUILD_DIR = "/app/frontend/build"

if os.path.exists(BUILD_DIR):
    app.mount("/static", StaticFiles(directory=f"{BUILD_DIR}/static"), name="static")

    @app.get("/")
    async def serve_root():
        return FileResponse(f"{BUILD_DIR}/index.html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = f"{BUILD_DIR}/{full_path}"
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{BUILD_DIR}/index.html")

else:

    @app.get("/")
    async def root():
        return {"message": "Malaria CDSS API", "docs": "/docs"}
