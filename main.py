from fastapi import FastAPI

from classification.router import router as classification_router

app = FastAPI(title="Plant Care AI Service")

app.include_router(classification_router, prefix="/classify", tags=["Classification"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Care AI Service!"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}