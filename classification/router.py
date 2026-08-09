from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from .preprocess import preprocess_image
from .inference import predict
from .schemas import ClassifyResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}


@router.post("/", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
async def classify_image(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    image_bytes = await file.read()

    try:
        image_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded image. The file may be corrupted or invalid."
        )

    try:
        result = predict(image_array)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to make a prediction with the model."
        )

    return result