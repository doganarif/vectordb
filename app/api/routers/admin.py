from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.services import VectorDBService, get_service

router = APIRouter()


class SaveResponse(BaseModel):
    saved_to: str


class LoadResponse(BaseModel):
    status: str


@router.post("/snapshots", response_model=SaveResponse)
def save_snapshot(service: VectorDBService = Depends(get_service)) -> dict[str, str]:
    path = service.save()
    return {"saved_to": str(path)}


@router.put("/snapshots", response_model=LoadResponse)
def load_snapshot(service: VectorDBService = Depends(get_service)) -> dict[str, str]:
    service.load()
    return {"status": "loaded"}
