from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.services import VectorDBService, get_service

router = APIRouter()


class SnapshotDTO(BaseModel):
    """Represents a database snapshot."""

    id: str = Field(..., description="Unique snapshot identifier")
    name: str = Field(..., description="Snapshot name")
    path: str = Field(..., description="Snapshot file path")
    created_at: datetime = Field(..., description="Creation timestamp")
    size_bytes: Optional[int] = Field(None, description="Snapshot size in bytes")


class CreateSnapshotDTO(BaseModel):
    """Request to create a new snapshot."""

    name: Optional[str] = Field(None, description="Optional snapshot name")


class RestoreSnapshotDTO(BaseModel):
    """Response for snapshot restore operation."""

    status: str = Field(..., description="Restore operation status")
    message: str = Field(..., description="Detailed status message")


class SnapshotListDTO(BaseModel):
    """List of available snapshots."""

    snapshots: list[SnapshotDTO] = Field(..., description="Available snapshots")
    total: int = Field(..., description="Total number of snapshots")


@router.get("/snapshots", response_model=SnapshotListDTO)
def list_snapshots(service: VectorDBService = Depends(get_service)) -> SnapshotListDTO:
    """List all available database snapshots.

    Returns:
        SnapshotListDTO with list of available snapshots
    """
    # Get the data directory
    data_dir = Path(service.snapshots._data_dir)
    snapshot_files = list(data_dir.glob("snapshot_*.json"))

    snapshots = []
    for file_path in sorted(snapshot_files, reverse=True):
        # Extract timestamp from filename (snapshot_YYYYMMDD_HHMMSS.json)
        filename = file_path.stem
        if filename.startswith("snapshot_"):
            timestamp_str = filename.replace("snapshot_", "")
            try:
                # Parse timestamp
                created_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                created_at = datetime.fromtimestamp(file_path.stat().st_mtime)

            snapshots.append(
                SnapshotDTO(
                    id=file_path.stem,
                    name=file_path.stem,
                    path=str(file_path),
                    created_at=created_at,
                    size_bytes=file_path.stat().st_size,
                )
            )

    return SnapshotListDTO(snapshots=snapshots, total=len(snapshots))


@router.post(
    "/snapshots", response_model=SnapshotDTO, status_code=status.HTTP_201_CREATED
)
def create_snapshot(
    payload: Optional[CreateSnapshotDTO] = None,
    service: VectorDBService = Depends(get_service),
) -> SnapshotDTO:
    """Create a new database snapshot.

    Args:
        payload: Optional snapshot configuration

    Returns:
        SnapshotDTO with created snapshot details
    """
    # Save the snapshot
    path = service.snapshots.save()

    # Generate snapshot ID and name
    snapshot_id = path.stem
    snapshot_name = payload.name if payload and payload.name else snapshot_id

    return SnapshotDTO(
        id=snapshot_id,
        name=snapshot_name,
        path=str(path),
        created_at=datetime.now(),
        size_bytes=path.stat().st_size if path.exists() else None,
    )


@router.get("/snapshots/{snapshot_id}", response_model=SnapshotDTO)
def get_snapshot(
    snapshot_id: str, service: VectorDBService = Depends(get_service)
) -> SnapshotDTO:
    """Get details of a specific snapshot.

    Args:
        snapshot_id: The snapshot identifier

    Returns:
        SnapshotDTO with snapshot details

    Raises:
        HTTPException: If snapshot not found
    """
    data_dir = Path(service.snapshots._data_dir)
    snapshot_path = data_dir / f"{snapshot_id}.json"

    if not snapshot_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot not found: {snapshot_id}",
        )

    # Extract timestamp from filename
    if snapshot_id.startswith("snapshot_"):
        timestamp_str = snapshot_id.replace("snapshot_", "")
        try:
            created_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            created_at = datetime.fromtimestamp(snapshot_path.stat().st_mtime)
    else:
        created_at = datetime.fromtimestamp(snapshot_path.stat().st_mtime)

    return SnapshotDTO(
        id=snapshot_id,
        name=snapshot_id,
        path=str(snapshot_path),
        created_at=created_at,
        size_bytes=snapshot_path.stat().st_size,
    )


@router.post(
    "/snapshots/{snapshot_id}/restore",
    response_model=RestoreSnapshotDTO,
    status_code=status.HTTP_200_OK,
)
def restore_snapshot(
    snapshot_id: str, service: VectorDBService = Depends(get_service)
) -> RestoreSnapshotDTO:
    """Restore database from a specific snapshot.

    Args:
        snapshot_id: The snapshot identifier to restore from

    Returns:
        RestoreSnapshotDTO with operation status

    Raises:
        HTTPException: If snapshot not found or restore fails
    """
    data_dir = Path(service.snapshots._data_dir)

    # Support both with and without .json extension
    if snapshot_id.endswith(".json"):
        snapshot_path = data_dir / snapshot_id
    else:
        snapshot_path = data_dir / f"{snapshot_id}.json"

    if not snapshot_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot not found: {snapshot_id}",
        )

    try:
        # Load the specific snapshot (synchronous operation)
        service.snapshots.load(snapshot_path)

        return RestoreSnapshotDTO(
            status="completed",
            message=f"Database restored from snapshot: {snapshot_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore snapshot: {str(e)}",
        )


@router.delete("/snapshots/{snapshot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_snapshot(
    snapshot_id: str, service: VectorDBService = Depends(get_service)
) -> None:
    """Delete a specific snapshot.

    Args:
        snapshot_id: The snapshot identifier to delete

    Raises:
        HTTPException: If snapshot not found
    """
    data_dir = Path(service.snapshots._data_dir)

    # Support both with and without .json extension
    if snapshot_id.endswith(".json"):
        snapshot_path = data_dir / snapshot_id
    else:
        snapshot_path = data_dir / f"{snapshot_id}.json"

    if not snapshot_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot not found: {snapshot_id}",
        )

    try:
        snapshot_path.unlink()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete snapshot: {str(e)}",
        )
