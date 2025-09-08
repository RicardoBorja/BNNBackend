import os
import aiofiles
import pandas as pd

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from app.config import conf_manager
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.service.file_service import FileService

router = APIRouter()
file_service = FileService()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class DatasetSelection(BaseModel):
    file_path: str
    has_header: bool


@router.get("/list")
async def get_available_files():
    """
    Get all available data files categorized by type (csv, image)

    Returns:
        Dict with two keys: 'csv' and 'image', each containing a list of file paths
    """
    try:
        files = file_service.get_available_files()
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select")
async def select_dataset(selection: DatasetSelection):
    """
    Selects a dataset (without returning its preview).
    """
    try:
        file_service.set_selected_dataset(selection.file_path, selection.has_header)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Selected file set to {selection.file_path}"}


@router.get("/preview")
async def get_file_preview(
    file_path: str = Query(..., description="Relative file path inside the data folder")
):
    """
    Returns a preview (first 10 lines) of the specified dataset.
    For "mnist" the preview is not applicable (returns None).
    """
    try:
        preview = []
        if file_path.lower() == "mnist":
            preview = None
        else:
            # Get the directory of this file (app/controller)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # If file_path starts with "data", remove it to avoid duplication.
            if file_path.startswith("data" + os.path.sep):
                relative_path = file_path[len("data" + os.path.sep) :]
            else:
                relative_path = file_path
            # Build the full path: (../data relative to app/controller)
            full_path = os.path.join(base_dir, "..", "data", relative_path)
            with open(full_path, "r") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    preview.append(line.rstrip("\n"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"preview": preview}


@router.post("/upload")
async def upload(file: UploadFile = File(...), separator: str = Form(",")):

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        contents = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong: " + str(e))
    finally:
        await file.close()

    try:
        df = pd.read_csv(file_path, sep=separator)
        # Replace out-of-range float values (like NaN) with None
        df = df.where(pd.notnull(df), None)
        preview = df.head(10).to_dict(orient="records")

        # Update configuration with the new file settings
        conf_manager.set_value("selected_file", safe_filename)
        conf_manager.set_value("loaded_data_path", file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload succeeded, but file could not be loaded: {e}",
        )

    return {
        "message": f"Successfully uploaded and loaded {safe_filename}",
        "preview": preview,
    }


@router.get("/current")
async def get_current_dataset():
    """
    Get the currently selected dataset from settings.conf

    Returns:
        Dict with the current dataset file path
    """
    try:
        return file_service.get_current_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
