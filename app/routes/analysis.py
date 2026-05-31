import io
import os
import shutil
import tempfile
import logging
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from app.services.agent import workflow, fix_step_callback
from app.services.executor import apply_transformation_plan
from app.schemas.analysis import HealthCheckResponse
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check() -> dict:
    """Returns the application health status and current configuration environment."""
    logger.info("Health check endpoint hit.")
    return {
        "status": "healthy",
        "message": "API is healthy",
        "environment": settings.app_env
    }


@router.post("/processing")
async def processing_file(
    file: UploadFile = File(...),
    instruction: str = Form(...),
):
    """
    Ingests a CSV file and a natural language instruction,
    processes it through the LangGraph workflow, and returns either a text answer 
    or a transformed CSV file.
    """
    logger.info(f"Received file upload '{file.filename}' with instruction: '{instruction}'")
    
    # Enforce CSV format
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    temp_dir = tempfile.gettempdir()
    tmp_path = None

    try:
        # Write uploaded stream to local temp file safely
        with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".csv") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        logger.info(f"Saved uploaded file to temporary path: {tmp_path}")

        # Run LangGraph Agent workflow
        state_input = {"instruction": instruction, "file_path": tmp_path}
        op = workflow.invoke(state_input)
        
        output_mode = op.get("output_mode")
        logger.info(f"Agent flow completed with output mode: {output_mode}")

        if output_mode == "text":
            return op.get("text_output", "")

        # CSV transformation pathway
        plan = op.get("transformation_plan") or []
        logger.info(f"Applying CSV transformation plan steps: {plan}")

        df = pd.read_csv(tmp_path)
        new_df = apply_transformation_plan(
            df,
            plan,
            op.get("instruction", instruction),
            fix_step_callback
        )

        # Buffer transformed DataFrame to CSV stream
        buffer = io.StringIO()
        new_df.to_csv(buffer, index=False)
        buffer.seek(0)

        filename_prefix = "transformed_"
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename_prefix}{file.filename}"
            },
        )

    except Exception as e:
        logger.exception("An error occurred during file processing execution")
        # Do not expose raw traceback details in production exceptions
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing the dataset: {str(e)}"
        )

    finally:
        # Guarantee removal of temporary file to avoid local disk leaks
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info(f"Temporary file {tmp_path} cleaned up successfully.")
            except Exception as cleanup_err:
                logger.error(f"Failed to delete temporary file {tmp_path}: {cleanup_err}")
