import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.models import DocumentCreate, DocumentPublic, DocumentsPublic, Message
from app.services import rag

router = APIRouter(prefix="/documents", tags=["documents"])


def _process_document(
    document_id: uuid.UUID,
    text: str,
    session_factory: Any,
) -> None:
    """Background task: chunk + embed and persist chunks."""
    from app.core.db import engine
    from sqlmodel import Session

    with Session(engine) as session:
        doc = session.get(__import__("app.models", fromlist=["Document"]).Document, document_id)
        if not doc:
            return
        doc.status = "processing"
        session.add(doc)
        session.commit()

        try:
            chunks = rag.ingest_text(text=text, document_id=document_id)
            crud.create_chunks(session=session, chunks=chunks)
            doc.chunk_count = len(chunks)
            doc.char_count = len(text)
            doc.status = "ready"
        except Exception:
            doc.status = "error"
        finally:
            session.add(doc)
            session.commit()


@router.get("/", response_model=DocumentsPublic)
def list_documents(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    docs, count = crud.get_documents(
        session=session, owner_id=current_user.id, skip=skip, limit=limit
    )
    return DocumentsPublic(data=docs, count=count)


@router.post("/upload", response_model=DocumentPublic, status_code=201)
async def upload_document(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    title: str | None = None,
    description: str | None = None,
) -> Any:
    """
    Upload a plain-text or PDF file. Chunking and embedding run in the background.
    Supported types: text/plain, application/pdf.
    """
    allowed_types = {"text/plain", "application/pdf", "text/markdown", "text/csv"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}",
        )

    raw_bytes = await file.read()

    # Extract text
    if file.content_type == "application/pdf":
        try:
            import pypdf
            import io

            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    doc_in = DocumentCreate(
        title=title or file.filename or "Untitled",
        description=description,
    )
    doc = crud.create_document(
        session=session, document_in=doc_in, owner_id=current_user.id
    )
    doc.file_type = file.content_type
    session.add(doc)
    session.commit()
    session.refresh(doc)

    background_tasks.add_task(_process_document, doc.id, text, None)
    return doc


@router.get("/{document_id}", response_model=DocumentPublic)
def get_document(
    document_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    doc = crud.get_document(session=session, doc_id=document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return doc


@router.delete("/{document_id}")
def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> Message:
    doc = crud.get_document(session=session, doc_id=document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    crud.delete_document(session=session, db_doc=doc)
    return Message(message="Document deleted successfully")
