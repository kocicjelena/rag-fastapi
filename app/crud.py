import uuid
from typing import Any

from sqlmodel import Session, select, func

from app.core.security import get_password_hash, verify_password
from app.models import (
    Document,
    DocumentChunk,
    DocumentCreate,
    User,
    UserCreate,
    UserUpdate,
)


# ──────────────────────────── User ────────────────────────────

def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
    user_data = user_in.model_dump(exclude_unset=True)
    extra_data: dict[str, Any] = {}
    if "password" in user_data:
        extra_data["hashed_password"] = get_password_hash(user_data.pop("password"))
    db_user.sqlmodel_update(user_data, update=extra_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    return session.exec(select(User).where(User.email == email)).first()


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user or not verify_password(password, db_user.hashed_password):
        return None
    return db_user


# ──────────────────────────── Document ────────────────────────────

def create_document(
    *, session: Session, document_in: DocumentCreate, owner_id: uuid.UUID
) -> Document:
    db_doc = Document.model_validate(document_in, update={"owner_id": owner_id})
    session.add(db_doc)
    session.commit()
    session.refresh(db_doc)
    return db_doc


def get_documents(
    *, session: Session, owner_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> tuple[list[Document], int]:
    count = session.exec(
        select(func.count()).where(Document.owner_id == owner_id)
    ).one()
    docs = session.exec(
        select(Document).where(Document.owner_id == owner_id).offset(skip).limit(limit)
    ).all()
    return list(docs), count


def get_document(*, session: Session, doc_id: uuid.UUID) -> Document | None:
    return session.get(Document, doc_id)


def delete_document(*, session: Session, db_doc: Document) -> None:
    session.delete(db_doc)
    session.commit()


# ──────────────────────────── Chunks ────────────────────────────

def create_chunks(
    *, session: Session, chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    for chunk in chunks:
        session.add(chunk)
    session.commit()
    return chunks


def get_chunks_by_document(
    *, session: Session, document_id: uuid.UUID
) -> list[DocumentChunk]:
    return list(
        session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ).all()
    )


def similarity_search(
    *,
    session: Session,
    query_embedding: list[float],
    top_k: int = 5,
    document_ids: list[uuid.UUID] | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Return (chunk, cosine_distance) sorted by ascending distance."""
    stmt = select(
        DocumentChunk,
        DocumentChunk.embedding.cosine_distance(query_embedding).label("score"),  # type: ignore
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))  # type: ignore
    stmt = stmt.order_by("score").limit(top_k)
    rows = session.exec(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
