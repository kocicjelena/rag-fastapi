import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, Relationship, SQLModel  # type: ignore[import]
# ──────────────────────────── User ────────────────────────────

class UserBase(SQLModel):
    email: str = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    documents: list["Document"] = Relationship(back_populates="owner")


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=40)


class UserUpdate(UserBase):
    email: str | None = Field(default=None, max_length=255)  # type: ignore
    password: str | None = Field(default=None, min_length=8, max_length=40)


class UserPublic(UserBase):
    id: uuid.UUID


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# ──────────────────────────── Document ────────────────────────────

class DocumentBase(SQLModel):
    title: str = Field(max_length=255)
    description: str | None = Field(default=None, max_length=1000)


class Document(DocumentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: uuid.UUID = Field(foreign_key="user.id", nullable=False, ondelete="CASCADE")
    owner: User | None = Relationship(back_populates="documents")
    chunks: list["DocumentChunk"] = Relationship(back_populates="document")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_type: str | None = Field(default=None, max_length=50)
    char_count: int = Field(default=0)
    chunk_count: int = Field(default=0)
    status: str = Field(default="pending", max_length=50)  # pending | processing | ready | error


class DocumentCreate(DocumentBase):
    pass


class DocumentPublic(DocumentBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime
    file_type: str | None
    char_count: int
    chunk_count: int
    status: str


class DocumentsPublic(SQLModel):
    data: list[DocumentPublic]
    count: int


# ──────────────────────────── DocumentChunk ────────────────────────────

class DocumentChunk(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        foreign_key="document.id", nullable=False, ondelete="CASCADE"
    )
    document: Document | None = Relationship(back_populates="chunks")
    content: str
    chunk_index: int
    embedding: Optional[list[float]] = Field(
        default=None, sa_column=Column(Vector(1536))
    )


# ──────────────────────────── Query / Chat ────────────────────────────

class QueryRequest(SQLModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = None  # restrict search to specific docs
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkResult(SQLModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    content: str
    score: float


class QueryResponse(SQLModel):
    question: str
    answer: str
    sources: list[ChunkResult]


# ──────────────────────────── Auth ────────────────────────────

class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(SQLModel):
    sub: str | None = None


class Message(SQLModel):
    message: str
