from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Annotated, Any
from pydantic import AnyUrl, BeforeValidator, computed_field


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    SQLALCHEMY_DATABASE_URL: str = "postgresql://username:password@ep-dawn-hat-asy4fkao-pooler.c-4.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG API"
    SECRET_KEY: str = "changethis"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    BACKEND_CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []

    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(o).rstrip("/") for o in self.BACKEND_CORS_ORIGINS] + ["http://localhost:5173"]

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "changethis"
    POSTGRES_DB: str = "rag_db"

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # OpenAI / Embedding settings
    OPENAI_API_KEYOPENAI_API_KEY: str = "changethis"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 1024

    # RAG settings
    TOP_K_RESULTS: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # First superuser
    FIRST_SUPERUSER: str = "kocicjelena@gmail.com"
    FIRST_SUPERUSER_PASSWORD: str = "npg_8PCYuR9XclnJ"


settings = Settings()
