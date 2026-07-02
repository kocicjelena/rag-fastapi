from sqlmodel import Session, create_engine, SQLModel

from app.core.config import settings

engine = create_engine(settings.SQLALCHEMY_DATABASE_URL)


def init_db(session: Session) -> None:
    # Enable pgvector extension
    session.exec(  # type: ignore
        __import__("sqlmodel").text("CREATE EXTENSION IF NOT EXISTS vector")
    )
    session.commit()
    SQLModel.metadata.create_all(engine)

    from app import crud
    from app.models import UserCreate

    user = crud.get_user_by_email(session=session, email=settings.FIRST_SUPERUSER)
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        crud.create_user(session=session, user_create=user_in)


def get_session():
    with Session(engine) as session:
        yield session
