from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.models import ChunkResult, QueryRequest, QueryResponse
from app.services import rag

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
def query_documents(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    query_in: QueryRequest,
) -> Any:
    """
    Ask a question. Returns an LLM-generated answer grounded in your documents.
    Optionally restrict search to specific document IDs.
    """
    # Validate document ownership if IDs supplied
    if query_in.document_ids:
        for doc_id in query_in.document_ids:
            doc = crud.get_document(session=session, doc_id=doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            if doc.owner_id != current_user.id and not current_user.is_superuser:
                raise HTTPException(status_code=403, detail=f"Access denied for document {doc_id}")

    # Embed query
    query_embedding = rag.embed_query(query_in.question)

    # Retrieve similar chunks
    results = crud.similarity_search(
        session=session,
        query_embedding=query_embedding,
        top_k=query_in.top_k,
        document_ids=query_in.document_ids,
    )

    if not results:
        return QueryResponse(
            question=query_in.question,
            answer="No relevant documents found to answer your question.",
            sources=[],
        )

    # Build context and generate answer
    context = rag.build_context([chunk.content for chunk, _ in results])
    answer = rag.generate_answer(question=query_in.question, context=context)

    sources = []
    for chunk, score in results:
        doc = crud.get_document(session=session, doc_id=chunk.document_id)
        sources.append(
            ChunkResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=doc.title if doc else "Unknown",
                content=chunk.content,
                score=round(1 - score, 4),  # convert cosine distance → similarity
            )
        )

    return QueryResponse(question=query_in.question, answer=answer, sources=sources)


@router.post("/stream")
def query_documents_stream(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    query_in: QueryRequest,
) -> StreamingResponse:
    """
    Same as POST /query but streams the LLM answer token-by-token using SSE.
    """
    if query_in.document_ids:
        for doc_id in query_in.document_ids:
            doc = crud.get_document(session=session, doc_id=doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            if doc.owner_id != current_user.id and not current_user.is_superuser:
                raise HTTPException(status_code=403, detail=f"Access denied for document {doc_id}")

    query_embedding = rag.embed_query(query_in.question)
    results = crud.similarity_search(
        session=session,
        query_embedding=query_embedding,
        top_k=query_in.top_k,
        document_ids=query_in.document_ids,
    )

    if not results:
        async def no_results():
            yield "data: No relevant documents found.\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    context = rag.build_context([chunk.content for chunk, _ in results])

    def event_stream():
        for token in rag.generate_answer_stream(question=query_in.question, context=context):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
