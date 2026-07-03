import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.corpus import router as corpus_router
from app.config import settings
from app.http_client import close_http_clients
from app.logging import configure_logging
from app.middleware.request_context import RequestContextMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await close_http_clients()


def create_app() -> FastAPI:
    configure_logging(json_logs=settings.log_json)
    app = FastAPI(title="SourceSight API", lifespan=lifespan)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(chat_router)
    app.include_router(corpus_router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
