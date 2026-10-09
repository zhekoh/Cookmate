"""HTTP entry point.

Run it with the factory flag, so the app is built at startup rather than at
import time:

    uv run uvicorn --factory rag.api:create_app --port 8000

``create_app`` loads the settings before it returns. A missing environment
variable therefore stops uvicorn from starting at all, which is the loud
failure we want, instead of a server that boots and then fails its first
real request.
"""

from fastapi import FastAPI

from rag.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    # Tests pass settings in directly; production reads the environment.
    settings = settings or get_settings()

    app = FastAPI(title="Cookmate RAG", version="0.1.0")
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up and serving.

        Deliberately checks nothing else. A health check that calls the
        database or a model API turns their outage into "restart this
        container", which fixes nothing and adds a restart loop. Dependency
        checks belong on a separate ``/ready`` endpoint (COO-70).
        """
        return {"status": "ok"}

    return app
