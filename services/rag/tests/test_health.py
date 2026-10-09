import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import ConfigError, Settings


def test_health_answers_ok(settings: Settings) -> None:
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_refuses_to_start_without_config() -> None:
    with pytest.raises(ConfigError):
        create_app()
