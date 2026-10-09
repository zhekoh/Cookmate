import pytest

from rag.config import ConfigError, get_settings
from tests.conftest import FAKE_ENV


@pytest.mark.usefixtures("fake_env")
def test_reads_all_settings_from_the_environment() -> None:
    settings = get_settings()

    assert settings.database_url.get_secret_value() == FAKE_ENV["DATABASE_URL"]
    assert settings.rag_service_token.get_secret_value() == FAKE_ENV["RAG_SERVICE_TOKEN"]


def test_missing_variables_fail_loudly_and_name_every_one() -> None:
    with pytest.raises(ConfigError) as excinfo:
        get_settings()

    for name in FAKE_ENV:
        assert name in str(excinfo.value)


def test_one_missing_variable_is_still_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in FAKE_ENV.items():
        if name != "VOYAGE_API_KEY":
            monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
        get_settings()


def test_an_empty_value_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Copying .env.example verbatim leaves the token as `RAG_SERVICE_TOKEN=`.
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RAG_SERVICE_TOKEN", "")

    with pytest.raises(ConfigError, match="RAG_SERVICE_TOKEN"):
        get_settings()


def test_config_error_never_contains_the_values_that_were_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression this guards: pydantic's own error prints its input, so
    # one missing variable used to put every other key in the crash log.
    for name, value in FAKE_ENV.items():
        if name != "RAG_SERVICE_TOKEN":
            monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    for name, value in FAKE_ENV.items():
        assert value not in message, f"{name} leaked into the error message"
    # And it is not hiding in a chained exception either.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


@pytest.mark.usefixtures("fake_env")
def test_secrets_are_masked_when_printed() -> None:
    printed = repr(get_settings())

    for name in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "RAG_SERVICE_TOKEN", "DATABASE_URL"):
        assert FAKE_ENV[name] not in printed
