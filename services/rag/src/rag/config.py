"""Configuration, read from the environment and validated once at startup.

Every setting the service needs is declared here, and nothing else in the
package reads ``os.environ``. That gives one place to look, and it means a
missing variable fails the boot with a clear message instead of failing the
first request that happens to need it, possibly hours later.

Secrets are typed ``SecretStr``. Printing or logging the settings object shows
``'**********'`` instead of the key, so a stray ``print(settings)`` or an
exception that includes it cannot leak a credential. Code that genuinely needs
the value calls ``.get_secret_value()``, which makes each use easy to find.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """The environment is missing or has invalid settings.

    Raised instead of pydantic's ``ValidationError`` because that error prints
    the input it received. With one variable missing, the input is every
    *other* variable, so a crash log would contain the API keys in plain text.
    This error names the variables and never their values.
    """


# A secret that must have a value. Present-but-empty is rejected too: copying
# .env.example as-is leaves `RAG_SERVICE_TOKEN=` blank, and a blank shared
# secret would let any caller in by sending an empty one.
RequiredSecret = Annotated[SecretStr, Field(min_length=1)]


class DatabaseSettings(BaseSettings):
    """Just enough to reach the database.

    Separate from ``Settings`` so tools that only touch the database, like the
    migration runner, do not demand API keys they never use.
    """

    model_config = SettingsConfigDict(
        # Values come from the process environment. Docker Compose and Fly both
        # inject them there; a local .env is loaded only for convenience and is
        # git-ignored. Both paths are relative to where the command runs, so the
        # repo's .env is found from the repo root or from services/rag. Neither
        # exists inside the container or the tests' temporary directory.
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        # The repo's .env also holds the Node API's variables. Ignore them
        # rather than rejecting the whole file.
        extra="ignore",
    )

    # Postgres with pgvector: chunks, vectors and metadata.
    database_url: RequiredSecret


class Settings(DatabaseSettings):
    """Everything the HTTP service needs."""

    # Generation, routing and the support check.
    anthropic_api_key: RequiredSecret
    # Embeddings and re-ranking.
    voyage_api_key: RequiredSecret
    # Shared secret the Cookmate API sends on every call. The service has no
    # users of its own, so this is its whole authentication story.
    rag_service_token: RequiredSecret

    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    """Build the full settings once and reuse them."""
    return _load(Settings)


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Build only the database settings once and reuse them."""
    return _load(DatabaseSettings)


def _load[T: BaseSettings](cls: type[T]) -> T:
    """Read ``cls`` from the environment.

    Raises ``ConfigError`` listing every bad variable, so one failed boot
    tells you everything that is wrong, not just the first thing.
    """
    try:
        return cls()
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        ]
        # `from None` drops the original exception from the traceback. Chaining
        # it would print the ValidationError, and the secrets with it.
        raise ConfigError(
            "RAG service configuration is invalid:\n  " + "\n  ".join(problems)
        ) from None
