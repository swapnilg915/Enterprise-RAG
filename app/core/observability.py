from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from app.core.config import Settings


class Observability:
    def __init__(self, settings: Settings) -> None:
        self.enabled = (
            settings.langfuse_enabled
            and bool(settings.langfuse_public_key)
            and bool(settings.langfuse_secret_key)
        )
        self.client: Any | None = None
        if not self.enabled:
            return
        try:
            from langfuse import get_client

            self.client = get_client()
        except Exception:
            self.enabled = False

    def span(
        self,
        name: str,
        *,
        input_data: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AbstractContextManager[Any]:
        if not self.enabled or self.client is None:
            return nullcontext()
        return self.client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_data,
            metadata=metadata,
        )

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            self.client.flush()
