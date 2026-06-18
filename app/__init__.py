"""DiffusionGemma demo backend package.

A small FastAPI application that streams a *block-diffusion* text-generation
visualisation over a WebSocket.  It can run in two interchangeable modes that
share one streaming contract:

* ``real`` -- drives the actual ``google/diffusiongemma-26B-A4B-it`` model via
  :mod:`app.engine` (requires a GPU, ``torch`` and ``transformers``).
* ``mock`` -- a GPU-free simulation (:mod:`app.mock_engine`) that produces the
  same WebSocket frames so the UI can be developed on a laptop.

Heavy machine-learning imports (``torch`` / ``transformers``) live *only* inside
:mod:`app.engine` real code paths, so importing this package -- and running the
server in mock mode -- needs nothing more than ``fastapi``, ``uvicorn`` and
``pydantic``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
