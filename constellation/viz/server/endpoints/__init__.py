"""FastAPI endpoint modules for the viz layer.

Each module exposes a `router: APIRouter` mounted by `server.app.create_app`.
Endpoints are intentionally thin — they parse query params, dispatch to
the registered kernel, and stream Arrow IPC batches via
`server.arrow_stream.batches_to_response`.
"""
