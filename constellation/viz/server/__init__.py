"""FastAPI server + session discovery for the viz layer.

`constellation.viz.server.app:create_app` returns the configured FastAPI
instance the CLI mounts under uvicorn. Endpoints under `endpoints/` are
thin: they parse query parameters into `TrackQuery`, dispatch to the
registered kernel via `tracks.get_kernel`, and stream Arrow IPC batches
through `arrow_stream.batches_to_response`.

The server is read-only over parquet — no compute initiation, no manifest
mutation. PR 2 introduces the command runner under
`constellation.viz.runner` for the dashboard's CLI-wrapping subprocess
streaming; the viz endpoints in PR 1 know nothing about it.
"""
