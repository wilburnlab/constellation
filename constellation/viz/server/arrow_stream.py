"""Arrow IPC streaming helper for FastAPI responses.

Wraps an iterator of `pa.RecordBatch`es into a `StreamingResponse` whose
body is a complete Arrow IPC stream (schema header + zero or more record
batches + EOS). The browser-side `apache-arrow` JS package decodes this
via `tableFromIPC(response)` directly.

Invariants:

- The schema is written exactly once, at the start of the stream, even
  if the iterator yields zero batches. (The IPC stream reader on the
  client tolerates an empty stream as long as the schema header is
  present.)
- Each batch is written in its own buffer flush so the framework
  streams body bytes incrementally — no buffering the full response.
- The caller is responsible for choosing `media_type` and any custom
  headers (e.g. `X-Track-Mode`); the helper just plumbs them through.

The helper is the only place in the package that touches
`pa.ipc.RecordBatchStreamWriter`; kernel modules emit batches as
ordinary Python iterators and never see the wire format.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator

import pyarrow as pa
from fastapi.responses import StreamingResponse


ARROW_IPC_MEDIA_TYPE = "application/vnd.apache.arrow.stream"


def encode_ipc_stream(
    schema: pa.Schema, batches: Iterable[pa.RecordBatch]
) -> Iterator[bytes]:
    """Yield successive byte chunks of an Arrow IPC stream.

    First yield is the schema header; subsequent yields are individual
    record-batch frames; the terminating EOS frame is appended after
    the iterable is exhausted. Batches whose schema does not match the
    declared `schema` raise via the `RecordBatchStreamWriter`.

    We back the writer with `io.BytesIO` rather than
    `pa.BufferOutputStream` because the latter closes itself on
    `getvalue()` — which makes incremental flushing impossible. The
    `io.BytesIO` sink supports `tell()`/`getvalue()` repeatedly without
    invalidating the underlying buffer.
    """
    sink = io.BytesIO()
    writer = pa.ipc.new_stream(sink, schema)
    last_pos = 0

    def _flush() -> bytes | None:
        nonlocal last_pos
        pos = sink.tell()
        if pos == last_pos:
            return None
        sink.seek(last_pos)
        chunk = sink.read(pos - last_pos)
        sink.seek(pos)
        last_pos = pos
        return chunk

    try:
        # Schema header is written by `new_stream` itself; flush it
        # eagerly so the client can begin decoding before any batch
        # arrives.
        chunk = _flush()
        if chunk:
            yield chunk

        for batch in batches:
            writer.write_batch(batch)
            chunk = _flush()
            if chunk:
                yield chunk
    finally:
        writer.close()
        chunk = _flush()
        if chunk:
            yield chunk


def batches_to_response(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    *,
    headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Build a `StreamingResponse` carrying an Arrow IPC stream.

    Sets `Content-Type: application/vnd.apache.arrow.stream` and
    `Cache-Control: no-store` (the data depends on query parameters
    and is cheap to regenerate). Caller-supplied headers override or
    extend these — typically `X-Track-Mode: vector|hybrid` so the
    frontend can branch its renderer without inspecting the schema.
    """
    base_headers = {
        "Cache-Control": "no-store",
        # Hint to nginx-style proxies (when a future hosted version
        # exists): don't buffer the streamed body.
        "X-Accel-Buffering": "no",
    }
    if headers:
        base_headers.update(headers)
    return StreamingResponse(
        encode_ipc_stream(schema, batches),
        media_type=ARROW_IPC_MEDIA_TYPE,
        headers=base_headers,
    )


def collect_to_table(
    schema: pa.Schema, batches: Iterable[pa.RecordBatch]
) -> pa.Table:
    """Materialize an iterator of batches into a single `pa.Table`.

    Used by tests + the export endpoint, where we need the full result
    in memory anyway. Production streaming paths use
    `batches_to_response`. The schema is enforced — empty inputs return
    an empty table with the declared schema.
    """
    batch_list = list(batches)
    if not batch_list:
        return schema.empty_table()
    return pa.Table.from_batches(batch_list, schema=schema)
