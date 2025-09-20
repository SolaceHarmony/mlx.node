Node/React Streaming Architecture
=================================

This document captures the architecture for delivering MLX workloads to React
19 and Next.js 14+ applications. It focuses on streaming-first patterns that
keep model execution on the server while progressively hydrating UI on the
client.

Goals
-----

* Reuse the MLX core library from the Node addon while presenting a React-
  friendly API surface.
* Stream intermediate tensor values, token batches, and metrics so React can
  render meaningful feedback before a full inference completes.
* Stay compatible with React 19 features such as Server Actions, Suspense, and
  the updated streaming primitives.

React 19 Capabilities
---------------------

* **Server Actions** let client components invoke MLX work on the server without
  exposing credentials or model files to the browser.
* **Suspense-first streaming SSR** allows the shell of a page to render while
  MLX-driven components stream their data. We can wrap each MLX widget in a
  Suspense boundary and progressively reveal results.
* **`react-dom/static` prerendering** provides a way to stage pages that depend
  on heavyweight MLX initialisation yet still ship their HTML incrementally.

Streaming Transport
-------------------

* Primary channel: **Server-Sent Events (SSE)** using the
  `text/event-stream` content type. It is lightweight, works with plain HTTP,
  and integrates with React/Next.js route handlers.
* Alternative channel: **ReadableStream** responses (or Vercel `streamUI`) for
  hosting environments where chunked JSON is preferable or extra framing is
  required.
* Each stream begins with a metadata frame (`tensorId`, shape, dtype, stride)
  followed by data frames carrying base64 or binary payloads. Heartbeats keep
  intermediates flowing and detect dropped clients.
* Error and completion frames are mandatory so React wrappers can close
  connections and allow MLX jobs to clean up GPU/CPU resources.

Client Integration
------------------

* Server Components open the SSE connection during render, handing an async
  iterator to child Client Components.
* Client Components subscribe via `EventSource`, reconstruct typed arrays from
  the streamed chunks, and stage them in React state with `useTransition` or
  `useDeferredValue` to keep the UI responsive.
* Hooks such as `useActionState`, `useFormStatus`, and `useOptimistic` expose
  loading and error feedback that mirrors the MLX job lifecycle.
* Provide a helper that maps streamed tensor metadata to `SharedArrayBuffer`
  allocation so large payloads avoid additional copies.

Operational Concerns
--------------------

* Run MLX work inside abortable contexts. When the HTTP connection closes,
  cancel the outstanding computation and release arrays.
* Rate-limit and back-pressure the stream writer to avoid overwhelming the
  client, especially on large tensor dumps.
* Log stream lifecycle events (open, heartbeat, close, error) for observability
  and to detect degraded inference behaviour.
* Keep the native addon context-aware: new wrappers should tuck their
  `Napi::FunctionReference`s into the shared ``AddonData`` instance so each
  Node environment gets its own constructors and cleanup hooks.

Next Steps
----------

1. Implement reusable Node helpers for building SSE responses from async
   iterators of MLX tensors.
2. Deliver React 19 hooks and components that subscribe to those streams and
   surface progressive results.
3. Document deployment recipes for Vercel, self-hosted Next.js, and plain Node
   servers, highlighting any platform-specific headers or buffering tweaks.
4. Add an end-to-end example that performs token streaming from an MLX GGUF
   model once the zero-copy constructors are available.

Library Support
---------------

The ``mlx-node`` package now exposes streaming helpers under ``streaming`` and
React-facing utilities under ``react``:

* ``streaming.eventStreamResponse`` converts an ``AsyncIterable`` of tensor
  frames into an SSE ``Response`` suitable for route handlers or Server
  Actions.
* ``streaming.tensorToFrames`` and ``streaming.tensorsToFrameStream`` transform
  real ``MLXArray`` instances into header/data/end frames, chunking payloads in
  base64 so they can be streamed without pre-serialising everything in memory.
* ``core.stream`` mirrors MLX's native stream API (``defaultStream``,
  ``newStream``, ``withStream``) and ``SSEOptions.stream`` lets you run the
  entire pipeline inside an explicit MLX stream or device context without
  additional boilerplate.
* ``react.createStreamHandler`` wraps a producer into a handler you can export
  directly from a Next.js route file.
* ``react.useTensorStream`` is a Client Component hook that subscribes to the
  SSE endpoint, tracks the latest frame, and optionally keeps a history buffer
  while invoking an ``onFrame`` callback for custom processing.

The ``streaming`` helpers automatically detect ``MLXArray`` payloads yielded
from your async generators, encode them with chunk sizes that you can override,
and honour optional metadata/id factories defined via ``SSEOptions.tensor``.
On the client, ``useTensorStream`` decodes binary chunks by default, handing
you ``Uint8Array`` views ready for typed-array reshaping.
