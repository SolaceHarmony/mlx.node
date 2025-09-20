import MLXArray from '../core/array';
import { streamContext as createStreamContext } from '../core/stream';
import { frameToMessage } from './encoding';
import { tensorToFrames } from './tensors';
import type {
  SSEMessage,
  SSEOptions,
  StreamSource,
  StreamValue,
  TensorStreamConfig,
  TensorStreamFrame,
} from './types';

const encoder = new TextEncoder();

type StreamContextHandle = {
  enter(): void;
  exit(): void;
};

function serializeMessage(message: SSEMessage): Uint8Array {
  const retryLine = message.retry ? `retry: ${message.retry}\n` : '';
  const eventLine = message.event ? `event: ${message.event}\n` : '';
  const idLine = message.id ? `id: ${message.id}\n` : '';
  const payload = `${message.data}`.split('\n').map((line) => `data: ${line}\n`).join('');
  const text = `${retryLine}${eventLine}${idLine}${payload}\n`;
  return encoder.encode(text);
}

function toAsyncIterable<T>(source: StreamSource<T>): AsyncIterable<T> {
  const resolved = typeof source === 'function' ? source() : source;
  if (resolved == null) {
    throw new Error('Stream source returned null/undefined');
  }
  if (Symbol.asyncIterator in resolved) {
    return resolved as AsyncIterable<T>;
  }
  if (Symbol.iterator in resolved) {
    return (async function* iterate() {
      for (const item of resolved as Iterable<T>) {
        yield item;
      }
    })();
  }
  throw new Error('Provided stream source is not iterable');
}

const isPromise = (value: unknown): value is PromiseLike<unknown> =>
  typeof value === 'object' && value !== null && 'then' in (value as any);

const isAsyncIterable = (value: unknown): value is AsyncIterable<unknown> =>
  typeof value === 'object' && value !== null && Symbol.asyncIterator in value;

const isIterable = (value: unknown): value is Iterable<unknown> =>
  typeof value === 'object' && value !== null && Symbol.iterator in value;

function isTensorFrame(value: unknown): value is TensorStreamFrame {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    typeof (value as { type: unknown }).type === 'string'
  );
}

function isSSEMessage(value: unknown): value is SSEMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    'data' in value &&
    typeof (value as { data: unknown }).data === 'string'
  );
}

function resolveFactoryValue<T>(
  value:
      | T
      | ((tensor: MLXArray, index: number) => T)
      | undefined,
  tensor: MLXArray,
  index: number,
): T | undefined {
  if (typeof value === 'function') {
    return (value as (tensor: MLXArray, index: number) => T)(tensor, index);
  }
  return value;
}

function getHeartbeatInterval(options?: SSEOptions): number | undefined {
  const { heartbeatIntervalMs } = options ?? {};
  if (heartbeatIntervalMs === undefined) {
    return 15000;
  }
  if (heartbeatIntervalMs <= 0) {
    return undefined;
  }
  return heartbeatIntervalMs;
}

export function createEventStream(
  source: StreamSource<StreamValue>,
  options?: SSEOptions,
): ReadableStream<Uint8Array> {
  const iterable = toAsyncIterable(source);
  const retryDelayMs = options?.retryDelayMs;
  const heartbeatMs = getHeartbeatInterval(options);
  const tensorOptions: TensorStreamConfig | undefined = options?.tensor;
  const streamOption = options?.stream;

  let tensorIndex = 0;
  let streamCtx: StreamContextHandle | null = null;

  return new ReadableStream<Uint8Array>({
    start(controller) {
      let heartbeatTimer: ReturnType<typeof setInterval> | undefined;
      let stopped = false;

      const push = (message: SSEMessage) => {
        if (retryDelayMs && !message.retry) {
          message.retry = retryDelayMs;
        }
        controller.enqueue(serializeMessage(message));
      };

      const pushFrame = (frame: TensorStreamFrame) => {
        push(frameToMessage(frame));
      };

      const stop = (reason?: unknown) => {
        if (stopped) {
          return;
        }
        stopped = true;
        if (heartbeatTimer) {
          clearInterval(heartbeatTimer);
        }
        if (streamCtx) {
          streamCtx.exit();
          streamCtx = null;
        }
        if (reason instanceof Error && options?.quiet !== true) {
          // eslint-disable-next-line no-console
          console.warn('[mlx.node] SSE stream stopped:', reason.message);
        }
        controller.close();
      };

      const emitTensor = async (tensor: MLXArray) => {
        const currentIndex = tensorIndex++;
        const chunkBytes = tensorOptions?.chunkBytes;
        const metadata = tensorOptions?.metadata
          ? resolveFactoryValue(tensorOptions.metadata, tensor, currentIndex)
          : undefined;
        const endMetadata = tensorOptions?.endMetadata
          ? resolveFactoryValue(tensorOptions.endMetadata, tensor, currentIndex)
          : undefined;
        const tensorId = tensorOptions?.idFactory
          ? tensorOptions.idFactory(tensor, currentIndex)
          : undefined;

        for await (const frame of tensorToFrames(
            tensor,
            {
              tensorId,
              chunkBytes,
              metadata,
              endMetadata,
            },
            currentIndex,
        )) {
          pushFrame(frame);
        }
      };

      const emitValue = async (value: StreamValue | undefined | null): Promise<void> => {
        if (value == null) {
          return;
        }

        if (isPromise(value)) {
          await emitValue(await value);
          return;
        }

        if (value instanceof MLXArray) {
          await emitTensor(value);
          return;
        }

        if (isTensorFrame(value)) {
          pushFrame(value);
          return;
        }

        if (typeof value === 'string') {
          push({ data: value });
          return;
        }

        if (isSSEMessage(value)) {
          push(value);
          return;
        }

        if (isAsyncIterable(value)) {
          for await (const inner of value) {
            await emitValue(inner as StreamValue);
          }
          return;
        }

        if (isIterable(value)) {
          for (const inner of value) {
            // eslint-disable-next-line no-await-in-loop
            await emitValue(inner as StreamValue);
          }
          return;
        }

        throw new Error('Unsupported value provided to event stream');
      };

      const pump = async () => {
        try {
          if (retryDelayMs) {
            push({ data: '', retry: retryDelayMs });
          }
          if (streamOption && !streamCtx) {
            streamCtx = createStreamContext(streamOption);
            streamCtx.enter();
          }
          for await (const value of iterable) {
            await emitValue(value as StreamValue);
          }
          stop();
        } catch (error) {
          stop(error);
        }
      };

      pump();

      if (heartbeatMs) {
        heartbeatTimer = setInterval(() => {
          if (stopped) {
            return;
          }
          push({ event: 'heartbeat', data: JSON.stringify({ timestamp: Date.now() }) });
        }, heartbeatMs);
      }

      if (options?.signal) {
        options.signal.addEventListener(
          'abort',
          () => {
            stop(options.signal?.reason);
          },
          { once: true },
        );
      }
    },
  });
}

export function eventStreamResponse(
  source: StreamSource<StreamValue>,
  options?: SSEOptions,
): Response {
  const stream = createEventStream(source, options);
  const headers = new Headers({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });
  if (options?.headers) {
    Object.entries(options.headers).forEach(([key, value]) => {
      headers.set(key, value);
    });
  }
  return new Response(stream, { headers });
}
