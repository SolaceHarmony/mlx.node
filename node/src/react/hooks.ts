import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { TensorDType, TensorStreamFrame } from '../streaming';
import { fromBase64 } from '../streaming';

export interface UseEventStreamState<T> {
  latest: T | null;
  isConnected: boolean;
  error: Error | null;
  start: () => void;
  stop: () => void;
}

export interface UseEventStreamOptions {
  autoStart?: boolean;
  eventSourceInit?: EventSourceInit;
}

export interface UseTensorStreamOptions extends UseEventStreamOptions {
  keepHistory?: boolean;
  onFrame?: (frame: TensorStreamFrame) => void;
  decodeBinary?: boolean;
}

export interface UseTensorStreamState
  extends UseEventStreamState<TensorStreamFrame> {
  history: readonly TensorStreamFrame[];
}

function ensureUrl(input: string | URL | (() => string | URL | null) | null): string | null {
  if (typeof input === 'function') {
    const resolved = input();
    return resolved ? resolved.toString() : null;
  }
  if (!input) {
    return null;
  }
  return input.toString();
}

function parseData<T>(event: MessageEvent<string>): T {
  return JSON.parse(event.data) as T;
}

function parseTensorFrame(
  event: MessageEvent<string>,
  decodeBinary: boolean,
): TensorStreamFrame | null {
  switch (event.type) {
    case 'tensor-header': {
      const payload = parseData<{
        tensorId: string;
        shape: number[];
        dtype: TensorDType;
        strides?: number[];
        metadata?: Record<string, unknown>;
      }>(event);
      return {
        type: 'header',
        tensorId: payload.tensorId,
        shape: payload.shape,
        dtype: payload.dtype,
        strides: payload.strides,
        metadata: payload.metadata,
      };
    }
    case 'tensor-chunk': {
      const payload = parseData<{
        tensorId: string;
        chunk: string;
        encoding?: 'base64' | 'binary' | 'json';
        sequence?: number;
        byteLength?: number;
      }>(event);
      const reportedEncoding = payload.encoding ?? 'base64';
      let encoding = reportedEncoding;
      let data: string | Uint8Array = payload.chunk;
      if (decodeBinary && (reportedEncoding === 'base64' || reportedEncoding === 'binary')) {
        const buffer = fromBase64(payload.chunk);
        data = buffer;
        encoding = 'binary';
      }
      return {
        type: 'data',
        tensorId: payload.tensorId,
        data,
        encoding,
        sequence: payload.sequence,
        byteLength:
          payload.byteLength ?? (data instanceof Uint8Array ? data.byteLength : undefined),
      };
    }
    case 'tensor-end': {
      const payload = parseData<{
        tensorId: string;
        metadata?: Record<string, unknown>;
      }>(event);
      return {
        type: 'end',
        tensorId: payload.tensorId,
        metadata: payload.metadata,
      };
    }
    case 'heartbeat': {
      const payload = parseData<{ timestamp: number }>(event);
      return {
        type: 'heartbeat',
        timestamp: payload.timestamp,
      };
    }
    case 'tensor-error': {
      const payload = parseData<{
        message: string;
        tensorId?: string;
        recoverable?: boolean;
        metadata?: Record<string, unknown>;
      }>(event);
      return {
        type: 'error',
        message: payload.message,
        tensorId: payload.tensorId,
        recoverable: payload.recoverable,
        metadata: payload.metadata,
      };
    }
    default:
      return null;
  }
}

function useIsomorphicEffect(effect: () => void | (() => void), deps: unknown[]) {
  const useAnyEffect = typeof window === 'undefined' ? useEffect : useEffect;
  // This indirection keeps the linter happy while avoiding SSR warnings.
  useAnyEffect(effect, deps);
}

export function useTensorStream(
  urlFactory: string | URL | (() => string | URL | null) | null,
  options?: UseTensorStreamOptions,
): UseTensorStreamState {
  const autoStart = options?.autoStart ?? true;
  const keepHistory = options?.keepHistory ?? false;
  const decodeBinary = options?.decodeBinary ?? true;
  const [latest, setLatest] = useState<TensorStreamFrame | null>(null);
  const [isConnected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const historyRef = useRef<TensorStreamFrame[]>([]);
  const sourceRef = useRef<EventSource | null>(null);
  const url = useMemo(() => ensureUrl(urlFactory), [urlFactory]);

  const stop = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setConnected(false);
  }, []);

  const start = useCallback(() => {
    if (sourceRef.current) {
      return;
    }
    if (!url || typeof window === 'undefined') {
      return;
    }
    const eventSource = new EventSource(url, options?.eventSourceInit);
    sourceRef.current = eventSource;
    setError(null);

    const handleFrame = (event: MessageEvent<string>) => {
      const frame = parseTensorFrame(event, decodeBinary);
      if (!frame) {
        return;
      }
      setLatest(frame);
      if (keepHistory) {
        historyRef.current = [...historyRef.current, frame];
      }
      options?.onFrame?.(frame);
      if (frame.type === 'error') {
        setError(new Error(frame.message));
      }
    };

    const handleError = (event: Event) => {
      const err = new Error('EventSource connection failed');
      setError(err);
      if (options?.autoStart === false) {
        stop();
      }
    };

    const listener = handleFrame as EventListener;
    eventSource.addEventListener('tensor-header', listener);
    eventSource.addEventListener('tensor-chunk', listener);
    eventSource.addEventListener('tensor-end', listener);
    eventSource.addEventListener('heartbeat', listener);
    eventSource.addEventListener('tensor-error', listener);
    eventSource.onerror = handleError;
    eventSource.onopen = () => {
      setConnected(true);
    };
  }, [keepHistory, options, stop, url]);

  useIsomorphicEffect(() => {
    if (autoStart) {
      start();
    }
    return stop;
  }, [autoStart, start, stop]);

  const history = useMemo(
    () => (keepHistory ? historyRef.current : []),
    [keepHistory, latest],
  );

  return {
    latest,
    history,
    isConnected,
    error,
    start,
    stop,
  };
}
