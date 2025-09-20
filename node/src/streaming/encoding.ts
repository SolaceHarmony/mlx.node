import type { TensorDataFrame, TensorStreamFrame, SSEMessage } from './types';

function toUint8Array(input: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  return new Uint8Array(
    input.buffer,
    input.byteOffset,
    input.byteLength,
  );
}

export function toBase64(input: ArrayBuffer | ArrayBufferView): string {
  const view = toUint8Array(input);
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(view).toString('base64');
  }
  let binary = '';
  for (let i = 0; i < view.byteLength; i += 1) {
    binary += String.fromCharCode(view[i]);
  }
  if (typeof btoa !== 'undefined') {
    return btoa(binary);
  }
  throw new Error('Base64 encoding is not supported in this environment');
}

export function fromBase64(input: string): Uint8Array {
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(input, 'base64'));
  }
  if (typeof atob !== 'undefined') {
    const binary = atob(input);
    const buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      buffer[i] = binary.charCodeAt(i);
    }
    return buffer;
  }
  throw new Error('Base64 decoding is not supported in this environment');
}

function encodeTensorData(frame: TensorDataFrame): string {
  const { data, encoding } = frame;
  if (typeof data === 'string') {
    return data;
  }
  if (encoding === 'binary') {
    return toBase64(data);
  }
  if (encoding === 'json') {
    const view = toUint8Array(data);
    const decoder = new TextDecoder();
    return decoder.decode(view);
  }
  return toBase64(data);
}

export function frameToMessage(frame: TensorStreamFrame): SSEMessage {
  switch (frame.type) {
    case 'header':
      return {
        event: 'tensor-header',
        id: frame.tensorId,
        data: JSON.stringify({
          tensorId: frame.tensorId,
          shape: frame.shape,
          dtype: frame.dtype,
          strides: frame.strides,
          metadata: frame.metadata,
        }),
      };
    case 'data':
      return {
        event: 'tensor-chunk',
        id: frame.tensorId,
        data: JSON.stringify({
          tensorId: frame.tensorId,
          chunk: encodeTensorData(frame),
          encoding: frame.encoding ?? 'base64',
          sequence: frame.sequence,
          byteLength: frame.byteLength,
        }),
      };
    case 'end':
      return {
        event: 'tensor-end',
        id: frame.tensorId,
        data: JSON.stringify({
          tensorId: frame.tensorId,
          metadata: frame.metadata,
        }),
      };
    case 'heartbeat':
      return {
        event: 'heartbeat',
        data: JSON.stringify({ timestamp: frame.timestamp }),
      };
    case 'error':
      return {
        event: 'tensor-error',
        id: frame.tensorId,
        data: JSON.stringify({
          message: frame.message,
          tensorId: frame.tensorId,
          recoverable: frame.recoverable,
          metadata: frame.metadata,
        }),
      };
    default: {
      const exhaustive: never = frame;
      throw new Error(`Unknown tensor stream frame ${(exhaustive as any)?.type}`);
    }
  }
}
