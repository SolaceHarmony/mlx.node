import addon from '../internal/addon';
import type { Device, DeviceLike } from './device';
import { normalizeDevice } from './device';

export type StreamLike = MLXStream | DeviceLike;

interface NativeStreamContext {
  enter(): void;
  exit(): void;
}

export class MLXStream {
  private readonly handle: any;

  private constructor(handle: any) {
    this.handle = handle;
  }

  static fromHandle(handle: any): MLXStream {
    return new MLXStream(handle);
  }

  get index(): number {
    return this.handle.index as number;
  }

  get device(): Device {
    const native = this.handle.device as Device;
    return {
      type: native.type,
      index: native.index ?? 0,
    };
  }

  equals(other: MLXStream): boolean {
    return this.handle.equals(other.toNative());
  }

  toString(): string {
    return this.handle.toString();
  }

  toNative(): any {
    return this.handle;
  }
}

export function isStream(value: unknown): value is MLXStream {
  return value instanceof MLXStream;
}

function toNativeStreamOrDevice(stream: StreamLike): any {
  if (isStream(stream)) {
    return stream.toNative();
  }
  return normalizeDevice(stream);
}

function createContext(stream: StreamLike): NativeStreamContext {
  return addon.stream(toNativeStreamOrDevice(stream)) as NativeStreamContext;
}

export function defaultStream(device?: DeviceLike | null): MLXStream {
  const arg = device == null ? undefined : normalizeDevice(device);
  const handle = arg ? addon.default_stream(arg) : addon.default_stream();
  return MLXStream.fromHandle(handle);
}

export function newStream(device?: DeviceLike | null): MLXStream {
  const arg = device == null ? undefined : normalizeDevice(device);
  const handle = arg ? addon.new_stream(arg) : addon.new_stream();
  return MLXStream.fromHandle(handle);
}

export function setDefaultStream(stream: MLXStream): void {
  addon.set_default_stream(stream.toNative());
}

export function synchronize(stream?: StreamLike | null): void {
  if (stream == null) {
    addon.synchronize();
    return;
  }
  if (!isStream(stream)) {
    const native = defaultStream(stream);
    addon.synchronize(native.toNative());
    return;
  }
  addon.synchronize(stream.toNative());
}

export function streamContext(stream: StreamLike): NativeStreamContext {
  return createContext(stream);
}

export async function withStream<T>(
  stream: StreamLike,
  fn: () => Promise<T> | T,
): Promise<T> {
  const ctx = createContext(stream);
  ctx.enter();
  let exited = false;
  const exit = () => {
    if (!exited) {
      exited = true;
      ctx.exit();
    }
  };
  try {
    const result = fn();
    if (result instanceof Promise) {
      try {
        const awaited = await result;
        exit();
        return awaited;
      } catch (error) {
        exit();
        throw error;
      }
    }
    exit();
    return result;
  } catch (error) {
    exit();
    throw error;
  }
}

export function toNativeStreamArgument(stream?: StreamLike | null): any {
  if (stream == null) {
    return undefined;
  }
  if (isStream(stream)) {
    return stream.toNative();
  }
  return normalizeDevice(stream);
}
