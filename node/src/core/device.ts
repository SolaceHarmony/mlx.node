export type DeviceType = 'cpu' | 'gpu';

export interface Device {
  type: DeviceType;
  index: number;
}

export type DeviceLike = Device | DeviceType | { type: DeviceType; index?: number };

export const cpu: Device = Object.freeze({ type: 'cpu' as const, index: 0 });
export const gpu = (index = 0): Device => ({ type: 'gpu', index });

export function normalizeDevice(device?: DeviceLike | null): Device {
  if (!device) {
    return { ...cpu };
  }
  if (typeof device === 'string') {
    return { type: device, index: 0 };
  }
  if ('type' in device) {
    return {
      type: device.type,
      index: device.index ?? 0,
    };
  }
  throw new TypeError('Unsupported device specification');
}

export function device(type: DeviceType, index = 0): Device {
  return { type, index };
}

