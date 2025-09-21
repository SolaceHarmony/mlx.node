export type DeviceType = 'cpu' | 'gpu';
export interface Device {
    type: DeviceType;
    index: number;
}
export type DeviceLike = Device | DeviceType | {
    type: DeviceType;
    index?: number;
};
export declare const cpu: Device;
export declare const gpu: (index?: number) => Device;
export declare function normalizeDevice(device?: DeviceLike | null): Device;
export declare function device(type: DeviceType, index?: number): Device;
