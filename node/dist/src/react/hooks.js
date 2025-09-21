"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.useTensorStream = useTensorStream;
const react_1 = require("react");
const streaming_1 = require("../streaming");
function ensureUrl(input) {
    if (typeof input === 'function') {
        const resolved = input();
        return resolved ? resolved.toString() : null;
    }
    if (!input) {
        return null;
    }
    return input.toString();
}
function parseData(event) {
    return JSON.parse(event.data);
}
function parseTensorFrame(event, decodeBinary) {
    switch (event.type) {
        case 'tensor-header': {
            const payload = parseData(event);
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
            const payload = parseData(event);
            const reportedEncoding = payload.encoding ?? 'base64';
            let encoding = reportedEncoding;
            let data = payload.chunk;
            if (decodeBinary && (reportedEncoding === 'base64' || reportedEncoding === 'binary')) {
                const buffer = (0, streaming_1.fromBase64)(payload.chunk);
                data = buffer;
                encoding = 'binary';
            }
            return {
                type: 'data',
                tensorId: payload.tensorId,
                data,
                encoding,
                sequence: payload.sequence,
                byteLength: payload.byteLength ?? (data instanceof Uint8Array ? data.byteLength : undefined),
            };
        }
        case 'tensor-end': {
            const payload = parseData(event);
            return {
                type: 'end',
                tensorId: payload.tensorId,
                metadata: payload.metadata,
            };
        }
        case 'heartbeat': {
            const payload = parseData(event);
            return {
                type: 'heartbeat',
                timestamp: payload.timestamp,
            };
        }
        case 'tensor-error': {
            const payload = parseData(event);
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
function useIsomorphicEffect(effect, deps) {
    const useAnyEffect = typeof window === 'undefined' ? react_1.useEffect : react_1.useEffect;
    // This indirection keeps the linter happy while avoiding SSR warnings.
    useAnyEffect(effect, deps);
}
function useTensorStream(urlFactory, options) {
    const autoStart = options?.autoStart ?? true;
    const keepHistory = options?.keepHistory ?? false;
    const decodeBinary = options?.decodeBinary ?? true;
    const [latest, setLatest] = (0, react_1.useState)(null);
    const [isConnected, setConnected] = (0, react_1.useState)(false);
    const [error, setError] = (0, react_1.useState)(null);
    const historyRef = (0, react_1.useRef)([]);
    const sourceRef = (0, react_1.useRef)(null);
    const url = (0, react_1.useMemo)(() => ensureUrl(urlFactory), [urlFactory]);
    const stop = (0, react_1.useCallback)(() => {
        if (sourceRef.current) {
            sourceRef.current.close();
            sourceRef.current = null;
        }
        setConnected(false);
    }, []);
    const start = (0, react_1.useCallback)(() => {
        if (sourceRef.current) {
            return;
        }
        if (!url || typeof window === 'undefined') {
            return;
        }
        const eventSource = new EventSource(url, options?.eventSourceInit);
        sourceRef.current = eventSource;
        setError(null);
        const handleFrame = (event) => {
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
        const handleError = (event) => {
            const err = new Error('EventSource connection failed');
            setError(err);
            if (options?.autoStart === false) {
                stop();
            }
        };
        const listener = handleFrame;
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
    const history = (0, react_1.useMemo)(() => (keepHistory ? historyRef.current : []), [keepHistory, latest]);
    return {
        latest,
        history,
        isConnected,
        error,
        start,
        stop,
    };
}
