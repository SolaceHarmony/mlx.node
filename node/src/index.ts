import addon from './internal/addon';

// Native addon utilities
export const native = {
  hello: (): string => addon.hello(),
};

// Re-export the core object itself
import * as coreNamespace from './core';
export const core = coreNamespace;

// Top-level re-exports using functions/getters for late binding and circular dependency safety
export const array = (...args: any[]) => coreNamespace.array(...args);
export const from_js_array = (...args: any[]) => coreNamespace.from_js_array(...args);
export const array_builder = (...args: any[]) => coreNamespace.array_builder(...args);
export const asarray = (...args: any[]) => coreNamespace.asarray(...args);
export const ArrayBuilder = coreNamespace.ArrayBuilder;
export const MLX_ARRAY_SYMBOL = coreNamespace.MLX_ARRAY_SYMBOL;
export const Array = coreNamespace.Array;
export const Stream = coreNamespace.Stream;
export const issubdtype = (...args: any[]) => coreNamespace.issubdtype(...args);
export const zeros = (...args: any[]) => coreNamespace.zeros(...args);
export const zeros_like = (...args: any[]) => coreNamespace.zeros_like(...args);
export const ones = (...args: any[]) => coreNamespace.ones(...args);
export const ones_like = (...args: any[]) => coreNamespace.ones_like(...args);
export const full = (...args: any[]) => coreNamespace.full(...args);
export const default_stream = (...args: any[]) => coreNamespace.default_stream(...args);
export const new_stream = (...args: any[]) => coreNamespace.new_stream(...args);
export const set_default_stream = (...args: any[]) => coreNamespace.set_default_stream(...args);
export const synchronize = (...args: any[]) => coreNamespace.synchronize(...args);
export const stream_context = (...args: any[]) => coreNamespace.stream_context(...args);
export const stream = (...args: any[]) => coreNamespace.stream(...args);
export const with_stream = (...args: any[]) => coreNamespace.with_stream(...args);
export const device = (...args: any[]) => coreNamespace.device(...args);
export const cpu = coreNamespace.cpu;
export const gpu = (index?: number) => coreNamespace.gpu(index);
export const reshape = (...args: any[]) => coreNamespace.reshape(...args);
export const transpose = (...args: any[]) => coreNamespace.transpose(...args);
export const moveaxis = (...args: any[]) => coreNamespace.moveaxis(...args);
export const swapaxes = (...args: any[]) => coreNamespace.swapaxes(...args);
export const add = (...args: any[]) => coreNamespace.add(...args);
export const multiply = (...args: any[]) => coreNamespace.multiply(...args);
export const subtract = (...args: any[]) => coreNamespace.subtract(...args);
export const divide = (...args: any[]) => coreNamespace.divide(...args);
export const power = (...args: any[]) => coreNamespace.power(...args);
export const equal = (...args: any[]) => coreNamespace.equal(...args);
export const not_equal = (...args: any[]) => coreNamespace.not_equal(...args);
export const less = (...args: any[]) => coreNamespace.less(...args);
export const less_equal = (...args: any[]) => coreNamespace.less_equal(...args);
export const greater = (...args: any[]) => coreNamespace.greater(...args);
export const greater_equal = (...args: any[]) => coreNamespace.greater_equal(...args);
export const maximum = (...args: any[]) => coreNamespace.maximum(...args);
export const minimum = (...args: any[]) => coreNamespace.minimum(...args);
export const where = (...args: any[]) => coreNamespace.where(...args);
export const arange = (...args: any[]) => coreNamespace.arange(...args);
export const random = coreNamespace.random;
export const tan = (...args: any[]) => coreNamespace.tan(...args);
export const sin = (...args: any[]) => coreNamespace.sin(...args);
export const cos = (...args: any[]) => coreNamespace.cos(...args);
export const arcsin = (...args: any[]) => coreNamespace.arcsin(...args);
export const arccos = (...args: any[]) => coreNamespace.arccos(...args);
export const arctan = (...args: any[]) => coreNamespace.arctan(...args);
export const arctan2 = (...args: any[]) => coreNamespace.arctan2(...args);
export const rsqrt = (...args: any[]) => coreNamespace.rsqrt(...args);
export const square = (...args: any[]) => coreNamespace.square(...args);
export const sign = (...args: any[]) => coreNamespace.sign(...args);
export const abs = (...args: any[]) => coreNamespace.abs(...args);
export const sqrt = (...args: any[]) => coreNamespace.sqrt(...args);
export const exp = (...args: any[]) => coreNamespace.exp(...args);
export const log = (...args: any[]) => coreNamespace.log(...args);
export const import_function = (...args: any[]) => coreNamespace.import_function(...args);
export const sum = (...args: any[]) => coreNamespace.sum(...args);
export const mean = (...args: any[]) => coreNamespace.mean(...args);
export const min = (...args: any[]) => coreNamespace.min(...args);
export const max = (...args: any[]) => coreNamespace.max(...args);
export const prod = (...args: any[]) => coreNamespace.prod(...args);
export const argmin = (...args: any[]) => coreNamespace.argmin(...args);
export const argmax = (...args: any[]) => coreNamespace.argmax(...args);
export const logsumexp = (...args: any[]) => coreNamespace.logsumexp(...args);
export const logcumsumexp = (...args: any[]) => coreNamespace.logcumsumexp(...args);
export const softmax = (...args: any[]) => coreNamespace.softmax(...args);
export const logaddexp = (...args: any[]) => coreNamespace.logaddexp(...args);
export const clip = (...args: any[]) => coreNamespace.clip(...args);
export const log1p = (...args: any[]) => coreNamespace.log1p(...args);
export const negative = (...args: any[]) => coreNamespace.negative(...args);
export const reciprocal = (...args: any[]) => coreNamespace.reciprocal(...args);
export const stop_gradient = (...args: any[]) => coreNamespace.stop_gradient(...args);
export const tanh = (...args: any[]) => coreNamespace.tanh(...args);
export const conjugate = (...args: any[]) => coreNamespace.conjugate(...args);
export const conj = (...args: any[]) => coreNamespace.conj(...args);
export const floor = (...args: any[]) => coreNamespace.floor(...args);
export const ceil = (...args: any[]) => coreNamespace.ceil(...args);
export const round = (...args: any[]) => coreNamespace.round(...args);
export const isnan = (...args: any[]) => coreNamespace.isnan(...args);
export const isinf = (...args: any[]) => coreNamespace.isinf(...args);
export const isposinf = (...args: any[]) => coreNamespace.isposinf(...args);
export const isneginf = (...args: any[]) => coreNamespace.isneginf(...args);
export const isfinite = (...args: any[]) => coreNamespace.isfinite(...args);
export const logical_and = (...args: any[]) => coreNamespace.logical_and(...args);
export const logical_or = (...args: any[]) => coreNamespace.logical_or(...args);
export const logical_not = (...args: any[]) => coreNamespace.logical_not(...args);
export const bitwise_and = (...args: any[]) => coreNamespace.bitwise_and(...args);
export const bitwise_or = (...args: any[]) => coreNamespace.bitwise_or(...args);
export const bitwise_xor = (...args: any[]) => coreNamespace.bitwise_xor(...args);
export const bitwise_shift_left = (...args: any[]) => coreNamespace.bitwise_shift_left(...args);
export const bitwise_shift_right = (...args: any[]) => coreNamespace.bitwise_shift_right(...args);
export const left_shift = (...args: any[]) => coreNamespace.left_shift(...args);
export const right_shift = (...args: any[]) => coreNamespace.right_shift(...args);
export const degrees = (...args: any[]) => coreNamespace.degrees(...args);
export const radians = (...args: any[]) => coreNamespace.radians(...args);
export const real = (...args: any[]) => coreNamespace.real(...args);
export const imag = (...args: any[]) => coreNamespace.imag(...args);
export const outer = (...args: any[]) => coreNamespace.outer(...args);
export const inner = (...args: any[]) => coreNamespace.inner(...args);
export const kron = (...args: any[]) => coreNamespace.kron(...args);
export const nan_to_num = (...args: any[]) => coreNamespace.nan_to_num(...args);
export const allclose = (...args: any[]) => coreNamespace.allclose(...args);
export const isclose = (...args: any[]) => coreNamespace.isclose(...args);
export const view = (...args: any[]) => coreNamespace.view(...args);
export const contiguous = (...args: any[]) => coreNamespace.contiguous(...args);
export const hadamard_transform = (...args: any[]) => coreNamespace.hadamard_transform(...args);
export const unflatten = (...args: any[]) => coreNamespace.unflatten(...args);
export const partition = (...args: any[]) => coreNamespace.partition(...args);
export const argpartition = (...args: any[]) => coreNamespace.argpartition(...args);
export const put_along_axis = (...args: any[]) => coreNamespace.put_along_axis(...args);
export const roll = (...args: any[]) => coreNamespace.roll(...args);
export const tri = (...args: any[]) => coreNamespace.tri(...args);
export const meshgrid = (...args: any[]) => coreNamespace.meshgrid(...args);
export const broadcast_arrays = (...args: any[]) => coreNamespace.broadcast_arrays(...args);
export const atleast_1d = (...args: any[]) => coreNamespace.atleast_1d(...args);
export const atleast_2d = (...args: any[]) => coreNamespace.atleast_2d(...args);
export const atleast_3d = (...args: any[]) => coreNamespace.atleast_3d(...args);
export const slice_update = (...args: any[]) => coreNamespace.slice_update(...args);
export const conv_general = (...args: any[]) => coreNamespace.conv_general(...args);
export const conv_transpose1d = (...args: any[]) => coreNamespace.conv_transpose1d(...args);
export const conv_transpose2d = (...args: any[]) => coreNamespace.conv_transpose2d(...args);
export const conv_transpose3d = (...args: any[]) => coreNamespace.conv_transpose3d(...args);
export const einsum = (...args: any[]) => coreNamespace.einsum(...args);
export const tensordot = (...args: any[]) => coreNamespace.tensordot(...args);
export const block_masked_mm = (...args: any[]) => coreNamespace.block_masked_mm(...args);
export const gather_mm = (...args: any[]) => coreNamespace.gather_mm(...args);
export const segmented_mm = (...args: any[]) => coreNamespace.segmented_mm(...args);
export const quantize = (...args: any[]) => coreNamespace.quantize(...args);
export const dequantize = (...args: any[]) => coreNamespace.dequantize(...args);
export const quantized_matmul = (...args: any[]) => coreNamespace.quantized_matmul(...args);
export const gather_qmm = (...args: any[]) => coreNamespace.gather_qmm(...args);
export const concat = (...args: any[]) => coreNamespace.concat(...args);
export const divmod = (...args: any[]) => coreNamespace.divmod(...args);
export const permute_dims = (...args: any[]) => coreNamespace.permute_dims(...args);
export const trunc = (...args: any[]) => coreNamespace.trunc(...args);
export const broadcast_shapes = (...args: any[]) => coreNamespace.broadcast_shapes(...args);
export const convolve = (...args: any[]) => coreNamespace.convolve(...args);
export const einsum_path = (...args: any[]) => coreNamespace.einsum_path(...args);
export const eval_op = (...args: any[]) => coreNamespace.eval_op(...args);
export const async_eval = (...args: any[]) => coreNamespace.async_eval(...args);
export const load = (...args: any[]) => coreNamespace.load(...args);
export const load_safetensors = (...args: any[]) => coreNamespace.load_safetensors(...args);
export const load_gguf = (...args: any[]) => coreNamespace.load_gguf(...args);
export const save = (...args: any[]) => coreNamespace.save(...args);
export const save_safetensors = (...args: any[]) => coreNamespace.save_safetensors(...args);
export const save_gguf = (...args: any[]) => coreNamespace.save_gguf(...args);
export const grad = (...args: any[]) => coreNamespace.grad(...args);
export const value_and_grad = (...args: any[]) => coreNamespace.value_and_grad(...args);
export const vjp = (...args: any[]) => coreNamespace.vjp(...args);
export const jvp = (...args: any[]) => coreNamespace.jvp(...args);
export const vmap = (...args: any[]) => coreNamespace.vmap(...args);
export const compile = (...args: any[]) => coreNamespace.compile(...args);
export const enable_compile = (...args: any[]) => coreNamespace.enable_compile(...args);
export const disable_compile = (...args: any[]) => coreNamespace.disable_compile(...args);
export const checkpoint = (...args: any[]) => coreNamespace.checkpoint(...args);
export const export_function = (...args: any[]) => coreNamespace.export_function(...args);
export const export_to_dot = (...args: any[]) => coreNamespace.export_to_dot(...args);

// Missing ones from my previous comm analysis
export const addmm = (...args: any[]) => coreNamespace.addmm(...args);
export const all = (...args: any[]) => coreNamespace.all(...args);
export const any = (...args: any[]) => coreNamespace.any(...args);
export const arccosh = (...args: any[]) => coreNamespace.arccosh(...args);
export const arcsinh = (...args: any[]) => coreNamespace.arcsinh(...args);
export const arctanh = (...args: any[]) => coreNamespace.arctanh(...args);
export const argsort = (...args: any[]) => coreNamespace.argsort(...args);
export const array_equal = (...args: any[]) => coreNamespace.array_equal(...args);
export const as_strided = (...args: any[]) => coreNamespace.as_strided(...args);
export const bitwise_invert = (...args: any[]) => coreNamespace.bitwise_invert(...args);
export const broadcast_to = (...args: any[]) => coreNamespace.broadcast_to(...args);
export const concatenate = (...args: any[]) => coreNamespace.concatenate(...args);
export const conv1d = (...args: any[]) => coreNamespace.conv1d(...args);
export const conv2d = (...args: any[]) => coreNamespace.conv2d(...args);
export const conv3d = (...args: any[]) => coreNamespace.conv3d(...args);
export const cosh = (...args: any[]) => coreNamespace.cosh(...args);
export const cummax = (...args: any[]) => coreNamespace.cummax(...args);
export const cummin = (...args: any[]) => coreNamespace.cummin(...args);
export const cumprod = (...args: any[]) => coreNamespace.cumprod(...args);
export const cumsum = (...args: any[]) => coreNamespace.cumsum(...args);
export const diag = (...args: any[]) => coreNamespace.diag(...args);
export const diagonal = (...args: any[]) => coreNamespace.diagonal(...args);
export const erf = (...args: any[]) => coreNamespace.erf(...args);
export const erfinv = (...args: any[]) => coreNamespace.erfinv(...args);
export const expand_dims = (...args: any[]) => coreNamespace.expand_dims(...args);
export const expm1 = (...args: any[]) => coreNamespace.expm1(...args);
export const eye = (...args: any[]) => coreNamespace.eye(...args);
export const fast = coreNamespace.fast;
export const fft = coreNamespace.fft;
export const flatten = (...args: any[]) => coreNamespace.flatten(...args);
export const floor_divide = (...args: any[]) => coreNamespace.floor_divide(...args);
export const identity = (...args: any[]) => coreNamespace.identity(...args);
export const linalg = coreNamespace.linalg;
export const linspace = (...args: any[]) => coreNamespace.linspace(...args);
export const matmul = (...args: any[]) => coreNamespace.matmul(...args);
export const number_of_elements = (...args: any[]) => coreNamespace.number_of_elements(...args);
export const pad = (...args: any[]) => coreNamespace.pad(...args);
export const remainder = (...args: any[]) => coreNamespace.remainder(...args);
export const repeat = (...args: any[]) => coreNamespace.repeat(...args);
export const sigmoid = (...args: any[]) => coreNamespace.sigmoid(...args);
export const sinh = (...args: any[]) => coreNamespace.sinh(...args);
export const slice = (...args: any[]) => coreNamespace.slice(...args);
export const sort = (...args: any[]) => coreNamespace.sort(...args);
export const split = (...args: any[]) => coreNamespace.split(...args);
export const squeeze = (...args: any[]) => coreNamespace.squeeze(...args);
export const stack = (...args: any[]) => coreNamespace.stack(...args);
export const std = (...args: any[]) => coreNamespace.std(...args);
export const take = (...args: any[]) => coreNamespace.take(...args);
export const take_along_axis = (...args: any[]) => coreNamespace.take_along_axis(...args);
export const tile = (...args: any[]) => coreNamespace.tile(...args);
export const topk = (...args: any[]) => coreNamespace.topk(...args);
export const trace = (...args: any[]) => coreNamespace.trace(...args);
export const tril = (...args: any[]) => coreNamespace.tril(...args);
export const triu = (...args: any[]) => coreNamespace.triu(...args);
export const variance = (...args: any[]) => coreNamespace.variance(...args);

// DType constants
export const bool = coreNamespace.bool;
export const int8 = coreNamespace.int8;
export const int16 = coreNamespace.int16;
export const int32 = coreNamespace.int32;
export const int64 = coreNamespace.int64;
export const uint8 = coreNamespace.uint8;
export const uint16 = coreNamespace.uint16;
export const uint32 = coreNamespace.uint32;
export const uint64 = coreNamespace.uint64;
export const float16 = coreNamespace.float16;
export const bfloat16 = coreNamespace.bfloat16;
export const float32 = coreNamespace.float32;
export const float64 = coreNamespace.float64;
export const complex64 = coreNamespace.complex64;
export const complexfloating = coreNamespace.complexfloating;
export const floating = coreNamespace.floating;
export const inexact = coreNamespace.inexact;
export const signedinteger = coreNamespace.signedinteger;
export const unsignedinteger = coreNamespace.unsignedinteger;
export const integer = coreNamespace.integer;
export const number = coreNamespace.number;
export const generic = coreNamespace.generic;
export const dtype = coreNamespace.dtype;
export const Dtype = coreNamespace.Dtype;
export const dtypeFromString = coreNamespace.dtypeFromString;
export const dtypeKeys = coreNamespace.dtypeKeys;
export const dtypeValues = coreNamespace.dtypeValues;
export const dtypeItems = coreNamespace.dtypeItems;
export const dtypeDir = coreNamespace.dtypeDir;
export const dtypeHas = coreNamespace.dtypeHas;
export const dtypeGet = coreNamespace.dtypeGet;
export const dtypeCategoryKeys = coreNamespace.dtypeCategoryKeys;
export const dtypeCategoryValues = coreNamespace.dtypeCategoryValues;
export const dtypeCategoryItems = coreNamespace.dtypeCategoryItems;

// Export other namespaces
import * as streaming from './streaming';
import * as optimizers from './optimizers';
import * as nn_init from './nn_init';
import * as nn from './nn';
import * as utils from './utils';

export { streaming, optimizers, nn_init, nn, utils };

// React hooks are optional
let react: any;
try {
  react = require('./react');
} catch (_) {
  react = {};
}
export { react };

// Re-export common tree and store utilities at top level for parity
export {
  tree_map,
  tree_map_with_path,
  tree_flatten,
  tree_unflatten,
  tree_reduce,
  tree_merge,
  Store,
} from './utils';
