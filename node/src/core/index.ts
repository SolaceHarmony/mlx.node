import * as arrayModule from './array';
import * as dtypeModule from './dtype';
import * as deviceModule from './device';
import * as streamModule from './stream';
import * as opsModule from './ops';

// Use getters to avoid capturing undefined during circular initialization
export const core = {
  get array() { return arrayModule.array; },
  get from_js_array() { return arrayModule.from_js_array; },
  get array_builder() { return arrayModule.array_builder; },
  get asarray() { return arrayModule.asarray; },
  get ArrayBuilder() { return arrayModule.ArrayBuilder; },
  get MLX_ARRAY_SYMBOL() { return arrayModule.MLX_ARRAY_SYMBOL; },
  get Array() { return arrayModule.MLXArray; },
  get Stream() { return streamModule.MLXStream; },
  get issubdtype() { return dtypeModule.issubdtype; },
  get zeros() { return arrayModule.zeros; },
  get zeros_like() { return arrayModule.zeros_like; },
  get ones() { return arrayModule.ones; },
  get ones_like() { return arrayModule.ones_like; },
  get full() { return arrayModule.full; },
  get default_stream() { return streamModule.default_stream; },
  get new_stream() { return streamModule.new_stream; },
  get set_default_stream() { return streamModule.set_default_stream; },
  get synchronize() { return streamModule.synchronize; },
  get stream_context() { return streamModule.stream_context; },
  get stream() { return streamModule.stream; },
  get with_stream() { return streamModule.with_stream; },
  get reshape() { return opsModule.reshape; },
  get transpose() { return opsModule.transpose; },
  get moveaxis() { return opsModule.moveaxis; },
  get swapaxes() { return opsModule.swapaxes; },
  get add() { return opsModule.add; },
  get multiply() { return opsModule.multiply; },
  get subtract() { return opsModule.subtract; },
  get divide() { return opsModule.divide; },
  get power() { return opsModule.power; },
  get equal() { return opsModule.equal; },
  get not_equal() { return opsModule.not_equal; },
  get less() { return opsModule.less; },
  get less_equal() { return opsModule.less_equal; },
  get greater() { return opsModule.greater; },
  get greater_equal() { return opsModule.greater_equal; },
  get maximum() { return opsModule.maximum; },
  get minimum() { return opsModule.minimum; },
  get where() { return opsModule.where; },
  get arange() { return opsModule.arange; },
  get random() { return opsModule.random; },
  get tan() { return opsModule.tan; },
  get sin() { return opsModule.sin; },
  get cos() { return opsModule.cos; },
  get arcsin() { return opsModule.arcsin; },
  get arccos() { return opsModule.arccos; },
  get arctan() { return opsModule.arctan; },
  get arctan2() { return opsModule.arctan2; },
  get rsqrt() { return opsModule.rsqrt; },
  get square() { return opsModule.square; },
  get sign() { return opsModule.sign; },
  get abs() { return opsModule.abs; },
  get sqrt() { return opsModule.sqrt; },
  get exp() { return opsModule.exp; },
  get log() { return opsModule.log; },
  get import_function() { return opsModule.import_function; },
  get sum() { return opsModule.sum; },
  get mean() { return opsModule.mean; },
  get min() { return opsModule.min; },
  get max() { return opsModule.max; },
  get prod() { return opsModule.prod; },
  get argmin() { return opsModule.argmin; },
  get argmax() { return opsModule.argmax; },
  get logsumexp() { return opsModule.logsumexp; },
  get logcumsumexp() { return opsModule.logcumsumexp; },
  get softmax() { return opsModule.softmax; },
  get logaddexp() { return opsModule.logaddexp; },
  get clip() { return opsModule.clip; },
  get log1p() { return opsModule.log1p; },
  get negative() { return opsModule.negative; },
  get reciprocal() { return opsModule.reciprocal; },
  get expand_dims() { return opsModule.expand_dims; },
  get squeeze() { return opsModule.squeeze; },
  get concatenate() { return opsModule.concatenate; },
  get take_along_axis() { return opsModule.take_along_axis; },
  get matmul() { return opsModule.matmul; },
  get sigmoid() { return opsModule.sigmoid; },
  get erf() { return opsModule.erf; },
  get tanh() { return opsModule.tanh; },
  get split() { return opsModule.split; },
  get addmm() { return opsModule.addmm; },
  get variance() { return opsModule.variance; },
  get std() { return opsModule.std; },
  get stack() { return opsModule.stack; },
  get conv1d() { return opsModule.conv1d; },
  get conv2d() { return opsModule.conv2d; },
  get conv3d() { return opsModule.conv3d; },
  get take() { return opsModule.take; },
  get pad() { return opsModule.pad; },
  get slice() { return opsModule.slice; },
  get as_strided() { return opsModule.as_strided; },
  get number_of_elements() { return opsModule.number_of_elements; },
  get fast() { return opsModule.fast; },
  get distributed() { return opsModule.distributed; },
  get linalg() { return opsModule.linalg; },
  get trace() { return opsModule.trace; },
  get ceil() { return opsModule.ceil; },
  get floor() { return opsModule.floor; },
  get round() { return opsModule.round; },
  get isnan() { return opsModule.isnan; },
  get isinf() { return opsModule.isinf; },
  get isfinite() { return opsModule.isfinite; },
  get logical_not() { return opsModule.logical_not; },
  get sinh() { return opsModule.sinh; },
  get cosh() { return opsModule.cosh; },
  get arcsinh() { return opsModule.arcsinh; },
  get arccosh() { return opsModule.arccosh; },
  get arctanh() { return opsModule.arctanh; },
  get degrees() { return opsModule.degrees; },
  get radians() { return opsModule.radians; },
  get erfinv() { return opsModule.erfinv; },
  get expm1() { return opsModule.expm1; },
  get cumsum() { return opsModule.cumsum; },
  get cumprod() { return opsModule.cumprod; },
  get cummax() { return opsModule.cummax; },
  get cummin() { return opsModule.cummin; },
  get floor_divide() { return opsModule.floor_divide; },
  get remainder() { return opsModule.remainder; },
  get logical_and() { return opsModule.logical_and; },
  get logical_or() { return opsModule.logical_or; },
  get bitwise_and() { return opsModule.bitwise_and; },
  get bitwise_or() { return opsModule.bitwise_or; },
  get bitwise_xor() { return opsModule.bitwise_xor; },
  get left_shift() { return opsModule.left_shift; },
  get right_shift() { return opsModule.right_shift; },
  get all() { return opsModule.all; },
  get any() { return opsModule.any; },
  get array_equal() { return opsModule.array_equal; },
  get flatten() { return opsModule.flatten; },
  get eye() { return opsModule.eye; },
  get identity() { return opsModule.identity; },
  get linspace() { return opsModule.linspace; },
  get tril() { return opsModule.tril; },
  get triu() { return opsModule.triu; },
  get broadcast_to() { return opsModule.broadcast_to; },
  get repeat() { return opsModule.repeat; },
  get tile() { return opsModule.tile; },
  get sort() { return opsModule.sort; },
  get argsort() { return opsModule.argsort; },
  get diag() { return opsModule.diag; },
  get diagonal() { return opsModule.diagonal; },
  get topk() { return opsModule.topk; },
  get default_device() { return opsModule.default_device; },
  get set_default_device() { return opsModule.set_default_device; },
  get is_available() { return opsModule.is_available; },
  get clear_cache() { return opsModule.clear_cache; },
  get get_active_memory() { return opsModule.get_active_memory; },
  get get_cache_memory() { return opsModule.get_cache_memory; },
  get get_peak_memory() { return opsModule.get_peak_memory; },
  get reset_peak_memory() { return opsModule.reset_peak_memory; },
  get set_cache_limit() { return opsModule.set_cache_limit; },
  get set_memory_limit() { return opsModule.set_memory_limit; },
  get set_wired_limit() { return opsModule.set_wired_limit; },
  get fft() { return opsModule.fft; },
  get log2() { return opsModule.log2; },
  get log10() { return opsModule.log10; },
  get isposinf() { return opsModule.isposinf; },
  get isneginf() { return opsModule.isneginf; },
  get bitwise_invert() { return opsModule.bitwise_invert; },
  get conjugate() { return opsModule.conjugate; },
  get conj() { return opsModule.conj; },
  get real() { return opsModule.real; },
  get imag() { return opsModule.imag; },
  get stop_gradient() { return opsModule.stop_gradient; },
  get outer() { return opsModule.outer; },
  get inner() { return opsModule.inner; },
  get kron() { return opsModule.kron; },
  get nan_to_num() { return opsModule.nan_to_num; },
  get allclose() { return opsModule.allclose; },
  get isclose() { return opsModule.isclose; },
  get view() { return opsModule.view; },
  get contiguous() { return opsModule.contiguous; },
  get hadamard_transform() { return opsModule.hadamard_transform; },
  get unflatten() { return opsModule.unflatten; },
  get partition() { return opsModule.partition; },
  get argpartition() { return opsModule.argpartition; },
  get put_along_axis() { return opsModule.put_along_axis; },
  get roll() { return opsModule.roll; },
  get tri() { return opsModule.tri; },
  get meshgrid() { return opsModule.meshgrid; },
  get broadcast_arrays() { return opsModule.broadcast_arrays; },
  get atleast_1d() { return opsModule.atleast_1d; },
  get atleast_2d() { return opsModule.atleast_2d; },
  get atleast_3d() { return opsModule.atleast_3d; },
  get slice_update() { return opsModule.slice_update; },
  get conv_general() { return opsModule.conv_general; },
  get conv_transpose1d() { return opsModule.conv_transpose1d; },
  get conv_transpose2d() { return opsModule.conv_transpose2d; },
  get conv_transpose3d() { return opsModule.conv_transpose3d; },
  get einsum() { return opsModule.einsum; },
  get tensordot() { return opsModule.tensordot; },
  get block_masked_mm() { return opsModule.block_masked_mm; },
  get gather_mm() { return opsModule.gather_mm; },
  get segmented_mm() { return opsModule.segmented_mm; },
  get quantize() { return opsModule.quantize; },
  get dequantize() { return opsModule.dequantize; },
  get quantized_matmul() { return opsModule.quantized_matmul; },
  get gather_qmm() { return opsModule.gather_qmm; },
  get concat() { return opsModule.concat; },
  get divmod() { return opsModule.divmod; },
  get permute_dims() { return opsModule.permute_dims; },
  get trunc() { return opsModule.trunc; },
  get broadcast_shapes() { return opsModule.broadcast_shapes; },
  get convolve() { return opsModule.convolve; },
  get einsum_path() { return opsModule.einsum_path; },
  get eval_op() { return opsModule.eval_op; },
  get async_eval() { return opsModule.async_eval; },
  get load() { return opsModule.load; },
  get load_safetensors() { return opsModule.load_safetensors; },
  get load_gguf() { return opsModule.load_gguf; },
  get save() { return opsModule.save; },
  get save_safetensors() { return opsModule.save_safetensors; },
  get save_gguf() { return opsModule.save_gguf; },
  get grad() { return opsModule.grad; },
  get value_and_grad() { return opsModule.value_and_grad; },
  get vjp() { return opsModule.vjp; },
  get jvp() { return opsModule.jvp; },
  get vmap() { return opsModule.vmap; },
  get compile() { return opsModule.compile_fn; },
  get enable_compile() { return opsModule.enable_compile; },
  get disable_compile() { return opsModule.disable_compile; },
  get checkpoint() { return opsModule.checkpoint; },
  get export_function() { return opsModule.export_function; },
  get export_to_dot() { return opsModule.export_to_dot; },
  get cpu() { return deviceModule.cpu; },
  get gpu() { return deviceModule.gpu; },
  get device() { return deviceModule.device; },
  get Dtype() { return dtypeModule.Dtype; },
  get dtype() { return dtypeModule.default; },
  get dtypeFromString() { return dtypeModule.fromString; },
  get dtypeKeys() { return dtypeModule.keys; },
  get dtypeValues() { return dtypeModule.values; },
  get dtypeItems() { return dtypeModule.items; },
  get dtypeDir() { return dtypeModule.dir; },
  get dtypeHas() { return dtypeModule.has; },
  get dtypeGet() { return dtypeModule.get; },
  get dtypeCategoryKeys() { return dtypeModule.categoryKeys; },
  get dtypeCategoryValues() { return dtypeModule.categoryValues; },
  get dtypeCategoryItems() { return dtypeModule.categoryItems; },
  get bool() { return dtypeModule.bool; },
  get int8() { return dtypeModule.int8; },
  get int16() { return dtypeModule.int16; },
  get int32() { return dtypeModule.int32; },
  get int64() { return dtypeModule.int64; },
  get uint8() { return dtypeModule.uint8; },
  get uint16() { return dtypeModule.uint16; },
  get uint32() { return dtypeModule.uint32; },
  get uint64() { return dtypeModule.uint64; },
  get float16() { return dtypeModule.float16; },
  get bfloat16() { return dtypeModule.bfloat16; },
  get float32() { return dtypeModule.float32; },
  get float64() { return dtypeModule.float64; },
  get complex64() { return dtypeModule.complex64; },
  get complexfloating() { return dtypeModule.complexfloating; },
  get floating() { return dtypeModule.floating; },
  get inexact() { return dtypeModule.inexact; },
  get signedinteger() { return dtypeModule.signedinteger; },
  get unsignedinteger() { return dtypeModule.unsignedinteger; },
  get integer() { return dtypeModule.integer; },
  get number() { return dtypeModule.number; },
  get generic() { return dtypeModule.generic; },
};

// Top-level re-exports using functions to ensure late binding
export const array = (...args: any[]) => core.array(...args);
export const from_js_array = (...args: any[]) => core.from_js_array(...args);
export const array_builder = (...args: any[]) => core.array_builder(...args);
export const asarray = (...args: any[]) => core.asarray(...args);
export const ArrayBuilder = core.ArrayBuilder;
export const MLX_ARRAY_SYMBOL = core.MLX_ARRAY_SYMBOL;
export const Array = core.Array;
export const Stream = core.Stream;
export const issubdtype = (...args: any[]) => core.issubdtype(...args);
export const zeros = (...args: any[]) => core.zeros(...args);
export const zeros_like = (...args: any[]) => core.zeros_like(...args);
export const ones = (...args: any[]) => core.ones(...args);
export const ones_like = (...args: any[]) => core.ones_like(...args);
export const full = (...args: any[]) => core.full(...args);
export const default_stream = (...args: any[]) => core.default_stream(...args);
export const new_stream = (...args: any[]) => core.new_stream(...args);
export const set_default_stream = (...args: any[]) => core.set_default_stream(...args);
export const synchronize = (...args: any[]) => core.synchronize(...args);
export const stream_context = (...args: any[]) => core.stream_context(...args);
export const stream = (...args: any[]) => core.stream(...args);
export const with_stream = (...args: any[]) => core.with_stream(...args);
export const reshape = (...args: any[]) => core.reshape(...args);
export const transpose = (...args: any[]) => core.transpose(...args);
export const moveaxis = (...args: any[]) => core.moveaxis(...args);
export const swapaxes = (...args: any[]) => core.swapaxes(...args);
export const add = (...args: any[]) => core.add(...args);
export const multiply = (...args: any[]) => core.multiply(...args);
export const subtract = (...args: any[]) => core.subtract(...args);
export const divide = (...args: any[]) => core.divide(...args);
export const power = (...args: any[]) => core.power(...args);
export const equal = (...args: any[]) => core.equal(...args);
export const not_equal = (...args: any[]) => core.not_equal(...args);
export const less = (...args: any[]) => core.less(...args);
export const less_equal = (...args: any[]) => core.less_equal(...args);
export const greater = (...args: any[]) => core.greater(...args);
export const greater_equal = (...args: any[]) => core.greater_equal(...args);
export const maximum = (...args: any[]) => core.maximum(...args);
export const minimum = (...args: any[]) => core.minimum(...args);
export const where = (...args: any[]) => core.where(...args);
export const arange = (...args: any[]) => core.arange(...args);
export const random = core.random;
export const tan = (...args: any[]) => core.tan(...args);
export const sin = (...args: any[]) => core.sin(...args);
export const cos = (...args: any[]) => core.cos(...args);
export const arcsin = (...args: any[]) => core.arcsin(...args);
export const arccos = (...args: any[]) => core.arccos(...args);
export const arctan = (...args: any[]) => core.arctan(...args);
export const arctan2 = (...args: any[]) => core.arctan2(...args);
export const rsqrt = (...args: any[]) => core.rsqrt(...args);
export const square = (...args: any[]) => core.square(...args);
export const sign = (...args: any[]) => core.sign(...args);
export const abs = (...args: any[]) => core.abs(...args);
export const sqrt = (...args: any[]) => core.sqrt(...args);
export const exp = (...args: any[]) => core.exp(...args);
export const log = (...args: any[]) => core.log(...args);
export const import_function = (...args: any[]) => core.import_function(...args);
export const sum = (...args: any[]) => core.sum(...args);
export const mean = (...args: any[]) => core.mean(...args);
export const min = (...args: any[]) => core.min(...args);
export const max = (...args: any[]) => core.max(...args);
export const prod = (...args: any[]) => core.prod(...args);
export const argmin = (...args: any[]) => core.argmin(...args);
export const argmax = (...args: any[]) => core.argmax(...args);
export const logsumexp = (...args: any[]) => core.logsumexp(...args);
export const logcumsumexp = (...args: any[]) => core.logcumsumexp(...args);
export const softmax = (...args: any[]) => core.softmax(...args);
export const logaddexp = (...args: any[]) => core.logaddexp(...args);
export const clip = (...args: any[]) => core.clip(...args);
export const log1p = (...args: any[]) => core.log1p(...args);
export const negative = (...args: any[]) => core.negative(...args);
export const reciprocal = (...args: any[]) => core.reciprocal(...args);
export const expand_dims = (...args: any[]) => core.expand_dims(...args);
export const squeeze = (...args: any[]) => core.squeeze(...args);
export const concatenate = (...args: any[]) => core.concatenate(...args);
export const take_along_axis = (...args: any[]) => core.take_along_axis(...args);
export const matmul = (...args: any[]) => core.matmul(...args);
export const sigmoid = (...args: any[]) => core.sigmoid(...args);
export const erf = (...args: any[]) => core.erf(...args);
export const tanh = (...args: any[]) => core.tanh(...args);
export const split = (...args: any[]) => core.split(...args);
export const addmm = (...args: any[]) => core.addmm(...args);
export const variance = (...args: any[]) => core.variance(...args);
export const std = (...args: any[]) => core.std(...args);
export const stack = (...args: any[]) => core.stack(...args);
export const conv1d = (...args: any[]) => core.conv1d(...args);
export const conv2d = (...args: any[]) => core.conv2d(...args);
export const conv3d = (...args: any[]) => core.conv3d(...args);
export const take = (...args: any[]) => core.take(...args);
export const pad = (...args: any[]) => core.pad(...args);
export const slice = (...args: any[]) => core.slice(...args);
export const as_strided = (...args: any[]) => core.as_strided(...args);
export const number_of_elements = (...args: any[]) => core.number_of_elements(...args);
export const fast = core.fast;
export const distributed = core.distributed;
export const linalg = core.linalg;
export const trace = (...args: any[]) => core.trace(...args);
export const ceil = (...args: any[]) => core.ceil(...args);
export const floor = (...args: any[]) => core.floor(...args);
export const round = (...args: any[]) => core.round(...args);
export const isnan = (...args: any[]) => core.isnan(...args);
export const isinf = (...args: any[]) => core.isinf(...args);
export const isfinite = (...args: any[]) => core.isfinite(...args);
export const logical_not = (...args: any[]) => core.logical_not(...args);
export const sinh = (...args: any[]) => core.sinh(...args);
export const cosh = (...args: any[]) => core.cosh(...args);
export const arcsinh = (...args: any[]) => core.arcsinh(...args);
export const arccosh = (...args: any[]) => core.arccosh(...args);
export const arctanh = (...args: any[]) => core.arctanh(...args);
export const degrees = (...args: any[]) => core.degrees(...args);
export const radians = (...args: any[]) => core.radians(...args);
export const erfinv = (...args: any[]) => core.erfinv(...args);
export const expm1 = (...args: any[]) => core.expm1(...args);
export const cumsum = (...args: any[]) => core.cumsum(...args);
export const cumprod = (...args: any[]) => core.cumprod(...args);
export const cummax = (...args: any[]) => core.cummax(...args);
export const cummin = (...args: any[]) => core.cummin(...args);
export const floor_divide = (...args: any[]) => core.floor_divide(...args);
export const remainder = (...args: any[]) => core.remainder(...args);
export const logical_and = (...args: any[]) => core.logical_and(...args);
export const logical_or = (...args: any[]) => core.logical_or(...args);
export const bitwise_and = (...args: any[]) => core.bitwise_and(...args);
export const bitwise_or = (...args: any[]) => core.bitwise_or(...args);
export const bitwise_xor = (...args: any[]) => core.bitwise_xor(...args);
export const left_shift = (...args: any[]) => core.left_shift(...args);
export const right_shift = (...args: any[]) => core.right_shift(...args);
export const all = (...args: any[]) => core.all(...args);
export const any = (...args: any[]) => core.any(...args);
export const array_equal = (...args: any[]) => core.array_equal(...args);
export const flatten = (...args: any[]) => core.flatten(...args);
export const eye = (...args: any[]) => core.eye(...args);
export const identity = (...args: any[]) => core.identity(...args);
export const linspace = (...args: any[]) => core.linspace(...args);
export const tril = (...args: any[]) => core.tril(...args);
export const triu = (...args: any[]) => core.triu(...args);
export const broadcast_to = (...args: any[]) => core.broadcast_to(...args);
export const repeat = (...args: any[]) => core.repeat(...args);
export const tile = (...args: any[]) => core.tile(...args);
export const sort = (...args: any[]) => core.sort(...args);
export const argsort = (...args: any[]) => core.argsort(...args);
export const diag = (...args: any[]) => core.diag(...args);
export const diagonal = (...args: any[]) => core.diagonal(...args);
export const topk = (...args: any[]) => core.topk(...args);
export const bitwise_invert = (...args: any[]) => core.bitwise_invert(...args);
export const conjugate = (...args: any[]) => core.conjugate(...args);
export const conj = (...args: any[]) => core.conj(...args);
export const real = (...args: any[]) => core.real(...args);
export const imag = (...args: any[]) => core.imag(...args);
export const stop_gradient = (...args: any[]) => core.stop_gradient(...args);
export const outer = (...args: any[]) => core.outer(...args);
export const inner = (...args: any[]) => core.inner(...args);
export const kron = (...args: any[]) => core.kron(...args);
export const nan_to_num = (...args: any[]) => core.nan_to_num(...args);
export const allclose = (...args: any[]) => core.allclose(...args);
export const isclose = (...args: any[]) => core.isclose(...args);
export const view = (...args: any[]) => core.view(...args);
export const contiguous = (...args: any[]) => core.contiguous(...args);
export const hadamard_transform = (...args: any[]) => core.hadamard_transform(...args);
export const unflatten = (...args: any[]) => core.unflatten(...args);
export const partition = (...args: any[]) => core.partition(...args);
export const argpartition = (...args: any[]) => core.argpartition(...args);
export const put_along_axis = (...args: any[]) => core.put_along_axis(...args);
export const roll = (...args: any[]) => core.roll(...args);
export const tri = (...args: any[]) => core.tri(...args);
export const meshgrid = (...args: any[]) => core.meshgrid(...args);
export const broadcast_arrays = (...args: any[]) => core.broadcast_arrays(...args);
export const atleast_1d = (...args: any[]) => core.atleast_1d(...args);
export const atleast_2d = (...args: any[]) => core.atleast_2d(...args);
export const atleast_3d = (...args: any[]) => core.atleast_3d(...args);
export const slice_update = (...args: any[]) => core.slice_update(...args);
export const conv_general = (...args: any[]) => core.conv_general(...args);
export const conv_transpose1d = (...args: any[]) => core.conv_transpose1d(...args);
export const conv_transpose2d = (...args: any[]) => core.conv_transpose2d(...args);
export const conv_transpose3d = (...args: any[]) => core.conv_transpose3d(...args);
export const einsum = (...args: any[]) => core.einsum(...args);
export const tensordot = (...args: any[]) => core.tensordot(...args);
export const block_masked_mm = (...args: any[]) => core.block_masked_mm(...args);
export const gather_mm = (...args: any[]) => core.gather_mm(...args);
export const segmented_mm = (...args: any[]) => core.segmented_mm(...args);
export const quantize = (...args: any[]) => core.quantize(...args);
export const dequantize = (...args: any[]) => core.dequantize(...args);
export const quantized_matmul = (...args: any[]) => core.quantized_matmul(...args);
export const gather_qmm = (...args: any[]) => core.gather_qmm(...args);
export const concat = (...args: any[]) => core.concat(...args);
export const divmod = (...args: any[]) => core.divmod(...args);
export const permute_dims = (...args: any[]) => core.permute_dims(...args);
export const trunc = (...args: any[]) => core.trunc(...args);
export const broadcast_shapes = (...args: any[]) => core.broadcast_shapes(...args);
export const convolve = (...args: any[]) => core.convolve(...args);
export const einsum_path = (...args: any[]) => core.einsum_path(...args);
export const eval_op = (...args: any[]) => core.eval_op(...args);
export const async_eval = (...args: any[]) => core.async_eval(...args);
export const load = (...args: any[]) => core.load(...args);
export const load_safetensors = (...args: any[]) => core.load_safetensors(...args);
export const load_gguf = (...args: any[]) => core.load_gguf(...args);
export const save = (...args: any[]) => core.save(...args);
export const save_safetensors = (...args: any[]) => core.save_safetensors(...args);
export const save_gguf = (...args: any[]) => core.save_gguf(...args);
export const grad = (...args: any[]) => core.grad(...args);
export const value_and_grad = (...args: any[]) => core.value_and_grad(...args);
export const vjp = (...args: any[]) => core.vjp(...args);
export const jvp = (...args: any[]) => core.jvp(...args);
export const vmap = (...args: any[]) => core.vmap(...args);
export const compile = (...args: any[]) => core.compile(...args);
export const enable_compile = (...args: any[]) => core.enable_compile(...args);
export const disable_compile = (...args: any[]) => core.disable_compile(...args);
export const checkpoint = (...args: any[]) => core.checkpoint(...args);
export const export_function = (...args: any[]) => core.export_function(...args);
export const export_to_dot = (...args: any[]) => core.export_to_dot(...args);
export const cpu = core.cpu;
export const gpu = (index?: number) => core.gpu(index);
export const device = (...args: any[]) => core.device(...args);
export const Dtype = core.Dtype;
export const dtype = core.dtype;
export const dtypeFromString = core.dtypeFromString;
export const dtypeKeys = core.dtypeKeys;
export const dtypeValues = core.dtypeValues;
export const dtypeItems = core.dtypeItems;
export const dtypeDir = core.dtypeDir;
export const dtypeHas = core.dtypeHas;
export const dtypeGet = core.dtypeGet;
export const dtypeCategoryKeys = core.dtypeCategoryKeys;
export const dtypeCategoryValues = core.dtypeCategoryValues;
export const dtypeCategoryItems = core.dtypeCategoryItems;
export const bool = core.bool;
export const int8 = core.int8;
export const int16 = core.int16;
export const int32 = core.int32;
export const int64 = core.int64;
export const uint8 = core.uint8;
export const uint16 = core.uint16;
export const uint32 = core.uint32;
export const uint64 = core.uint64;
export const float16 = core.float16;
export const bfloat16 = core.bfloat16;
export const float32 = core.float32;
export const float64 = core.float64;
export const complex64 = core.complex64;
export const complexfloating = core.complexfloating;
export const floating = core.floating;
export const inexact = core.inexact;
export const signedinteger = core.signedinteger;
export const unsignedinteger = core.unsignedinteger;
export const integer = core.integer;
export const number = core.number;
export const generic = core.generic;
export const fft = core.fft;

export default core;
