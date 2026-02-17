/**
 * Neural network loss functions.
 *
 * Port of mlx.nn.losses from Python MLX.
 * Every tensor operation goes through C++ MLX bindings — no pure JS math.
 */
import MLXArray from '../core/array';
import {
  abs,
  add,
  clip,
  divide,
  exp,
  log,
  logaddexp,
  logsumexp,
  maximum,
  mean,
  minimum,
  multiply,
  negative,
  power,
  square,
  squeeze,
  sqrt,
  subtract,
  sum,
  take_along_axis,
  less,
  where,
  expand_dims,
  linalg,
} from '../core/ops';

export type Reduction = 'none' | 'mean' | 'sum';

function _reduce(loss: MLXArray, reduction: Reduction): MLXArray {
  if (reduction === 'mean') {
    return mean(loss);
  }
  if (reduction === 'sum') {
    return sum(loss);
  }
  return loss;
}

/**
 * Computes the cross entropy loss.
 *
 * @param logits - Unnormalized logits.
 * @param targets - Ground truth values (class indices or probabilities).
 * @param options.weights - Optional per-sample weights.
 * @param options.axis - Axis over which to compute softmax. Default: -1.
 * @param options.labelSmoothing - Label smoothing factor in [0, 1). Default: 0.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function cross_entropy(
  logits: MLXArray,
  targets: MLXArray,
  options?: {
    weights?: MLXArray;
    axis?: number;
    labelSmoothing?: number;
    reduction?: Reduction;
  },
): MLXArray {
  const axis = options?.axis ?? -1;
  const labelSmoothing = options?.labelSmoothing ?? 0.0;
  const reduction = options?.reduction ?? 'none';

  if (labelSmoothing < 0 || labelSmoothing >= 1) {
    throw new Error(`Label smoothing must be in [0, 1), got ${labelSmoothing}.`);
  }

  // Whether targets are class indices or probabilities
  const targetsAsProbs = targets.shape.length === logits.shape.length;

  let score: MLXArray;
  if (targetsAsProbs) {
    score = sum(multiply(logits, targets), axis);
  } else {
    // targets[..., None] → expand last dim, then take and squeeze
    const targetsExpanded = expand_dims(targets, -1);
    score = squeeze(take_along_axis(logits, targetsExpanded, axis), -1);
  }

  const logsumexpLogits = logsumexp(logits, axis);

  let loss: MLXArray;
  if (labelSmoothing > 0) {
    const adjustedScore = multiply(1 - labelSmoothing, score);
    const meanLogits = mean(logits, axis);
    const smoothedLoss = multiply(-labelSmoothing, meanLogits);
    loss = add(subtract(logsumexpLogits, adjustedScore), smoothedLoss);
  } else {
    loss = subtract(logsumexpLogits, score);
  }

  if (options?.weights) {
    loss = multiply(loss, options.weights);
  }

  return _reduce(loss, reduction);
}

/**
 * Computes the binary cross entropy loss.
 *
 * @param inputs - Predicted values (logits if withLogits=true, probabilities otherwise).
 * @param targets - Binary target values in {0, 1}.
 * @param options.weights - Optional per-sample weights.
 * @param options.withLogits - Whether inputs are logits. Default: true.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'mean'.
 */
export function binary_cross_entropy(
  inputs: MLXArray,
  targets: MLXArray,
  options?: {
    weights?: MLXArray;
    withLogits?: boolean;
    reduction?: Reduction;
  },
): MLXArray {
  const withLogits = options?.withLogits ?? true;
  const reduction = options?.reduction ?? 'mean';

  let loss: MLXArray;
  if (withLogits) {
    loss = subtract(logaddexp(0, inputs), multiply(inputs, targets));
  } else {
    const logInputsClip = clip(log(inputs), -100, null);
    const logInputsInvClip = clip(log(subtract(1, inputs)), -100, null);
    loss = negative(
      add(
        multiply(targets, logInputsClip),
        multiply(subtract(1, targets), logInputsInvClip),
      ),
    );
  }

  if (options?.weights) {
    loss = multiply(loss, options.weights);
  }

  return _reduce(loss, reduction);
}

/**
 * Computes the L1 loss.
 *
 * @param predictions - Predicted values.
 * @param targets - Target values.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'mean'.
 */
export function l1_loss(
  predictions: MLXArray,
  targets: MLXArray,
  options?: { reduction?: Reduction },
): MLXArray {
  const reduction = options?.reduction ?? 'mean';
  const loss = abs(subtract(predictions, targets));
  return _reduce(loss, reduction);
}

/**
 * Computes the mean squared error loss.
 *
 * @param predictions - Predicted values.
 * @param targets - Target values.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'mean'.
 */
export function mse_loss(
  predictions: MLXArray,
  targets: MLXArray,
  options?: { reduction?: Reduction },
): MLXArray {
  const reduction = options?.reduction ?? 'mean';
  const loss = square(subtract(predictions, targets));
  return _reduce(loss, reduction);
}

/**
 * Computes the negative log likelihood loss.
 *
 * @param inputs - Predicted distribution in log space.
 * @param targets - Target class indices.
 * @param options.axis - Distribution axis. Default: -1.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function nll_loss(
  inputs: MLXArray,
  targets: MLXArray,
  options?: { axis?: number; reduction?: Reduction },
): MLXArray {
  const axis = options?.axis ?? -1;
  const reduction = options?.reduction ?? 'none';

  const targetsExpanded = expand_dims(targets, -1);
  const gathered = take_along_axis(inputs, targetsExpanded, axis);
  const loss = negative(squeeze(gathered, -1));
  return _reduce(loss, reduction);
}

/**
 * Computes the Gaussian negative log likelihood loss.
 *
 * @param inputs - Predicted means.
 * @param targets - Target values (samples).
 * @param vars - Predicted variances.
 * @param options.full - Include constant term. Default: false.
 * @param options.eps - Minimum variance for stability. Default: 1e-6.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'mean'.
 */
export function gaussian_nll_loss(
  inputs: MLXArray,
  targets: MLXArray,
  vars: MLXArray,
  options?: { full?: boolean; eps?: number; reduction?: Reduction },
): MLXArray {
  const eps = options?.eps ?? 1e-6;
  const reduction = options?.reduction ?? 'mean';

  const safeVars = maximum(vars, eps);
  let loss = multiply(
    0.5,
    add(log(safeVars), divide(square(subtract(targets, inputs)), safeVars)),
  );

  if (options?.full) {
    // 0.5 * log(2 * pi)
    const c = 0.5 * Math.log(2 * Math.PI);
    loss = add(loss, c);
  }

  return _reduce(loss, reduction);
}

/**
 * Computes the Kullback-Leibler divergence loss.
 *
 * @param inputs - Log probabilities for the predicted distribution.
 * @param targets - Log probabilities for the target distribution.
 * @param options.axis - Distribution axis. Default: -1.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function kl_div_loss(
  inputs: MLXArray,
  targets: MLXArray,
  options?: { axis?: number; reduction?: Reduction },
): MLXArray {
  const axis = options?.axis ?? -1;
  const reduction = options?.reduction ?? 'none';

  const loss = sum(
    multiply(exp(targets), subtract(targets, inputs)),
    axis,
  );
  return _reduce(loss, reduction);
}

/**
 * Computes the smooth L1 loss.
 *
 * @param predictions - Predicted values.
 * @param targets - Target values.
 * @param options.beta - Threshold for L2/L1 transition. Default: 1.0.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'mean'.
 */
export function smooth_l1_loss(
  predictions: MLXArray,
  targets: MLXArray,
  options?: { beta?: number; reduction?: Reduction },
): MLXArray {
  const beta = options?.beta ?? 1.0;
  const reduction = options?.reduction ?? 'mean';

  const diff = abs(subtract(predictions, targets));
  const cond = less(diff, beta);
  const loss = where(
    cond,
    multiply(0.5, divide(square(diff), beta)),
    subtract(diff, 0.5 * beta),
  );

  return _reduce(loss, reduction);
}

/**
 * Computes the triplet loss.
 *
 * @param anchors - Anchor samples.
 * @param positives - Positive samples.
 * @param negatives - Negative samples.
 * @param options.axis - Distribution axis. Default: -1.
 * @param options.p - Norm degree. Default: 2.
 * @param options.margin - Triplet margin. Default: 1.0.
 * @param options.eps - Small constant for stability. Default: 1e-6.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function triplet_loss(
  anchors: MLXArray,
  positives: MLXArray,
  negatives: MLXArray,
  options?: {
    axis?: number;
    p?: number;
    margin?: number;
    eps?: number;
    reduction?: Reduction;
  },
): MLXArray {
  const axis = options?.axis ?? -1;
  const p = options?.p ?? 2;
  const margin = options?.margin ?? 1.0;
  const eps = options?.eps ?? 1e-6;
  const reduction = options?.reduction ?? 'none';

  const posDist = sqrt(add(sum(power(subtract(anchors, positives), p), axis), eps));
  const negDist = sqrt(add(sum(power(subtract(anchors, negatives), p), axis), eps));
  const loss = maximum(add(subtract(posDist, negDist), margin), 0);
  return _reduce(loss, reduction);
}

/**
 * Computes the hinge loss.
 *
 * @param inputs - Predicted values.
 * @param targets - Target values (-1 or 1).
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function hinge_loss(
  inputs: MLXArray,
  targets: MLXArray,
  options?: { reduction?: Reduction },
): MLXArray {
  const reduction = options?.reduction ?? 'none';
  const loss = maximum(subtract(1, multiply(inputs, targets)), 0);
  return _reduce(loss, reduction);
}

/**
 * Computes the Huber loss.
 *
 * @param inputs - Predicted values.
 * @param targets - Target values.
 * @param options.delta - L1/L2 threshold. Default: 1.0.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function huber_loss(
  inputs: MLXArray,
  targets: MLXArray,
  options?: { delta?: number; reduction?: Reduction },
): MLXArray {
  const delta = options?.delta ?? 1.0;
  const reduction = options?.reduction ?? 'none';

  const errors = subtract(inputs, targets);
  const absErrors = abs(errors);
  const quadratic = minimum(absErrors, delta);
  const linear = subtract(absErrors, quadratic);
  const loss = add(multiply(0.5, power(quadratic, 2)), multiply(delta, linear));
  return _reduce(loss, reduction);
}

/**
 * Computes the log cosh loss.
 *
 * @param inputs - Predicted values.
 * @param targets - Target values.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function log_cosh_loss(
  inputs: MLXArray,
  targets: MLXArray,
  options?: { reduction?: Reduction },
): MLXArray {
  const reduction = options?.reduction ?? 'none';
  const errors = subtract(inputs, targets);
  const loss = subtract(logaddexp(errors, negative(errors)), Math.log(2));
  return _reduce(loss, reduction);
}

/**
 * Computes the margin ranking loss.
 *
 * @param inputs1 - Scores for first input.
 * @param inputs2 - Scores for second input.
 * @param targets - Labels (1 or -1).
 * @param options.margin - Margin. Default: 0.0.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function margin_ranking_loss(
  inputs1: MLXArray,
  inputs2: MLXArray,
  targets: MLXArray,
  options?: { margin?: number; reduction?: Reduction },
): MLXArray {
  const margin = options?.margin ?? 0.0;
  const reduction = options?.reduction ?? 'none';

  const differences = subtract(inputs1, inputs2);
  const loss = maximum(
    add(multiply(negative(targets), differences), margin),
    0,
  );
  return _reduce(loss, reduction);
}

/**
 * Computes the cosine similarity loss between two inputs.
 *
 * @param x1 - First set of inputs.
 * @param x2 - Second set of inputs.
 * @param options.axis - Embedding axis. Default: 1.
 * @param options.eps - Minimum denominator for numerical stability. Default: 1e-8.
 * @param options.reduction - 'none' | 'mean' | 'sum'. Default: 'none'.
 */
export function cosine_similarity_loss(
  x1: MLXArray,
  x2: MLXArray,
  options?: { axis?: number; eps?: number; reduction?: Reduction },
): MLXArray {
  const axis = options?.axis ?? 1;
  const eps = options?.eps ?? 1e-8;
  const reduction = options?.reduction ?? 'none';

  const x1Norm = linalg.norm(x1, null, axis);
  const x2Norm = linalg.norm(x2, null, axis);

  const loss = divide(
    sum(multiply(x1, x2), axis),
    maximum(multiply(x1Norm, x2Norm), eps),
  );

  return _reduce(loss, reduction);
}
