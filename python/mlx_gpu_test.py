import mlx.core as mx
print('device info', mx.metal.device_info())
x = mx.zeros((2,2))
print(x)
matrix = mx.ones((2,2)) @ mx.ones((2,2))
mx.eval(matrix)
print(matrix)
