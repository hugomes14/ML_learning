import tensorflow as tf

#cria um tensor

tensor_n_d = tf.constant(
    value=[1,2,3],
    dtype=tf.float32
)

#diagonal tensor
eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None, #confina as colunas mas os valores continuam na diagonal como se fosse uma matriz linha*linha mas cortada na coluna x
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)

#cria um tensor de dimensoes dims e com elementos de valor value

fill_tensor = tf.fill(
    dims= [1, 3, 4],
    value= 4,
    name=None
)
#aqui é de valor 1
ones_tensor = tf.ones(
    shape= [5, 3], 
    dtype=tf.dtypes.float32,
    name=None
)
#aqui é de valor 0
zeros_tensor = tf.zeros(
        shape= [5, 3], 
    dtype=tf.dtypes.float32,
    name=None
)

#imita o input pelas dimensoes mas com value 1
ones_like_tensor = tf.ones_like(
    input=ones_tensor,
    dtype=tf.dtypes.float32,
    name=None
)
#imita o input pelas dimensoes mas com value 0
zeros_like_tensor = tf.zeros_like(
    input=ones_tensor,
    dtype=tf.dtypes.float32,
    name=None
)

#devolve a dimensão do tensor (1D, 2D,...)

rank = tf.rank(
    ones_like_tensor
    )
 
#cria um tensor random

random_tensor = tf.random.normal(
    shape=[5,5],
    mean=0.0,
    stddev=0.1,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

#indexing

tensor_two_d = tf.constant([[1,2,3],[4,5,6],[7,8,9], [10,11,12]], dtype=tf.float32)

print(tensor_two_d[0:2, 0:2]) # [[1. 2.], [4. 5.]]

