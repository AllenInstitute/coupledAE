import tensorflow as tf

def mean_squared_error(X,Y):
    return tf.math.reduce_mean(tf.math.squared_difference(X, Y))

def mse(zi, zj):
    '''MSE coupling loss function. Shrinks coupled representations to a point if not batchnormalized.\n
    Tensors `zi, zj` should have shape (batch_size,latent_dim) \n
    `latent_dim` is an `int`
    '''
    def loss(y_true, y_pred):
        return mean_squared_error(zi, zj)
    return loss

def compute_cov(z):
    '''Calculate covariance tensor of shape (latent_dim, latent_dim) \n
     Tensor `z` has shape (batchsize, latent_dim) \n
    `latent_dim` is inferred from shape of z. \n
    '''
    Z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    return tf.matmul(tf.transpose(Z), Z)/tf.cast(tf.shape(Z)[0] - 1, tf.float32) + 1e-3 * tf.eye(tf.shape(z)[1])

def fullcov(zi, zj):
    '''Full covariance based loss \n
    Tensors `zi, zj` should have shape (batch_size,latent_dim) \n
    `latent_dim` is an `int`
    '''

    batch_size = tf.shape(zi)[0]
    latent_dim = tf.shape(zi)[1]
    Zi = tf.reshape(zi, [batch_size, 1, latent_dim])
    Zj = tf.reshape(zj, [batch_size, 1, latent_dim])
    Sj_inv = tf.linalg.inv(tf.reshape(compute_cov(zj), [1, latent_dim, latent_dim]))
    L = tf.reduce_mean(tf.matmul(tf.matmul(
        Zj - Zi, tf.tile(Sj_inv, [batch_size, 1, 1])),
        tf.transpose(Zj - Zi, perm=[0, 2, 1])), axis=None)
    def loss(y_true, y_pred):
        return L
    return loss

def minvar(zi, zj):
    '''The loss is symmetric for the i-j autoencoder pair \n
    Tensors `zi, zj` should have shape (batch_size,latent_dim) \n
    '''
    batch_size = tf.shape(zi)[0]
    vars_i = tf.math.square(tf.linalg.svd(zi - tf.reduce_mean(zi, axis=0), compute_uv=False))/tf.cast(batch_size - 1, tf.float32)
    vars_j = tf.math.square(tf.linalg.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False))/tf.cast(batch_size - 1, tf.float32)
    minvar_i = tf.reduce_min(vars_i, axis=None)
    minvar_j = tf.reduce_min(vars_j, axis=None)
    L = mean_squared_error(zi, zj)/tf.math.minimum(minvar_i, minvar_j)
    def loss(y_true, y_pred):
        return L
    return loss