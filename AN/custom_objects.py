import keras.backend as K
from keras import initializers

class ANInitializer(initializers.Initializer):
    """Initialization for gamma and beta weights according to BigGan paper 
    (A. Brock, J. Donahue, and K. Simonyan. Large scale gan
    training for high fidelity natural image synthesis. arXiv
    preprint arXiv:1809.11096, 2018.)
    
        This initialization is equal to :  scale * N(0, 1) + bias
         
        # Arguments:
          scale: rescaling factor
          bias: bias factor
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused
        # Returns:
          an initialization for the variable
          
    """
    def __init__(self, scale=0.1, bias=0., seed=1997):
        super(ANInitializer, self).__init__()
        self.scale = scale
        self.bias = bias
        self.seed = seed

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()
        return self.scale * K.random_normal(shape=shape, mean=0.0, stddev=1, seed=self.seed) + self.bias