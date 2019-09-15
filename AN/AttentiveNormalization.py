import keras.backend as K
from keras import layers
from AN.custom_objects import ANInitializer



class AttentiveNormalization(layers.BatchNormalization):
    
    def __init__(self, n_mixture=5, momentum=0.99, epsilon=0.1, axis=-1, **kwargs):
        super(AttentiveNormalization, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)

        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
            
        self.n_mixture = n_mixture
        
    def build(self, input_shape):
        if len(input_shape) != 4 and len(input_shape) != 3:
            raise ValueError('expected 3D or 4D input, got shape {}'.format(input_shape))
            
        super(AttentiveNormalization, self).build(input_shape)
        
        dim = input_shape[self.axis]
        shape = (self.n_mixture, dim) # K x C 
        
        self.FC = layers.Dense(self.n_mixture, activation="sigmoid")
        self.FC.build(input_shape) # (N, C)
        
        if len(input_shape) == 4:
            self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        else:
            self.GlobalAvgPooling = layers.GlobalAveragePooling1D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
        
        self._trainable_weights = self.FC.trainable_weights
        
        self.learnable_weights = self.add_weight(name='gamma2', 
                                      shape=shape,
                                      initializer=ANInitializer(scale=0.1, bias=1.),
                                      trainable=True)

        self.learnable_bias = self.add_weight(name='bias2', 
                                    shape=shape,
                                    initializer=ANInitializer(scale=0.1, bias=0.),
                                    trainable=True)
        

    def call(self, input):
        # input is a batch of shape : (N, H, W, C)
        avg = self.GlobalAvgPooling(input) # N x C 
        attention = self.FC(avg) # N x K 
        gamma_readjust = K.dot(attention, self.learnable_weights) # N x C
        beta_readjust  = K.dot(attention, self.learnable_bias)  # N x C
        
        out_BN = super(AttentiveNormalization, self).call(input) # rescale input, N x H x W x C

        # broadcast if needed
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            if len(input_shape) == 4:
                gamma_readjust = gamma_readjust[:, None, None, :]
                beta_readjust  = beta_readjust[:, None, None, :]
            else:
                gamma_readjust = gamma_readjust[:, None, :]
                beta_readjust  = beta_readjust[:, None, :]

        return gamma_readjust * out_BN + beta_readjust

    def get_config(self):
        config = {
            'n_mixture' : self.n_mixture
        }
        base_config = super(AttentiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
