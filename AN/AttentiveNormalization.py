import keras.backend as K
from keras import layers
from AN.custom_objects import ANInitializer


class AttentiveNormalization(layers.BatchNormalization):
    def __init__(self, n_mixture=5, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(AttentiveNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon  = epsilon
        self.axis = axis
        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
        self.n_mixture = n_mixture
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
            
        dim = input_shape[self.axis]
        shape = (self.n_mixture, dim) # K x C 
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        self.FC = layers.Dense(self.n_mixture, activation="sigmoid")
        self.FC.build((dim, self.n_mixture))
        self._trainable_weights = self.FC.trainable_weights
    
        self.gamma = self.add_weight(name='gamma', 
                                      shape=shape,
                                      initializer=ANInitializer(scale=0.1, bias=1.),
                                      trainable=True)

        self.beta = self.add_weight(name='beta', 
                                    shape=shape,
                                    initializer=ANInitializer(scale=0.1, bias=0.),
                                    trainable=True)


        super(AttentiveNormalization, self).build(input_shape)


    def call(self, input):
        # input is a batch of shape : (N, H, W, C)
        avg = self.GlobalAvgPooling(input) # N x C 
        attention = self.FC(avg) # N x K 
        gamma_readjust = K.dot(attention, self.gamma) # N x C
        beta_readjust  = K.dot(attention, self.beta)  # N x C
        
        out_BN = layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon, 
                                           axis=self.axis, center=False, scale=False)(input) # rescale input, N x H x W x C
        
        # broadcast if needed
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            gamma_readjust = gamma_readjust[:, None, None, :]
            beta_readjust  = beta_readjust[:, None, None, :]
       
        return gamma_readjust * out_BN + beta_readjust

    def get_config(self):
        config = {
            'axis' : self.axis,
            'momentum' : self.momentum,
            'epsilon' : self.epsilon,
            'n_mixture' : self.n_mixture
        }
        base_config = super(AttentiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))