import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        print(grad_name)
        @ops.RegisterGradient(grad_name) #the gradient function is a function that takes the original Operation and n Tensor objects #
                                         # (representing the gradients with respect to each output of the op)
                                        #and returns m Tensor objects #
                                        # (representing the partial gradients with respect to each input of the op).
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()
