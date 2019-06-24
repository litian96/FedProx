from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class PerGodGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed gold Gradient Descent"""
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerGodGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)
            self._zeros_slot(v, "gold", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        
        vstar = self.get_slot(var, "vstar")
        gold = self.get_slot(var, "gold")

        var_update = state_ops.assign_sub(var, lr_t*(grad + gold + mu_t*(var-vstar))) #Update 'ref' by subtracting 'value
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def set_params(self, cog, avg_gradient, client):
        with client.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, cog):
                vstar = self.get_slot(variable, "vstar")
                vstar.load(value, client.model.sess)
        
        # get old gradient
        gprev = client.get_grads()

        # Find g_t - F'(old)
        gdiff = [g1-g2 for g1,g2 in zip(avg_gradient, gprev)]

        with client.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, grad in zip(all_vars, gdiff):
                gold = self.get_slot(variable, "gold")
                gold.load(grad, client.model.sess)
