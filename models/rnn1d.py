# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
With Comments by Andrew Jreissaty.
'''

from typing import Any, Callable, Optional, Tuple
from jax._src.typing import Array, ArrayLike, DType

# The Any type: A special kind of type is Any. A static type checker will treat every type as being compatible with Any and Any as
# being compatible with every type.
# This means that it is possible to perform any operation or method call on a value of type Any and assign it to any variable:
# e.g.
#from typing import Any
#a: Any = None
#a = []          # OK
#a = 2           # OK
# ...
# The Callable type: Functions – or other callable objects – can be annotated using collections.abc.Callable or typing.Callable.
# Callable[[int], str] signifies a function that takes a single parameter of type int and returns a str.
# ...
# Optional: Optional[X] is equivalent to X | None (or Union[X, None]).
# ...
# typing.Tuple: Deprecated alias for tuple. tuple and Tuple are special-cased in the type system; see Annotating tuples for more details.

# Not sure why we need all the above. ALso I think all these objects are classes.

import jax #A neural network library and ecosystem for JAX designed for flexibility.
#JAX: Google JAX is a machine learning framework for transforming numerical functions, to be used in Python

'''
flax: A neural network library and ecosystem for JAX designed for flexibility.

Linen is a neural network API developed based on learning from our users and the broader JAX community.
Linen improves on much of the former flax.nn API (removed since v0.4.0), such as submodule sharing and better
support for non-trainable variables. Moreover, Linen builds on a "functional core", enabling direct usage of JAX
transformations such as vmap, remat or scan inside your modules. In Linen, Modules behave much closer to vanilla
Python objects, while still letting you opt-in to the concise single-method pattern many of our users love.
'''

'''
functools — Higher-order functions and operations on callable objects

The functools module is for higher-order functions: functions that act on or return other functions.
In general, any callable object can be treated as a function for the purposes of this module.

functools.partial(func, /, *args, **keywords)
Return a new partial object which when called will behave like func called with the positional arguments args and
keyword arguments keywords. If more arguments are supplied to the call, they are appended to args. If additional
keyword arguments are supplied, they extend and override keywords. Roughly equivalent to:
'''

'''
utils: A grab-bag of utility functions and objects

states_to_local_indices(x)
https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.HomogeneousHilbert.html
Returns a tensor with the same shape of x, where all local values are converted to indices in the range 0…self.shape[i].
This function is guaranteed to be jax-jittable.

For the Fock space this returns x, but for other hilbert spaces such as Spin this returns an array of indices.


'''

from flax import linen as nn
from jax import numpy as jnp
from functools import partial

from .utils import states_to_local_indices, log_coeffs_to_log_probs

from netket.nn.masked_linear import default_kernel_init # pretty sure default_kernel_init = lecun_normal()
#from jax.nn.initializers import normal
from netket.nn import activation as nkactivation
from netket.utils.types import DType, NNInitFunc
from netket.models import AbstractARNN

from flax.linen.activation import sigmoid, tanh
from flax.linen.initializers import orthogonal, uniform

# jax.random.PRNGKey(seed, *, impl=None)[source]: Create a pseudo-random number generator (PRNG) key given an integer seed
#PRNGKey = jax.random.PRNGKey(seed = 111)
PRNGKey = Any # I guess this means any number?
Shape = Tuple[int, ...] # guessing you are specifying that Shape will always be a tupe of integers
Dtype = Any   # Not sure why we need to say that Dtype and Array are "Any"
Array = Any
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]


class RNN1D(AbstractARNN): # the subclass is what allows netket to perform autoregressive generation of sampling, I think.
                           # We use netket's autoregressive sampling method, but it's 
    """Autoregressive neural network with dense layers."""
    L: int
    """RNN dimensions (# of spins in 1D chain)"""
    
    numSpins_A: int
    """number of spins in region A for calculation of Renyi entropy"""
    
    dhidden: int
    """Hidden state size"""
    
    
    rnn_cell_type: str = 'RNN' # this is essentially a vanilla RNN
    """RNN cell."""
    
    activation_fn: Callable[..., Any] = tanh
    """activation function used for output and memory update (default: tanh)."""
    
    gate_fn : Callable[..., Any] = sigmoid
    """activation function used for gates, in case GRU is used for memory update (default: sigmoid)"""
    
    
    activation: Callable[[Array], Array] = nkactivation.reim_selu # you want the activation function to be callable and to be able to take
                                                                  # an array as input and output another array
    # https://netket.readthedocs.io/en/latest/api/_generated/nn/netket.nn.activation.reim_selu.html
    # Scaled exponential linear unit activation function
    """the nonlinear activation function between hidden layers (default: reim_selu)."""
    
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float32
    """the dtype of the computation (default: float32)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights that transform the inputs & hidden states in all RNN transformations except some of the GRU
       transformations (default: lecun_normal)"""
    
    recurrent_kernel_init: NNInitFunc = orthogonal()
    """initializer for the weights that transfrom the hidden states in some of the GRU transformations (default: orthogonal)"""

    bias_init: NNInitFunc = uniform(1.0)

    positive_or_complex: str = 'complex' # pRNN or cRNN based on Mohamed's paper. Default is pRNN
    
    softmaxOnOrOff: str = 'on' # argument that tells us whether or not the modulus function will be on or off
                               # (can only be on when the softmax function is off, but can be off as well when the softmax is off) 
    
    modulusOnOrOff: str = 'off' # argument that tells us whether or not the Softmax function will be on or off
    
    signOfProbsIntoPhase: str = 'off' # argument that tells us whether or not to incorporate the sign of the unnormalized conditional (in the event
                                      # that softmax and modulus functions both off and we are using MCMC. 'on' means do incorporate
                                      # any negative sign in the conditionals into the phase.
    
    """initializer for the biases."""
    # netket.utils.types is an netket.utils contains Utility functions and classes.


    # When an instance of RNN2D is created, it is initialized with the methods encoded in self below.
    def setup(self):
        # Recall, RNNCell(...), GRUCell(...) and MDRNNCell(...) are also objects / instances of classes.
        
        #self.L = self.Lx * self.Ly
        if self.rnn_cell_type == 'RNN':
            self.rnncell = RNNCell(hidden_features=self.dhidden,
                                   n_visible_states=self.hilbert.local_size, # hilbert comes from the AbstractARNN subclass
                                   kernel_init=self.kernel_init,
                                   bias_init=self.bias_init,
                                   param_dtype=self.param_dtype,
                                   activation_fn=self.activation_fn)
        elif self.rnn_cell_type == 'GRU':
            self.rnncell = GRUCell(hidden_features=self.dhidden,
                                   n_visible_states=self.hilbert.local_size,
                                   kernel_init=self.kernel_init,
                                   recurrent_kernel_init = self.recurrent_kernel_init,
                                   bias_init=self.bias_init,
                                   param_dtype=self.param_dtype,
                                   activation_fn = self.activation_fn,
                                   gate_fn = self.gate_fn)
        elif self.rnn_cell_type == 'GRU_Mohamed':
            self.rnncell = GRUCell_Mohamed(hidden_features=self.dhidden,
                                   n_visible_states=self.hilbert.local_size,
                                   kernel_init=self.kernel_init,
                                   recurrent_kernel_init = self.recurrent_kernel_init,
                                   bias_init=self.bias_init,
                                   param_dtype=self.param_dtype,
                                   activation_fn = self.activation_fn,
                                   gate_fn = self.gate_fn)
        elif self.rnn_cell_type == 'MDRNN':
            self.rnncell = MDRNNCell(hidden_features=self.dhidden,
                                     n_visible_states=self.hilbert.local_size,
                                     kernel_init=self.kernel_init,
                                     bias_init=self.bias_init,
                                     param_dtype=self.param_dtype,
                                     activation_fn = self.activation_fn)
        else:
            raise NotImplementedError # without an RNN cell type, we want there to be an error and the simulation / code execution to be killed.

        if self.softmaxOnOrOff == 'on':
            self.outputDense = nn.Dense(features=(self.hilbert.local_size - 1) * 2,
                                        kernel_init=self.kernel_init,
                                        use_bias=self.use_bias,
                                        bias_init=self.bias_init,
                                        param_dtype=self.param_dtype)
        elif self.softmaxOnOrOff == 'off': # whether modulus is on or off, we will need three elements for a complex wavefunction
            if self.positive_or_complex == 'complex':
                self.outputDense = nn.Dense(features=self.hilbert.local_size + 1, # need 2 elements for amp, 1 for phase if softmax is OFF and wavefunction is complex.
                                            kernel_init=self.kernel_init,
                                            use_bias=self.use_bias,
                                            bias_init=self.bias_init,
                                            param_dtype=self.param_dtype)
            elif self.positive_or_complex == 'positive': # whether modulus is on or off, we will need 2 elements for a positive wavefunction
                self.outputDense = nn.Dense(features=self.hilbert.local_size, # need 2 elements for amp and none for phase if softmax is OFF and wavefunction is positive.
                                            kernel_init=self.kernel_init,
                                            use_bias=self.use_bias,
                                            bias_init=self.bias_init,
                                            param_dtype=self.param_dtype)
        # self.hilbert.local_size I think is somehow a feature of any AbstractARNN object... actually yes.

    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          ...
          AJ: pretty sure Hilbert.size is just the number of spins, L.

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        """

        # AJ: numpy.ndarray.ndim: Number of array dimensions.
        inputs_dim = inputs.ndim # want this to be 2, right?
        # if there are more batch dims, flatten
        if inputs_dim > 2:
            inputs = jnp.reshape(inputs, (-1, inputs.shape[-1])) # i.e. inputs.shape[-1] is the last dimension of size Hilbert.size
                                                                 # (number of spins), and so then flatten everything else to produce
                                                                 # a two dimensional array, where the -1 represents the size of the
                                                                 # remaining dimension if the last dimension has size inputs.shape[-1]
        #inputs_shape = list(inputs.shape) # guessing inputs.shape is an np.array, and for some reason, we want it in list form.
        idx = states_to_local_indices(inputs, self.hilbert)

        
        #producing the one-hot vectors
        inputs_one_hot = jax.nn.one_hot(idx, self.hilbert.local_size, dtype=jnp.int32, axis=-1) # axis = -1 I think means introduce the
                                                                                                # new dimension at the end. So the shape
                                                                                                # goes from (batch,numspins) to
                                                                                                # (batch,numspins,2)
        inputs_one_hot = jnp.reshape(inputs_one_hot, (-1, *inputs_one_hot.shape[1:])) # I'm not sure why we do this reshaping operation.
                                                                                      # Guessing it's the thing we saw above, "if there
                                                                                      # are more batch dims, flatten"
        log_psi = conditionals_log_psi(inputs_one_hot,
                                       self.L,
                                       self.hilbert.local_size,
                                       self.dhidden,
                                       self.rnncell,
                                       self.outputDense,
                                       self.positive_or_complex,
                                       self.softmaxOnOrOff,
                                       self.modulusOnOrOff,
                                       self.signOfProbsIntoPhase
                                       )
        
        if self.positive_or_complex == 'complex':
            # cRNN
            p = jnp.exp(2 * log_psi.real)
        
        elif self.positive_or_complex == 'positive':
            # pRNN
            p = jnp.exp(2 * log_psi) # log_psi is real if we are dealing with a pRNN. At least it should be. Test it.

        return p

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Computes the log wave-functions for input configurations.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          ...
          AJ: OK so the input is the full sample, for each sample.

        Returns:
          The log psi with dimension (batch,).
          ...
          AJ: i.e. return log psi (full) for each configuration
        """
        idx = states_to_local_indices(inputs, self.hilbert)

        inputs_one_hot = jax.nn.one_hot(idx, self.hilbert.local_size, dtype=jnp.int32, axis=-1)

        log_psi = conditionals_log_psi(inputs_one_hot,
                                       self.L,
                                       self.hilbert.local_size,
                                       self.dhidden,
                                       self.rnncell,
                                       self.outputDense,
                                       self.positive_or_complex,
                                       self.softmaxOnOrOff,
                                       self.modulusOnOrOff,
                                       self.signOfProbsIntoPhase
                                       ) # shape (batch, Hilbert.size, self.hilbert.local_size)

        # Makes total sense. Contract along one-hot and L dimensions to get the log_psi for each sample, producing an array
        # of length batch... inputs_one hot has same shape as log_psi
        log_psi = (log_psi * inputs_one_hot).sum(axis=(1, 2))
        
        return log_psi # jnp.array of length (batch)


@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, None), out_axes=0)
def conditionals_log_psi(x: Array, L: int, local_size: int, dhidden: int,
                         rnncell: Callable,
                         outputdense: Callable,
                         positive_or_complex: str, softmaxOnOrOff: str, modulusOnOrOff: str, signOfProbsIntoPhase: str) -> Array:
    """
    Computes the log of the conditional wave-functions for each site to take each value.

    Args:
      x: configurations with dimensions (batch, Hilbert.size, self.hilbert.local_size).
      ...
      AJ: that's not necessarily true. Above, the call is conditionals_log_psi(inputs_one_hot, etc.) but inputs_one_hot is of
      dimension (batch, Hilbert.size, 2), right? YES.
      ...
      Also, I think Ly is the vertical dimension (number of rows) and Lx is horizontal dimension (number of columns)
      ...
      Additional arguments:
      positive_or_complex: will equal "positive" or "complex", and corresponds to pRNN or cRNN

    Returns:
      The log psi with dimensions (batch, Hilbert.size, Hilbert.local_size).
    """
    input_states = {}
    rnn_states = {}
    log_psi = jnp.zeros((L, local_size), dtype=complex)
    
    # Zero states
    nx = -1
    input_states[f"{nx}"] = jnp.zeros(local_size)
    rnn_states[f"{nx}"] = jnp.zeros(dhidden)

    for nx in range(L):
        h_state = rnn_states[f"{nx - 1}"]
        v_state = input_states[f"{nx - 1}"]
        rnn_output, new_h_state = rnncell(h_state, v_state) # up to here, I get it. rnn_output, new_h_state are the same
                                                            # they are the new hidden vectors, probably of type ndarray
                                                            # or jnp.array (i.e. the np.array type of jax). Dimensionality
                                                            # is of course (batch, dhidden), but really just (dhidden) because
                                                            # jax.vmap takes care of the batch dimension
        #print("outputdense(rnn_output) =",outputdense(rnn_output))
        #jax.debug.print("outputdense(rnn_output): {}", outputdense(rnn_output))
        logProb = log_coeffs_to_log_probs(outputdense(rnn_output), local_size, positive_or_complex, softmaxOnOrOff, modulusOnOrOff, signOfProbsIntoPhase)
        
        input_states[f"{nx}"] = x[nx] # pull the one-hot vector associated with the sample at position (nx,ny)
        rnn_states[f"{nx}"] = new_h_state
        log_psi = log_psi.at[nx].set(jnp.nan_to_num(logProb, nan=-35)) # I think this replaces any NaN values that result in the logProb array
        '''
        We'll have to check if this nan to num thing is appropriate when we use MCMC on an unnormalized wavefunction... probably it's OK
        for now.
        '''

    return log_psi # shape (L,local_size), and ultimately, through jax.vmap, shape becomes (batch, L, local_size)
    # log_psi will be real if we are dealing with a pRNN (positive_or_complex = 'positive'), complex if cRNN (positive_or_complex = 'complex')


'''
When you initialize an instance of RNNcell, all the variables you initialize it with become part of "self", I think.
e.g. down here, we would have self.hidden_features, self.n_visible states, etc.
'''
### RNN CELLS ###
class RNNCell(nn.Module):
    r"""RNN cell"""
    hidden_features: int
    """Hidden state size"""
    n_visible_states: int
    """Visible state size"""
    gate_fn: Callable[..., Any] = sigmoid
    """activation function used for gates (default: sigmoid)""" # I wonder what gates we are talking about here. I.e. gates in GRU?
                                                                # clearly this activation function is not needed for a non-GRU cell.
    activation_fn: Callable[..., Any] = tanh
    """activation function used for output and memory update (default: tanh).""" # i.e. to produce the final output?
    
    #kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal()
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    """initializer function for the kernels that transform the hidden state and input (default: lecun_normal).""" # initializing the weight
    
    #bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform(1.0)
    """initializer for the bias parameters (default: zeros)"""
    # flax.linen.initializers.uniform(scale=0.01, dtype=<class 'jax.numpy.float32'>): Builds an initializer that returns real
    # uniformly-distributed random arrays.
    
    dtype: Optional[Dtype] = None
    """the dtype of the computation (default: None)."""
    # I don't understand this "Optional[Dtype]" crap. Maybe this is the dtype of the output of the RNN cell?
    
    param_dtype: Dtype = jnp.float32
    """the dtype passed to parameter initializers (default: float32)."""
    
    layerNorm: bool = False # What is this?


    def setup(self) -> None:
        '''
        Defining dense layer with and without a bias? Maybe
        '''

        self.dense = nn.Dense(
            features=self.hidden_features, # I think these are the output featu
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)

        if self.layerNorm: # We're not really using this for now so forget about it
            self.layernorm = nn.LayerNorm()


    @nn.compact
    def __call__(self, carry, inputs):
        """Standard RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output

        """
        
        # h1 represents the hidden state produced by the horizontal adjacent cell or the vertical adjacent cell. h2 is the other.
        # v1 represents the input vector (visible state) produced by the horizontal adjacent cell or the vertical adjacent cell. v2 is the
        # other.
        h = carry # I think it's a jnp array of some kind. Yes, it's a jnp.array (equivalent to a np.array)
        v = inputs

        # jnp.append(h1, v1) concatenates h1 and v1, h2 and v2 also concatenated below. Then self.dense_1 applies a linear transformation
        # to the concatenated (h1, v1) vector and produces an output vector (new hidden state) of dimension self.hidden_features
        # (see above definition of self.dense_1). self.dense_2 does the same and includes a bias vector. Roeland clearly is normalizing
        # the preact_1 and preact_2 vectors. Why? After normalization, the new hidden state is computed by doing by applying the
        # output activation function. new_carry is output twice from this function call because one of the copies is essentially the
        # "new hidden" state whereas the other will be fed into a Softmax layer to produce the output vector at the given site.
        # ...
        preact = self.dense(jnp.append(h, v))
        #print("-----------")
        #print("preact =",preact)
        #print("-------------")
        if self.layerNorm:
            preact = self.layernorm(preact)
            
        new_carry = self.activation_fn(preact)
        return new_carry, new_carry # This code produces the new hidden state.


class GRUCell(nn.Module):
    r"""GRU cell"""
    hidden_features: int
    """Hidden state size"""
    n_visible_states: int
    """Visible state size"""
    gate_fn: Callable[..., Any] = sigmoid
    """activation function used for gates (default: sigmoid)"""
    activation_fn: Callable[..., Any] = tanh
    """activation function used for output and memory update (default: tanh)."""
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        default_kernel_init)
    """initializer function for the kernels that transform the input (default: lecun_normal)."""
    recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        orthogonal())
    """initializer function for the kernels that transform the hidden state (default: orthogonal)."""
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform(1.0)
    """initializer for the bias parameters (default: zeros)"""
    dtype: Optional[Dtype] = None
    """the dtype of the computation (default: None)."""
    param_dtype: Dtype = jnp.float32
    """the dtype passed to parameter initializers (default: float32)."""

    # Defining all the linear transformations needed in a 2D GRU.
    def setup(self) -> None: # this setup "function" returns an object of type None... something like that. The point is, we don't want
                             # setup() to return anything
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_ir = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hr = nn.Dense(
            features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)
        self.dense_iz = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hz = nn.Dense(
            features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)
        self.dense_in = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hn = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)
        self.dense_merge = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)

    @nn.compact
    def __call__(self, carry, inputs):
        """Gated recurrent unit (GRU) cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(carry))
        z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(carry))
        # add bias because the linear transformations aren't directly summed.
        n = self.activation_fn(self.dense_in(inputs) + r * self.dense_hn(carry))
        new_h = (1. - z) * n + z * self.dense_merge(carry)
        '''
        This last line is the line that slightly differs from Mohamed's implementation in his original paper, otherwise everything
        is the same as his original GRU implementation
        '''
        return new_h, new_h # pretty sure that, since nn.module is a subclass of this class, that every instance of the GRUCell class
                            # inherits the methods of the nn.module class, and the nn.module class MUST contain a call function
                            # that produces an output vector and a new state vector, I'm pretty sure. This is enforced by the
                            # way nn.module is defined, i.e. flax.linen.module (I think)


# Mohamed's GRU, lifted directly from his paper
class GRUCell_Mohamed(nn.Module):
    r"""GRU cell"""
    hidden_features: int
    """Hidden state size"""
    n_visible_states: int
    """Visible state size"""
    gate_fn: Callable[..., Any] = sigmoid
    """activation function used for gates (default: sigmoid)"""
    activation_fn: Callable[..., Any] = tanh
    """activation function used for output and memory update (default: tanh)."""
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        default_kernel_init)
    """initializer function for the kernels that transform the input (default: lecun_normal)."""
    recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        orthogonal())
    """initializer function for the kernels that transform the hidden state (default: orthogonal)."""
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform(1.0)
    """initializer for the bias parameters (default: zeros)"""
    dtype: Optional[Dtype] = None
    """the dtype of the computation (default: None)."""
    param_dtype: Dtype = jnp.float32
    """the dtype passed to parameter initializers (default: float32)."""

    # Defining all the linear transformations needed in a 2D GRU.
    def setup(self) -> None: # this setup "function" returns an object of type None... something like that. The point is, we don't want
                             # setup() to return anything
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_ir = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hr = nn.Dense(
            features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)
        self.dense_iz = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hz = nn.Dense(
            features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)
        self.dense_in = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)
        self.dense_hn = nn.Dense(
            features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init)

    @nn.compact
    def __call__(self, carry, inputs):
        """Gated recurrent unit (GRU) cell, lifted exactly from Mohamed's paper.
        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """

        r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(carry))
        #r = self.dense_ir(inputs) + self.dense_hr(carry)
        z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(carry))
        #z = self.dense_iz(inputs) + self.dense_hz(carry)
        # add bias because the linear transformations aren't directly summed.
        n = self.activation_fn(self.dense_in(inputs) + r * self.dense_hn(carry))
        #n = self.dense_in(inputs) + r * self.dense_hn(carry)
        new_h = (1. - z) * carry + z * n
        #new_h = (1. - z) * n + z * self.dense_merge(carry)
        '''
        This last line is the line that slightly differs from Mohamed's implementation in his original paper, otherwise everything
        is the same as his original GRU implementation
        '''
        return new_h, new_h # pretty sure that, since nn.module is a subclass of this class, that every instance of the GRUCell class
                            # inherits the methods of the nn.module class, and the nn.module class MUST contain a call function
                            # that produces an output vector and a new state vector, I'm pretty sure. This is enforced by the
                            # way nn.module is defined, i.e. flax.linen.module (I think)


'''
NOT
'''
# This tensorized linear transformation is NOT being used for a GRU
class TensorDense(nn.Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """
    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: jax.lax.Precision = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init # initializer function for the weight matrices in the
                                                                                # linear transformations of the RNN cells.
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    dot_general: DotGeneralT = jax.lax.dot_general # General dot product/contraction operator.
                                                   # DotGeneralT = Callable[..., Array].
                                                   # meaning we want dot_general to be a function we can call, that produces an Array
                                                   # as its output. Although we defined "Array = Any" up top... so much going on here.

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          carry: The nd-array left of the contraction.
          inputs: The nd-array right of the contraction.

        Returns:
          The transformed input.
        """

        W = self.param('W',
                       self.kernel_init,
                       (self.features, self.features, jnp.shape(inputs)[-1]),
                       self.param_dtype)

        if self.use_bias:
            b = self.param('b', self.bias_init, (self.features,), # shape of b is self.features... one-dimensional array
                           self.param_dtype)

        else:
            b = None
        y = jnp.einsum('j,k->jk', carry, inputs)
        z = self.dot_general(y,
                             W,
                             (((0, 1), (1, 2)), ((), ())) 
                             )
        # (0,1) is the tuple containing the contracting dimensions of y, (1,2) is the tuple containing the contracting dimensions of W.
        # dimension of z is z.ndim, which should be self.features.

        
        # The comma is there to specify that (1) is a tuple, with one element that is 1, and the comma just reminds you that this is a tuple
        # and not a number
        if b is not None:
            z += jnp.reshape(b, (1,) * (z.ndim - 1) + (-1,)) # z.ndim is the dimenson of the z vector and I'm pretty sure it's
                                                             # a method in-built to jax.lax.dot_general... right? Whatever, assume it's
                                                             # correct and it means what I think it means, the dimension of z, which
                                                             # is clearly self.features, the remaining dimension of W that wasn't
                                                             # contracted over.
                                                             # The reshape operation is super confusing here. Supposed z.ndim = 3. Then
                                                             # (1,) * (z.ndim - 1) + (-1,) = (1,1,1,-1), which is a tuple,
                                                             # and reshaping b, which has the same shape as z, to (1,1,1,-1) produces
                                                             # a shape of (1,1,1,z.ndim) = (1,1,1,3). But z is of shape (3). So adding
                                                             # z to b (I tested this) produces an array of shape (1,1,1,3). Not sure why
                                                             # we need the extra dimensions.
                                                             # ...
                                                             # ...
                                                             # UPDATE!!!!!: I'm wrong: z.ndim is the number of array dimensions. Not the
                                                             # dimension of the z vector... In our case, z.ndim = 1, so b is being reshaped
                                                             # to an array of length (-1,), meaning an array of length len(b). So nothing changes,
                                                             # b doesn't change.
        return z


# This class is self-explanatory. Tensorized RNN cell, not a GRU though.
'''
A tensorized cell, not a tensorized GRU. TensorDense does the tensorization, and MDRNN applies the activation function (tanh here).
A GRU has a lot more going on, so this is not a GRU.... YEP AGREED.
'''
class MDRNNCell(nn.Module):
    """Tensorised cell"""
    hidden_features: int
    """Hidden state size"""
    n_visible_states: int
    """Visible state size"""
    gate_fn: Callable[..., Any] = sigmoid
    """activation function used for gates (default: sigmoid)"""
    activation_fn: Callable[..., Any] = tanh
    """activation function used for output and memory update (default: tanh)."""
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        default_kernel_init)
    """initializer function for the kernels that transform the hidden state and input (default: lecun_normal)."""
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = uniform(1.0)
    """initializer for the bias parameters (default: zeros)"""
    dtype: Optional[Dtype] = None
    """the dtype of the computation (default: None)."""
    param_dtype: Dtype = jnp.float32
    """the dtype passed to parameter initializers (default: float32)."""

    # In the above cells we were using nn.Dense to define the dense linear transformations. Now we use Tensorcell
    def setup(self) -> None:
        self.tensor_dense = TensorDense(features=self.hidden_features,
                                        use_bias=True,
                                        dtype=self.param_dtype,
                                        param_dtype=self.param_dtype,
                                        kernel_init=self.kernel_init,
                                        bias_init=self.bias_init)

    @nn.compact
    def __call__(self, carry, inputs):
        """Tensorized RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        preact = self.tensor_dense(carry, inputs)
        new_h = self.activation_fn(preact)
        return new_h, new_h