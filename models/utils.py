import jax.numpy as jnp
import jax

from netket.utils.types import Array


def integer_to_binary(x: Array, bitsize: int, axis=1, negative=False):
    x = jnp.flip(jnp.mod(jnp.right_shift(jnp.expand_dims(x, axis), jnp.arange(bitsize)), 2), axis=axis)
    if negative:
        return 2 * x - 1
    else:
        return x


def binary_to_integer(binary_string: Array, axis=1):
    shape = binary_string.shape
    return jnp.sum(jnp.flip(binary_string, axis=axis).astype(jnp.int32) * 2 ** jnp.arange(shape[axis]),
                   axis=axis)


def spin_Lz_to_zero_indexed(x: Array, local_size: int) -> Array:
    return (x + local_size - 1).astype(jnp.int32) // 2


def spin_zero_indexed_to_Lz(x: Array, local_size: int) -> Array:
    return x * 2 - local_size + 1


def log_coeffs_to_log_probs(logCoeffs: Array, local_size: int, positive_or_complex: str, softmaxOnOrOff: str,
                            modulusOnOrOff: str, signOfProbsIntoPhase: str): # if we are working with spins, logCoeffs is of size 2
    # signOfProbsIntoPhase --> can only be 'on' if modulus and softmax functions both off.

    if softmaxOnOrOff == 'on':
        if positive_or_complex == 'complex': # logCoeffs will be of size 2 in the case "Softmax = on, wavefunction = complex"
            phase = 1j * jnp.concatenate([jnp.array([0.0]), logCoeffs[local_size - 1:]])
            amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[: local_size - 1]])
            to_be_returned = 0.5 * jax.nn.log_softmax(amp) + phase
            x = amp
            #x.tolist()
            #print("amp =",float(x[0]))
            #jax.debug.print("debug: {}", logCoeffs)
            
        elif positive_or_complex == 'positive': # logCoeffs will be of size 2 in the case "Softmax = on, wavefunction = positive"
            to_be_returned = 0.5 * jax.nn.log_softmax(logCoeffs)
            
    elif softmaxOnOrOff == 'off':
        if modulusOnOrOff == 'on':
            if positive_or_complex == 'complex': # logCoeffs will be of size 3 in the case "Softmax = off, wavefunction = complex"
                phase = 1j * jnp.concatenate([jnp.array([0.0]), logCoeffs[local_size:]]) # 3rd element of logCoeffs to be used for phase
                #amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[: local_size]])
                amp = jnp.array(logCoeffs[:local_size]) # first 2 elements will be the conditionals
                amp_abs_squared = jnp.absolute(amp)**2
                cond_probs = amp_abs_squared / jnp.sum(amp_abs_squared)
                to_be_returned = 0.5 * jax.numpy.log(cond_probs) + phase
                
            elif positive_or_complex == 'positive': # logCoeffs will be of size 2 in the case "Softmax = off, wavefunction = positive"
                amp_abs_squared = jnp.absolute(logCoeffs)**2
                cond_probs = amp_abs_squared / jnp.sum(amp_abs_squared)
                to_be_returned = 0.5 * jax.numpy.log(cond_probs)
        
        elif modulusOnOrOff == 'off':
            if signOfProbsIntoPhase == 'off': # 'off' means don't incorporate sign of Probs into phase. Use absolute value instead.
                # The wavefunction will now not be normalized. We will have to use Markov-Chain Monte Carlo here.
                if positive_or_complex == 'complex': # logCoeffs will be of size 3 in the case "Softmax = off, wavefunction = complex"
                    # Phase doesn't change if the modulus is off
                    phase = 1j * jnp.concatenate([jnp.array([0.0]), logCoeffs[local_size:]]) # 3rd element of logCoeffs to be used for phase
                    #amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[: local_size]])
                    cond_probs = jnp.absolute(logCoeffs[:local_size]) # I think we have to make the 2-component array positive, because the
                                                                      # conditionals have to be positive... sqrt(prob) * e^(phase) so clearly
                                                                      # prob must be positive so taking the absolute value of logCoeffs[:local_size]
                                                                      # is essential I think... And that introduces a slight non-linearity
                                                                      # unfortunately... although the absolute value is pretty linear.
                    to_be_returned = 0.5 * jax.numpy.log(cond_probs) + phase
                    
                elif positive_or_complex == 'positive': # logCoeffs will be of size 2 in the case "Softmax = off, wavefunction = positive"
                    cond_probs = jnp.absolute(logCoeffs) # I think we have to make the 2-component array positive, because the
                                                         # conditionals have to be positive... sqrt(prob) * e^(phase) so clearly
                                                         # prob must be positive so taking the absolute value of logCoeffs[:local_size]
                                                         # is essential I think... And that introduces a slight non-linearity
                                                         # unfortunately... although the absolute value is pretty linear.
                    to_be_returned = 0.5 * jax.numpy.log(cond_probs)
            
            elif signOfProbsIntoPhase == 'on': # ok, now no absolute value: take sign of unnormalized conditional probs and incorporate
                                               # sqrt(-1) = e^(-i)
                # The wavefunction will now not be normalized. We will have to use Markov-Chain Monte Carlo here.
                if positive_or_complex == 'complex': # logCoeffs will be of size 3 in the case "Softmax = off, wavefunction = complex"
                    # Phase doesn't change if the modulus is off


                    
                    phase = 1j * jnp.concatenate([jnp.array([0.0]), logCoeffs[local_size:]]) # 3rd element of logCoeffs to be used for phase
                    phase = phase.at[0].set(phase[0] + 1j * jnp.pi / 4 * (1 - jnp.sign(logCoeffs[0])))
                    phase = phase.at[1].set(phase[1] + 1j * jnp.pi / 4 * (1 - jnp.sign(logCoeffs[1])))
                    
                    #amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[: local_size]])
                    cond_probs = jnp.absolute(logCoeffs[:local_size]) # I think we have to make the 2-component array positive, because the
                                                                      # conditionals have to be positive... sqrt(prob) * e^(phase) so clearly
                                                                      # prob must be positive so taking the absolute value of logCoeffs[:local_size]
                                                                      # is essential I think... And that introduces a slight non-linearity
                                                                      # unfortunately... although the absolute value is pretty linear.
                    to_be_returned = 0.5 * jax.numpy.log(cond_probs) + phase
                    
                elif positive_or_complex == 'positive': # logCoeffs will be of size 2 in the case "Softmax = off, wavefunction = positive"
                    cond_probs = jnp.absolute(logCoeffs) # I think we have to make the 2-component array positive, because the
                                                         # conditionals have to be positive... sqrt(prob) * e^(phase) so clearly
                                                         # prob must be positive so taking the absolute value of logCoeffs[:local_size]
                                                         # is essential I think... And that introduces a slight non-linearity
                                                         # unfortunately... although the absolute value is pretty linear.
                    to_be_returned = 0.5 * jax.numpy.log(cond_probs)
            
        '''
        Absolute value is still a nonlinearity... but much less of one than the modulus function or Softmax. We'll get Roeland to check this.
        '''
    return to_be_returned
    # jax.nn.log_softmax: Computes the logarithm of the softmax function, which rescales elements to the range


def states_to_local_indices(inputs: Array, hilbert):
    return hilbert.states_to_local_indices(inputs)
