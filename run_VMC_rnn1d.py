#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:28:52 2024

@author: andrewjreissaty91
"""

import sys
print(sys.version)

import jax
import numpy as np
import time
import itertools as it
from VMC_rnn1d import main
from netket.utils.types import DType


#physical_devices = jax.devices('gpu')
#print("Num GPUs:", len(physical_devices))



if __name__ == '__main__':
    from argparse import ArgumentParser

    # Let us keep track of the execution time
    start_time = time.time()
    
    parser = ArgumentParser(description='Set parameters')
    parser.add_argument('--model', default='rnn')
    parser.add_argument('--lattice', default='1d')
    parser.add_argument('--L', default=10)
    parser.add_argument('--numSpins_A', default=5)
    parser.add_argument('--dhidden_min', default=50)
    parser.add_argument('--dhidden_max', default=50)
    parser.add_argument('--dhidden_step', default=1)
    parser.add_argument('--nsamples_training', default=500)
    parser.add_argument('--nsamples_final_calculation', default=10**5)
    parser.add_argument('--chunk_size',default=500)
    parser.add_argument('--numrealizations', default=10) # number of initializations of the wavefunction per data point
    parser.add_argument('--rnn_cell', default='GRU_Mohamed')
    parser.add_argument('--activation_fn', default='tanh')
    parser.add_argument('--gate_fn', default='sigmoid')
    parser.add_argument('--softmaxOnOrOff', default='on') # on/off = turn on or off the softmax and go from there.
    parser.add_argument('--modulusOnOrOff', default='off') # on/off = turn on or off the modulus function. Can only be on when
                                                          # softmax function is off, but can be off when softmax function is on or
                                                           # off.
    parser.add_argument('--signOfProbsIntoPhase', default='off') # if softmax is Off and modulus is Off, do we want to incorporate the sign
                                                                 # of the conditional probs into the phase or not.
                                                                 # Default will be to have softmax on, so this will only matter if the
                                                                 # softmax and modulus function are both off.
    parser.add_argument('--GaussianInitializer', default=1) # if 1, initialize as we always have. If 0, initialize using another initializer, usually Xavier Glorot.
    parser.add_argument('--XavierGlorotUniform', default=0) # if 1, initialize using Xavier Glorot uniform initializer, if 0, then no.
    parser.add_argument('--XavierGlorotNormal', default=0) # if 1, initialize using Xavier Glorot normal initializer, if 0, then no.
    # Widths for Gaussian initializers IF GaussianInitializer == 1 and if the Xavier Glorot initializers are 0 (only one of those 3 will ever be 1)
    parser.add_argument('--width_min', default=0.30)
    parser.add_argument('--width_max', default=0.30)
    parser.add_argument('--width_step', default=0.10)
    #parser.add_argument('--seed', default=0)
    parser.add_argument('--bias', default=1) # Using 1 or 0 to define True or False
    parser.add_argument('--weight_sharing', default=1)
    parser.add_argument('--autoreg', default=1)
    parser.add_argument('--exactOrMC', default='MC')
    parser.add_argument('--lr_schedule', default=0) # 1 means use a schedule, 0 means no schedule
    parser.add_argument('--lr', default=5e-3) # either fixed lr if no schedule, or initial value of lr if schedule
    parser.add_argument('--lr_decay_rate', default=0.1)
    parser.add_argument('--SR_diag_shift', default=0.1)
    parser.add_argument('--n_iter', default=500000)
    parser.add_argument('--ExactDiag', default=1) # Using 1 or 0 to define True or False
    parser.add_argument('--slurm_nr', default=0)
    #New argument
    parser.add_argument('--batch', default=0)

    # Hamiltonian
    parser.add_argument('--Hamiltonian', default='Heisenberg') #'TFIM', 'MHS', 'Heisenberg'. May do 'XXZ' down the line.
    
    # TFIM arguments
    parser.add_argument('--transverse_field',default=1)
    parser.add_argument('--Ising_coupling',default=1)

    # Heisenberg arguments
    parser.add_argument('--Heisenberg_coupling', default=1)  # The J in "-J"... or is it +J? I'm not sure...
    parser.add_argument('--constrain_total_sz', default=0) # 1 = constrain to 0. 0 = don't constrain.

    # XXZ arguments... UPDATE: not doing XXZ for now. 5 avril, 2025
    parser.add_argument('--Delta', default=1)  # Delta = 1 makes the model a Heisenberg model

    # TFIM and XXZ (combined) arguments... UPDATE: not doing XXZ for now. 5 avril, 2025
    parser.add_argument('--PBC_or_OBC', default='OBC')  # open or periodic boundary conditions
    
    # SR or Adam
    parser.add_argument('--SR_or_Adam',default='Adam')
    
    # pRNN or cRNN
    parser.add_argument('--positive_or_complex',default='complex')
    
    # ARD vs MCMC
    parser.add_argument('--sampler', default = 'ARD') # 'ARD' or 'MCMC'

    # convergence criteria
    parser.add_argument('--rel_error_conv',default=1e-3)
    parser.add_argument('--var_conv',default=1e-3)
    parser.add_argument('--exp_smoothing_factor',default=0.30) # smoothing factor for exponential moving averages of variance and energy
    
    # Show progress bar or not
    parser.add_argument('--show_progress',default=1)
    
    # MCMC inputs
    parser.add_argument('--n_chains', default=480) # was originally 480
    parser.add_argument('--n_discard_per_chain', default= 60) # was originally 20
    parser.add_argument('--sweep_size', default= 60) # was originally 160
    
    # dtype for RNN
    parser.add_argument('--dtype',default='jax.numpy.float64')

    args = parser.parse_args()
    model = str(args.model)
    lattice = str(args.lattice)
    L = int(args.L)
    numSpins_A = int(args.numSpins_A)
    dhidden_min = int(args.dhidden_min)
    dhidden_max = int(args.dhidden_max)
    dhidden_step = int(args.dhidden_step)
    nsamples_training = int(args.nsamples_training)
    nsamples_final_calculation = int(args.nsamples_final_calculation)
    chunk_size = int(args.chunk_size)
    numrealizations= int(args.numrealizations)
    rnn_cell = str(args.rnn_cell)
    
    activation_fn = str(args.activation_fn)
    gate_fn = str(args.gate_fn)
    softmaxOnOrOff = str(args.softmaxOnOrOff)
    modulusOnOrOff = str(args.modulusOnOrOff)
    signOfProbsIntoPhase = str(args.signOfProbsIntoPhase)

    GaussianInitializer = bool(int(args.GaussianInitializer))
    XavierGlorotUniform = bool(int(args.XavierGlorotUniform))
    XavierGlorotNormal = bool(int(args.XavierGlorotNormal))
    
    width_min = float(args.width_min)
    width_max = float(args.width_max)
    width_step = float(args.width_step)
    bias= bool(int(args.bias))
    weight_sharing= bool(int(args.weight_sharing))
    autoreg = bool(int(args.autoreg))
    #data = bool(int(args.data))
    #plot = bool(int(args.plot))
    #show = bool(int(args.show))
    exactOrMC = str(args.exactOrMC)
    lr_schedule = bool(int(args.lr_schedule))
    lr = float(args.lr)
    lr_decay_rate = float(args.lr_decay_rate)
    SR_diag_shift = float(args.SR_diag_shift)
    n_iter = int(args.n_iter)
    ExactDiag= bool(int(args.ExactDiag)) # perform exact diagonalization or not... depending on the model.
    
    slurm_nr = int(args.slurm_nr) # should be from 0-99
    batch = int(args.batch)
    
    Hamiltonian = str(args.Hamiltonian)

    # TFIM
    Gamma = float(args.transverse_field)
    J = float(args.Ising_coupling)

    # Heisenberg
    J_Heis = float(args.Heisenberg_coupling)
    constrain_total_sz = bool(int(args.constrain_total_sz))  # are we constraining total_sz to zero or not.

    # XXZ... UPDATE: not doing XXZ for now. 5 avril, 2025
    Delta = float(args.Delta)

    # Boundary conditions for TFIM or Heisenberg model... UPDATE: not doing XXZ for now. 5 avril, 2025
    PBC_or_OBC = str(args.PBC_or_OBC)
    
    SR_or_Adam = str(args.SR_or_Adam)
    
    positive_or_complex = str(args.positive_or_complex)
    
    # New quantity: MCMC or ARD sampler
    sampler = str(args.sampler)
    
    # New quantities needed for MCMC
    n_chains = int(args.n_chains)
    n_discard_per_chain = int(args.n_discard_per_chain)
    sweep_size = int(args.sweep_size)
    
    rel_error_conv = float(args.rel_error_conv)
    var_conv = float(args.var_conv)
    exp_smoothing_factor = float(args.exp_smoothing_factor)
    
    show_progress = bool(int(args.show_progress))

    dtype = str(args.dtype)

    if GaussianInitializer == 1 and XavierGlorotUniform == 0 and XavierGlorotNormal == 0:
    
        dhidden_list = np.arange(dhidden_min,dhidden_max+dhidden_step/10,dhidden_step,dtype=int)
        width_list = np.arange(width_min,width_max+width_step/10,width_step,dtype=float)
        dhidden_width_list = list(it.product(dhidden_list, width_list))
        print("len(dhidden_width_list) =",len(dhidden_width_list))
        dhidden = dhidden_width_list[slurm_nr + 1000*batch][0]
        width = dhidden_width_list[slurm_nr + 1000*batch][1]

    elif GaussianInitializer == 0 and XavierGlorotUniform == 1 and XavierGlorotNormal == 0:

        dhidden_list = np.arange(dhidden_min, dhidden_max + dhidden_step / 10, dhidden_step, dtype=int)
        dhidden = dhidden_list[slurm_nr + 1000 * batch]

        width = 0.0 # ceremonial value only

    elif GaussianInitializer == 0 and XavierGlorotUniform == 0 and XavierGlorotNormal == 1:

        dhidden_list = np.arange(dhidden_min, dhidden_max + dhidden_step / 10, dhidden_step, dtype=int)
        dhidden = dhidden_list[slurm_nr + 1000 * batch]

        width = 0.0 # ceremonial value only



    
    print()
    print(f'1D chain of size {L}\n')
    print("Gaussian initializer:", GaussianInitializer)
    print("Xavier Glorot Uniform:", XavierGlorotUniform)
    print("Xavier Glorot Normal:", XavierGlorotNormal)
    print("dhidden =",dhidden)
    if GaussianInitializer == 1 and XavierGlorotUniform == 0 and XavierGlorotNormal == 0:
        print("width =",width)
    print()
    
    config = {
        # MODEL PARAMS
        'L': L,
        'numSpins_A': numSpins_A,
        'lattice': '1d',
        
        # VMC PARAMS
        'nsamples_training': nsamples_training,
        'nsamples_final_calculation': nsamples_final_calculation,
        'chunk_size': chunk_size,
        'model': 'rnn',
        'bias': bias,
        'dhidden': dhidden,
        'weight_sharing': weight_sharing,
        'rnn_cell': rnn_cell,
        
        'activation_fn': activation_fn,
        'gate_fn': gate_fn,
        'softmaxOnOrOff': softmaxOnOrOff,
        'modulusOnOrOff': modulusOnOrOff,
        'signOfProbsIntoPhase': signOfProbsIntoPhase,
        
        'autoreg': autoreg,

        'GaussianInitializer': GaussianInitializer,
        'XavierGlorotUniform': XavierGlorotUniform,
        'XavierGlorotNormal': XavierGlorotNormal,

        'width': width,
        'numrealizations': numrealizations,
        'lr_schedule': lr_schedule,
        'lr': lr,
        'lr_decay_rate': lr_decay_rate,
        'SR_diag_shift': SR_diag_shift,
        'n_iter': n_iter,
        'ExactDiag': ExactDiag,
        'Gamma': Gamma,
        'J': J,
        'J_Heis': J_Heis,
        'constrain_total_sz': constrain_total_sz,
        'Delta': Delta,
        'Hamiltonian': Hamiltonian,
        'PBC_or_OBC': PBC_or_OBC,
        'SR_or_Adam': SR_or_Adam,
        'positive_or_complex': positive_or_complex,
        
        'sampler': sampler,
        
        # Quantities needed for MCMC
        'n_chains': n_chains,
        'n_discard_per_chain': n_discard_per_chain,
        'sweep_size': sweep_size,
        
        'exactOrMC': exactOrMC,
        
        'rel_error_conv': rel_error_conv,
        'var_conv': var_conv,
        'exp_smoothing_factor': exp_smoothing_factor,
        'show_progress': show_progress,

        'dtype': dtype,
        }

    main(config)
    
    print("------------")
    print("Execution time = --- %s seconds ---" % (time.time() - start_time))
