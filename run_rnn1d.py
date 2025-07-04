#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 01:09:26 2024

@author: andrewjreissaty91
"""

import jax
import numpy as np
import time
import itertools as it
from sample_rnn1d import main
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
    parser.add_argument('--L_power_of_two_or_fixed_interval', default='fixed')
    # if we want to do fixed interval, then L_min and L_max are even fixed interval limits with an even L_step value.
    parser.add_argument('--L_min', default=5)
    parser.add_argument('--L_max', default=5)
    parser.add_argument('--L_step', default=2)
    # if we want to do powers of two, then we need arguments to specify the min and max power of 2 we will test.
    parser.add_argument('--L_min_power',default=7)
    parser.add_argument('--L_max_power',default=8)
    # May need a chunking threshold of L above which we will use chunking for the entropy calculation
    parser.add_argument('--L_chunk_threshold', default=60)

    #parser.add_argument('--numSpins_A', default=10)
    parser.add_argument('--dhidden_min', default=20)
    parser.add_argument('--dhidden_max', default=20)
    parser.add_argument('--dhidden_step', default=1)
    parser.add_argument('--numsamples', default=3) # default = 960 if MCMC, with chunk_size = 16. def = 1000 if ARD, w/ chunk_s = 100.
    parser.add_argument('--chunk_size',default=1000) 
    parser.add_argument('--numrealizations', default=3) # number of initializations of the wavefunction per data point
    
    # Adding activation function options, to test effect of linearization on entanglement entropy scaling
    parser.add_argument('--rnn_cell', default='RNN')
    parser.add_argument('--activation_fn', default='tanh')
    parser.add_argument('--gate_fn', default='sigmoid')
    parser.add_argument('--softmaxOnOrOff', default='on') # on/off = turn on or off the softmax and go from there.
    parser.add_argument('--modulusOnOrOff', default='off') # on/off = turn on or off the modulus function. Can only be on when
                                                           # softmax function is off, but can be off when softmax function is on or
                                                           # off.
    parser.add_argument('--signOfProbsIntoPhase', default='off') # if softmax is Off and modulus is Off, do we want (yes or no, i.e on or off) to incorporate the sign
                                                                 # of the conditional probs into the phase or not.
                                                                 # Default will be to have softmax on, so this will only matter if the
                                                                 # softmax and modulus function are both off.
    #parser.add_argument('--fullyPropagatedError', default='off') # if this is 'on', we propagate all the errors to estimate
                                                                 # the true error bars on the entropy and correlation
                                                                 # function data, even if the RNN sampling error can be
                                                                 # ignored. Assume this argument applies to both
                                                                 # entropy and correlation functions.
    parser.add_argument('--purity_correlation_sampling_errors', default=0) # 1 = calculate and store RNN sampling errors for purity and correlation functions.
    parser.add_argument('--purityLogSumExp_correlation_sampling_errors', default=0) # use LogSumExp to calculate purity er UPDATE 11/02/2025: just gonna calculate the logExpSum from now on when this is one, not the entropy or purity


    parser.add_argument('--width_min', default=0.6)
    parser.add_argument('--width_max', default=0.6)
    parser.add_argument('--width_step', default=0.5)
    parser.add_argument('--extra_widths',default = "")
    #parser.add_argument('--seed', default=0)
    parser.add_argument('--bias', default=1) # Using 1 or 0 to define True or False
    parser.add_argument('--weight_sharing', default=1)
    parser.add_argument('--autoreg', default=1)
    parser.add_argument('--data', default=1)
    parser.add_argument('--plot', default=0)
    parser.add_argument('--show', default=0)
    parser.add_argument('--exactOrMC', default='MC')
    parser.add_argument('--positive_or_complex', default='complex')
    
    # New argument: Numpy or Jax function to calculate the entropy, choose one or the other.
    parser.add_argument('--numpy_or_jax', default='jax') # 'numpy' if numpy, 'jax' if jax.
    parser.add_argument('--sampler', default = 'ARD') # 'ARD' or 'MCMC'
    
    # Quantities needed for MCMC, assuming sampler is chosen to be 'MCMC'. Can input
    # these quantities anyways, and only use them if MCMC is invoked.
    parser.add_argument('--n_chains', default=480) # was originally 480
    parser.add_argument('--n_discard_per_chain', default= 60) # was originally 20
    parser.add_argument('--sweep_size', default= 60) # was originally 160

    parser.add_argument('--slurm_nr', default=0)
    #New argument
    parser.add_argument('--batch', default=0)
    
    
    # WHAT TO CALCULATE: if quantity == 1, then calculate that property, if quantity == 0, then we are not calculating it.
    parser.add_argument('--calcEntropy',default=1)
    parser.add_argument('--calcEntropyCleanUp',default=0) # new input to direct "clean up" of shitty purity values caused by what I think can only be faulty compilation or machine issues.
    parser.add_argument('--calcRankQGT',default=0)
    parser.add_argument('--calcEntangSpectrum',default=0)
    parser.add_argument('--calcAdjacentGapRatio',default=0)
    
    # WHAT TO CALCULATE: AREA-LAW VOLUME-LAW Investigation
    parser.add_argument('--calcEntropy_scaling',default=0)

    # WHAT TO CALCULATE: Correlation function
    parser.add_argument('--calcCorr_1d_specific_dist',default=0) # 0 or 1. Specific distance correlation function
    parser.add_argument('--calcCorr_1d_upTo_dist', default=0) # 0 or 1. Calculating correlation UP TO a specific distance.
                                                                              # i.e. if we say up to a distance of 2, means calculate
                                                                              # nearest neighbor and next nearest neighbor correlations
                                                                              # this is the function we are most interested in to investigate the
                                                                              # properties of these random states.
    
    
    # If we want correlations between spins a specific distance apart, specify the distance
    parser.add_argument('--d_specific_corr', default = 1) # distance = 1 means nearest neighbor. This function will only be used if
                                                          # calcCorrelation_1d_specific_dist = 1
                                                          # We will mainly be interested in the "upTo" correlations.
                                                          
    parser.add_argument('--d_up_to_corr', default = 5) # distance = 1 means nearest neighbor. For L = 10, we will go to 5 as
                                                       # they do in the MBL Huse/Pal paper
    parser.add_argument('--newSamplesOrNot_corr', default=0)
    # if == 1, let's produce new samples for every expectation value in the correlation function for each realiz. Every single expectation value (there are many, over site pairs, etc.)
    # if == 0, let's use the same set of samples for every expectation value... by doing phi.reset() at the start of the for loop over realizations.
    
    # tolerance for SVD for rank calculation
    #parser.add_argument('--tol_rank_calc',default=1e-3)
    parser.add_argument('--tol_rank_calc',default=0) #IF tol_rank_cal == 0, then don't input tol_rank_cal into                                                                                  
                                                     #qgt rank calculation function in sample_rnn1d.py file.  
    
    # dtype for RNN
    parser.add_argument('--dtype',default='jax.numpy.float64')
    
    # New arguments: to calculate the density matrix Pauli String expansion coefficients
    # We will always just calculate this when we are not calculating anything else.
    parser.add_argument('--calcFourierTransform_rho',default=0)  # full pure state density matrix
    parser.add_argument('--calcFourierTransform_rhoA',default=0) # reduced density matrix of subsystem A
    parser.add_argument('--max_kBody_term',default=4)
    
    args = parser.parse_args()
    model = str(args.model)
    lattice = str(args.lattice)
    # New hyperparameter to decide if we want to do powers of 2 for L or not.
    L_power_of_two_or_fixed_interval = str(args.L_power_of_two_or_fixed_interval) # 'fixed' or 'power'
    L_min = int(args.L_min)
    L_max = int(args.L_max)
    L_step = int(args.L_step)
    L_min_power = int(args.L_min_power)
    L_max_power = int(args.L_max_power)
    L_chunk_threshold = int(args.L_chunk_threshold)
    #numSpins_A = int(args.numSpins_A)
    dhidden_min = int(args.dhidden_min)
    dhidden_max = int(args.dhidden_max)
    dhidden_step = int(args.dhidden_step)
    numsamples = int(args.numsamples)
    chunk_size = int(args.chunk_size)
    numrealizations= int(args.numrealizations)
    
    # RNN cell type + activation function + gate function (gate function used for GRU)
    rnn_cell = str(args.rnn_cell)
    activation_fn = str(args.activation_fn)
    gate_fn = str(args.gate_fn)
    softmaxOnOrOff = str(args.softmaxOnOrOff)
    modulusOnOrOff = str(args.modulusOnOrOff)
    signOfProbsIntoPhase = str(args.signOfProbsIntoPhase)
    purity_correlation_sampling_errors = bool(int(args.purity_correlation_sampling_errors))
    purityLogSumExp_correlation_sampling_errors = bool(int(args.purityLogSumExp_correlation_sampling_errors))
    #fullyPropagatedError = str(args.fullyPropagatedError)
    
    
    
    width_min = float(args.width_min)
    width_max = float(args.width_max)
    width_step = float(args.width_step)
    extra_widths = str(args.extra_widths)
    bias= bool(int(args.bias))
    weight_sharing= bool(int(args.weight_sharing))
    autoreg = bool(int(args.autoreg))
    data = bool(int(args.data))
    plot = bool(int(args.plot))
    show = bool(int(args.show))
    exactOrMC = str(args.exactOrMC)
    positive_or_complex = str(args.positive_or_complex)
    
    slurm_nr = int(args.slurm_nr) # should be from 0-99
    batch = int(args.batch)
    
    calcEntropy = bool(int(args.calcEntropy))
    calcEntropyCleanUp = bool(int(args.calcEntropyCleanUp))
    calcRankQGT = bool(int(args.calcRankQGT))
    calcEntangSpectrum = bool(int(args.calcEntangSpectrum))
    calcAdjacentGapRatio = bool(int(args.calcAdjacentGapRatio))
    
    # Volume Law - Area Law analysis. Entanglement scaling law.
    calcEntropy_scaling = bool(int(args.calcEntropy_scaling))
    
    # New quantities: Correlations
    calcCorr_1d_specific_dist = bool(int(args.calcCorr_1d_specific_dist))
    calcCorr_1d_upTo_dist = bool(int(args.calcCorr_1d_upTo_dist))
    d_specific_corr = int(args.d_specific_corr)
    d_up_to_corr = int(args.d_up_to_corr)
    newSamplesOrNot_corr = bool(int(args.newSamplesOrNot_corr))

    
    # tolerance for jax.numpy.matrix_rank calculation
    tol_rank_calc = float(args.tol_rank_calc)
    print("tol_rank_calc =",tol_rank_calc)
    
    # dtype for RNN parameters
    dtype = str(args.dtype)
    
    # NEW QUANTITIES
    # Fourier Transform - related quantities
    calcFourierTransform_rho = bool(int(args.calcFourierTransform_rho))
    calcFourierTransform_rhoA = bool(int(args.calcFourierTransform_rhoA))
    max_kBody_term = int(args.max_kBody_term)
    
    # New quantity: numpy or jax
    numpy_or_jax = str(args.numpy_or_jax)
    
    # New quantity: MCMC or ARD sampler
    sampler = str(args.sampler)
    
    # New quantities needed for MCMC
    n_chains = int(args.n_chains)
    n_discard_per_chain = int(args.n_discard_per_chain)
    sweep_size = int(args.sweep_size)
    
    
    dhidden_list = np.arange(dhidden_min,dhidden_max+dhidden_step/10,dhidden_step,dtype=int)
    width_list = np.arange(width_min,width_max+width_step/10,width_step,dtype=float)
    
    # Extra_widths analysis:
    #print("extra_widths =",extra_widths,type(extra_widths))
    if len(extra_widths) == 1:
        extra_widths = float(extra_widths)
        #print("extra_widths =",extra_widths)
        width_list = np.append(width_list,extra_widths)
        width_list = np.sort(width_list)
    elif len(extra_widths) > 1:
        extra_widths = extra_widths.split(",") # now you should have a list of strings
        #print("extra_widths =",extra_widths,type(extra_widths))
        extra_widths = [float(s) for s in extra_widths]
        width_list = np.append(width_list,extra_widths)
        width_list = np.sort(width_list)

    if L_power_of_two_or_fixed_interval == 'fixed':
        L_list = np.arange(L_min,L_max+L_step/10,L_step,dtype=int)
        print("L_list fixed interval =",L_list)
    elif L_power_of_two_or_fixed_interval == 'power': # power of two
        L_list = 2**np.arange(L_min_power,L_max_power+1)
        print("L_list power_of_two =",L_list)
    
    dhidden_width_L_list = list(it.product(dhidden_list, width_list, L_list))
    dhidden = dhidden_width_L_list[slurm_nr + 1000*batch][0]
    width = dhidden_width_L_list[slurm_nr + 1000*batch][1]
    L = dhidden_width_L_list[slurm_nr + 1000*batch][2]
    L = int(L) # was getting an assertion error without this. Need this now.
    
    print()
    print(f'1D chain of size {L}\n')
    print("dhidden =",dhidden)
    print("width =",width)
    print()
    
    if calcCorr_1d_specific_dist == True:
        print("Calculating (avg) specific correlation at distance = ",d_specific_corr)
    
    if calcCorr_1d_upTo_dist == True:
        print("Calculating (avg) correlations UP TO distance = ",d_up_to_corr)
    
    config = {
        # MODEL PARAMS
        'L': L,
        'L_chunk_threshold': L_chunk_threshold,
        #'numSpins_A': numSpins_A,
        'lattice': '1d',
        
        # VMC PARAMS
        'nsamples': numsamples,
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
        'purity_correlation_sampling_errors': purity_correlation_sampling_errors,
        'purityLogSumExp_correlation_sampling_errors': purityLogSumExp_correlation_sampling_errors,
        #'fullyPropagatedError': fullyPropagatedError,
        
        'autoreg': autoreg,
        'width': width,
        'numrealizations': numrealizations,
        'exactOrMC': exactOrMC,
        
        # pRNN or cRNN
        'positive_or_complex': positive_or_complex,
        
        # What To Calculate
        'calcEntropy': calcEntropy,
        'calcEntropyCleanUp': calcEntropyCleanUp,
        'calcRankQGT': calcRankQGT,
        'calcEntangSpectrum': calcEntangSpectrum,
        'calcAdjacentGapRatio': calcAdjacentGapRatio,
        
        # Entanglement Scaling Law analysis
        'calcEntropy_scaling': calcEntropy_scaling,
        
        # What To Calculate: Correlations
        'calcCorr_1d_specific_dist': calcCorr_1d_specific_dist,
        'calcCorr_1d_upTo_dist': calcCorr_1d_upTo_dist,
        
        # Correlation-related arguments
        'd_specific_corr': d_specific_corr,
        'd_up_to_corr': d_up_to_corr,
        'newSamplesOrNot_corr': newSamplesOrNot_corr,
        
        # tolerance for rank calculation
        'tol_rank_calc': tol_rank_calc,
        
        # dtype for RNN parameters,
        'dtype': dtype,
        
        # META PARAMS
        #'SEED': 0
        'DATA': data,
        'PLOT': plot,
        'SHOW': show,
        
        # Quantities for Fourier Transform calculation
        'calcFourierTransform_rho': calcFourierTransform_rho,
        'calcFourierTransform_rhoA': calcFourierTransform_rhoA,
        'max_kBody_term': max_kBody_term,
        
        # Numpy or Jax
        'numpy_or_jax': numpy_or_jax,
        
        # ARD or MCMC sampler
        'sampler': sampler,
        
        # Quantities needed for MCMC
        'n_chains': n_chains,
        'n_discard_per_chain': n_discard_per_chain,
        'sweep_size': sweep_size,
        }

    main(config)

    print("------------")
    print("Execution time = --- %s seconds ---" % (time.time() - start_time))