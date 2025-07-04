#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:26:33 2024


@author: andrewjreissaty91
"""

# This is the code I am writing for VMC with the TFIM. We will use the 1D RNN for now as our model.
import jax
from jax import numpy as jnp
from jax import random
from jax.nn.initializers import normal, glorot_uniform, glorot_normal, zeros, constant
import optax
from netket.nn.masked_linear import default_kernel_init
from functools import partial
#from models.utils import states_to_local_indices
import itertools as it

import netket as nk
from netket.operator.spin import sigmax,sigmay,sigmaz  # an instance of the class nk.operator.LocalOperator

import networkx

from models.rnn1d import RNN1D
#from models.rnn2d import RNN2D
#from plot import plot_qgt_matrix, plot_qgt_spectrum
#import matplotlib.pyplot as plt
import numpy as np
import string

#from matplotlib import pyplot as plt

#from scipy.linalg import ishermitian
import scipy

from scipy.sparse.linalg import eigsh

from flax.linen.activation import sigmoid, tanh

import os

import time


# Following the netket VMC prescription:

def main(config):
    #parser.add_argument('--lattice', default='1d')
    lattice = config['lattice']
    L = config['L']
    numSpins_A = config['numSpins_A']
    dhidden = config['dhidden']

    GaussianInitializer = config['GaussianInitializer']
    XavierGlorotUniform = config['XavierGlorotUniform']
    XavierGlorotNormal = config['XavierGlorotNormal']

    width = config['width']
    weight_sharing = config['weight_sharing']
    autoreg = config['autoreg'] # just telling us if the model is autoregressive or not
    numrealizations = config['numrealizations']
    bias = config['bias']
    rnn_cell = config['rnn_cell']
    
    activation_fn = config['activation_fn']  # new argument to be able to specify activation functions in Vanilla RNN and GRU
    gate_fn = config['gate_fn']  # new argument to be able to specify gate function in GRU
    softmaxOnOrOff = config['softmaxOnOrOff']
    modulusOnOrOff = config['modulusOnOrOff']
    signOfProbsIntoPhase = config['signOfProbsIntoPhase']
    
    dtype = config['dtype']
    #SEED = config['seed']
    nsamples_training = config['nsamples_training'] # batch size, andrew 
    nsamples_final_calculation = config['nsamples_final_calculation']
    chunk_size = config['chunk_size']
    lr_schedule = config['lr_schedule']
    lr = config['lr']
    lr_decay_rate = config['lr_decay_rate']
    SR_diag_shift = config['SR_diag_shift']
    n_iter = config['n_iter']
    ExactDiag = config['ExactDiag'] # True or False

    # TFIM
    Gamma = config['Gamma'] # transverse field
    J = config['J'] # Ising coupling

    # Heisenberg
    J_Heis = config['J_Heis']
    constrain_total_sz = config['constrain_total_sz']

    # XXZ... UPDATE: not doing XXZ for now. 5 avril, 2025
    Delta = config['Delta']

    Hamiltonian = config['Hamiltonian'] # will be 'TFIM', 'MHS' or 'Heisenberg' for now. We may do the 'XXZ' model down the line.
    PBC_or_OBC = config['PBC_or_OBC']
    SR_or_Adam = config['SR_or_Adam']
    positive_or_complex = config['positive_or_complex']
    
    # New quantity: MCMC or ARD sampler
    ARD_vs_MCMC = config['sampler']
    
    # New quantities needed for MCMC:
    n_chains = config['n_chains']
    n_disc_chain = config['n_discard_per_chain']  # n_discard_per_chain
    sweep_size = config['sweep_size']
    
    exactOrMC = config['exactOrMC'] # 'exact' or 'MC' for sample generation... for VMC it will always be MC, since we are using Netket.
    
    rel_error_conv = config['rel_error_conv']
    var_conv = config['var_conv']
    exp_smoothing_factor = config['exp_smoothing_factor']
    
    show_progress = config['show_progress']
    
    # dtype for RNN parameters,
    dtype = config['dtype']
    #dtype_for_file_naming = dtype[10:]
    if dtype == 'jax.numpy.float16':
        dtype_for_file_naming = 'float16'
    elif dtype == 'jax.numpy.float32':
        dtype_for_file_naming = 'float32'
    elif dtype == 'jax.numpy.float64':
        dtype_for_file_naming = 'float64'
    

    
    #Gamma = 1
    if Hamiltonian == 'TFIM':
        # Hilbert space
        hilbert = nk.hilbert.Spin(s=1/2, N=L)
        H = sum([-Gamma*sigmax(hilbert,i) for i in range(L)]) # I think H is a local operator
        #J = 1
        if PBC_or_OBC == 'PBC':
            H += sum([-J*sigmaz(hilbert,i)*sigmaz(hilbert,(i+1)%L) for i in range(L)])
        elif PBC_or_OBC == 'OBC':
            H += sum([-J*sigmaz(hilbert,i)*sigmaz(hilbert,(i+1)%L) for i in range(L-1)])
    elif Hamiltonian == 'Heisenberg':
        if PBC_or_OBC == 'PBC':
            g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        elif PBC_or_OBC == 'OBC':
            g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

        # Define the Hilbert space based on this graph
        # We impose to have a fixed total magnetization of zero
        if constrain_total_sz:
            hilbert = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
        else:
            hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        # calling the Heisenberg Hamiltonian
        H = nk.operator.Heisenberg(hilbert=hilbert, graph=g, J=J_Heis,sign_rule=True)

    elif Hamiltonian == 'MHS': # Modified Haldane Shastry model
        hilbert = nk.hilbert.Spin(s=1/2, N=L)
        H = 0
        for k in range(L):
            for j in range(k):
                d_jk = L / np.pi * np.abs(np.sin((k-j)*np.pi/L))
                H += 1/(d_jk**2) * (-sigmax(hilbert,j)*sigmax(hilbert,k) - sigmay(hilbert,j)*sigmay(hilbert,k) + sigmaz(hilbert,j)*sigmaz(hilbert,k))
    
    
    
    
    print('----------')
    if Hamiltonian == 'TFIM':
        print('TFIM VMC')
        print('----------')
        print('h (transverse field) =',Gamma)
        print('J (Ising coupling) =',J)
        print('Boundary conditions:',PBC_or_OBC)
        print('----------')

    elif Hamiltonian == 'Heisenberg':
        print('Heisenberg model VMC')
        print('J (Heisenberg coupling) =', J_Heis)
        if constrain_total_sz:
            print('Total sz = 0')
        else:
            print('Total sz NOT constrained')
        print('----------')

    elif Hamiltonian == 'MHS':
        print('Modified Haldane-Shastry VMC')
        print('----------')




    print(f'1D chain of size {L}\n')
    #hilbert = nk.hilbert.Spin(1 / 2, L) # create the Hilbert space using the netket hilbert.Spin class

    if Hamiltonian == 'TFIM':
        filename_energies_exact = f'exact_energies_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                  + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) \
                                  + '_' + PBC_or_OBC + '.npy'

        filename_eigenstates_exact = f'exact_eigenstates_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                     + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) \
                                     + '_' + PBC_or_OBC + '.npy'
    elif Hamiltonian == 'Heisenberg':
        filename_energies_exact = f'exact_energies_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                  + '_J' + '{:.1f}'.format(J_Heis) + '_' + PBC_or_OBC + '.npy'

        filename_eigenstates_exact = f'exact_eigenstates_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                  + '_J' + '{:.1f}'.format(J_Heis) + '_' + PBC_or_OBC + '.npy'
    elif Hamiltonian == 'MHS':
        filename_energies_exact = f'exact_energies_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                  + '.npy'

        filename_eigenstates_exact = f'exact_eigenstates_lattice{lattice}_Ham{Hamiltonian}_L{L}' \
                                     + '.npy'


    
    #Print ExactDiag results first
    if L <= 24: # Let's assume we are only testing values of L that are multiples of 10 if L > 24. I think we can
                # do exact diagonalization at L = 24, pretty sure our computer can do it...
                # here we will accept all values of h/J. In general we will work with OBC's because our RNN
                # does not have PBC's encoded in it, but we could do PBC's
        if ExactDiag:
            if os.path.isfile(filename_energies_exact) and os.path.isfile(filename_eigenstates_exact):
                eig_vals = np.load(filename_energies_exact)
                eig_vecs = np.load(filename_eigenstates_exact)
                print("eigenvalues with scipy sparse:", eig_vals)
            else:
                print("no exact diag was previously performed, or we only have energies or eigenstates alone")
                print('Performing exact diagonalization.')
                sp_h = H.to_sparse()
                print("sp_h.shape =", sp_h.shape)
                # Since this is just a regular scipy sparse matrix, we can just use any sparse diagonalization routine in there to find the eigenstates.
                # For example, this will find the two lowest eigenstates
                eig_vals, eig_vecs = eigsh(sp_h, k=2,
                                           which="SA")  # which = which eigenvalues/vectors to find. 'SA’ : Smallest (algebraic) eigenvalues.
                print("eigenvalues with scipy sparse:", eig_vals)
                # Save in present directory
                np.save(filename_energies_exact, eig_vals)
                np.save(filename_eigenstates_exact, eig_vecs)
                # E_gs = eig_vals[0]
                # Now let's save the eigenvalues and eigenvectors in two files.

        else:
            if os.path.isfile(filename_energies_exact):
                eig_vals = np.load(filename_energies_exact)
            else:
                print("exact energies file not found, exiting simulation because ExactDiag(=0) not requested either")
                return 0

            if os.path.isfile(filename_energies_exact):
                eig_vecs = np.load(filename_eigenstates_exact)
            else:
                print("exact eigenstate file not found, exiting simulation because ExactDiag(=0) not requested either")
                return 0
            # Here's let's check if we have already done the exact diag. I want to save the eigenvectors and eigenvalues
            # if possible. We'll try to do this for L = 20 or something on the cluster.

    elif L > 24 and L % 10 == 0 and Gamma/J == 1.0 and PBC_or_OBC == 'OBC' and Hamiltonian == 'TFIM':
        # We will compare to DMRG results, but only if h/J == 1.0 and for multiples of 10 (I don't have h/J == 0.5 results here)
        # Right now, we don't have the exact energies for the Heisenberg model. May need to get them from DMRG or a
        # Jordan-Wigner Transformation. We stick to TFIM for now at the critical point for large systems.
        if ExactDiag:
            L_list_DMRG_OBC = [30,40,50,60,70,80,90,100,1000]
            eig_vals_L10_L1000_DMRG_OBC = [-37.8380982304,-50.5694337844,-63.3011891370,-76.0331561023,-88.7652446334,
                                       -101.4974094169,-114.2296251736,-126.9618766964,-1272.8762945220]

            assert(len(L_list_DMRG_OBC) == len(eig_vals_L10_L1000_DMRG_OBC))
            L_index = L_list_DMRG_OBC.index(L) # this will cause an error if we don't have the desired value of L
                                               # in the list already, e.g. L = 200.
            E0 = eig_vals_L10_L1000_DMRG_OBC[L_index]
            eig_vals = np.array([E0]) # we write the ground state energy in array form to not have
                                                        # to modify code later, because we use eig_vals[0] for the
                                                        # ground state
            print("DMRG ground state energy:", eig_vals)
        else:
            print("Chosen hyperparameters are OK, but can't compare to exact energies, because ExactDiag was set to 0.")
            return 0
    else:
        print("The code is not set up to deal with the chosen hyperparameters. Check, adjust and try again.")
        return 0


        
    
    #print(f'Saving data: {DATA}') # Ahh DATA tells us if we want to save the data or not, PLOT tells us if we want to plot or not.
    #print(f'Plotting: {PLOT}')
    print('_' * 30)
    print(f'Number of samples for training = {nsamples_training}')
    print(f'RNN Cell = {rnn_cell}')
    print(f'Bias = {bias}')
    print(f'RNN dhidden = {dhidden}')
    print(f'numrealizations = {numrealizations}')
    print(f'lr_schedule = {lr_schedule}')
    if lr_schedule:
        print('lr_decay_rate = ',lr_decay_rate)
    else:
        print('No decaying of learning rate')

    if Hamiltonian == 'TFIM' or Hamiltonian == 'Heisenberg':
        if ARD_vs_MCMC == 'ARD':
            if exactOrMC == 'MC':
                path_name = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming
        elif ARD_vs_MCMC == 'MCMC':
            if exactOrMC == 'MC':
                path_name = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming \
                            + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'


    elif Hamiltonian == 'MHS':
        if ARD_vs_MCMC == 'ARD':
            if exactOrMC == 'MC':
                path_name = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming
        elif ARD_vs_MCMC == 'MCMC':
            if exactOrMC == 'MC':
                path_name = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                            + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                            + f'numrealiz_{numrealizations}/dhidden_{dhidden}' + f'{f"_width_{width:.2f}/" if GaussianInitializer else "/"}' + dtype_for_file_naming \
                            + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'

    '''
    Checkpointing
    '''
    # if the files all exist, kill the execution:
    if GaussianInitializer:
        ckpt_path = './data_VMC_GaussianInit'
    elif XavierGlorotUniform:
        ckpt_path = './data_VMC_XavierGlorotUniform'
    elif XavierGlorotNormal:
        ckpt_path = './data_VMC_XavierGlorotNormal'

    if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
        if activation_fn == 'identity' or gate_fn == 'identity':
            if softmaxOnOrOff == 'on':
                ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
        else:  # neither activation_fn nor gate_fn is identity
            if softmaxOnOrOff == 'on':
                ckpt_path += f'/no_horiz_corr_VMC/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
    elif rnn_cell == 'RNN':
        if activation_fn == 'identity':
            if softmaxOnOrOff == 'on':
                ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
        else:  # neither activation_fn nor gate_fn is identity
            if softmaxOnOrOff == 'on':
                ckpt_path += f'/no_horiz_corr_VMC/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'




    if Hamiltonian == 'TFIM':
        if SR_or_Adam == 'Adam':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                       + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                       + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                       + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                            + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) \
                                            + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                            + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                            + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            # n_iter goes in the title but it will not be the simulation time- the simulation will be cut short
            # whenever the criteria for convergence have been achieved.
        elif SR_or_Adam == 'SR':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' + SR_or_Adam \
                                       + '_diagshift' + '{:.3f}'.format(SR_diag_shift) + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                       + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                       + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                            + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                            + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                            + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                            + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'




    elif Hamiltonian == 'Heisenberg':
        if SR_or_Adam == 'Adam':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                       + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                       + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                       + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                       + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                            + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                            + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) \
                                            + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                            + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                            + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

        elif SR_or_Adam == 'SR':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                       + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                       + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                       + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                       + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                       + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                            + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                            + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                            + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                            + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                            + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
            # n_iter goes in the title but it will not be the simulation time- the simulation will be cut short
            # whenever the criteria for convergence have been achieved.

    elif Hamiltonian == 'MHS':
        if SR_or_Adam == 'Adam':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_' + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                       + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                       + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_' + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) \
                                            + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                            + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                            + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
        elif SR_or_Adam == 'SR':
            filepath_conv_time_array = ckpt_path + '/convtime' + '_' + SR_or_Adam \
                                       + '_diagshift' + '{:.3f}'.format(SR_diag_shift) + '_lr' + '{:.0e}'.format(lr) \
                                       + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                       + '_niter' + '{:.0f}'.format(n_iter) \
                                       + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                       + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                       + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            filepath_conv_time_array_ckpt = ckpt_path + '/convtime_ckpt' + '_' + SR_or_Adam \
                                            + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                            + '_lr' + '{:.0e}'.format(lr) \
                                            + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                            + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                            + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                            + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
            # n_iter goes in the title but it will not be the simulation time- the simulation will be cut short
            # whenever the criteria for convergence have been achieved.


                                        
    
    filepath_config_txt = ckpt_path + '/config.txt'
    #filepath_rankFullness_qgt_array = ckpt_path + '/rankFullness_qgt_array.npy'

    # if directory doesn't exist, create it
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    #print(ckpt_path)
    
    # if the config file doesn't exist yet, 
    if not os.path.isfile(filepath_config_txt): 
        with open(filepath_config_txt, 'w') as file:
            for k, v in config.items():
                file.write(k + f'={v}\n') # saving the dictionary items of "config" in a file... \n is new line object.
                
    
    #conv_time_array = [] # will store convergence time for all the realizations
    
    # Before looping through the realizations: Check if the relevant files exist or not. If filepath_conv_time_array exists, then
    # we have all 100 realizations already completed, so there is no need to proceed.
    if os.path.isfile(filepath_conv_time_array):
        print()
        print("The desired convergence time file already exists.")
        print("dhidden =",dhidden)
        if GaussianInitializer:
            print("width = ",width)
        print("avg conv_time across realizations =", np.mean(np.load(filepath_conv_time_array), axis=0))
        print("Exiting Simulation")
        print()
        return 0
        
    # If we are here, there are at least some realizations / simulations we haven't completed yet. So now let's check where to start
    # the loop (i.e. what simulation we are at.) via the checkpointing file
    if os.path.isfile(filepath_conv_time_array_ckpt): # if statement to check if ckpt file exists (it should if simulations have
                                                      # been completed already
        conv_time_array = list(np.load(filepath_conv_time_array_ckpt))
        indexWhereToStart = len(conv_time_array) # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                                                 # so we want to start at i = 3, i.e. i = len(conv_time_array)
        print("indexWhereToStart before checking for HIGHER/LOWER NUMBER OF REALIZATIONS =",indexWhereToStart)
        print("Data already exists for some realizations (but not all). Loading the data.")
    else: # No simulations have been completed yet
        conv_time_array = [] # will store convergence time for all the realizations
        indexWhereToStart = 0
        print("indexWhereToStart before checking for HIGHER/LOWER NUMBER OF REALIZATIONS =", indexWhereToStart)
        print("No simulations have been performed. Start from scratch.")
    
    #indexWhereToStart = 0
    print()
    if indexWhereToStart == numrealizations:
        print("The simulation has already been completed. Nothing further will be run")
    else:
        print("Starting from realization =", indexWhereToStart)


    '''
    HIGHER # of realizations. CHECK FOLDERS WITH HIGHER NUMBER OF REALIZATIONS, to avoid having to recalculate
    '''
    found_higher = 0
    if (indexWhereToStart == 0):
        print()

        print("Actually wait. First checking to see if data for same hyperparameters but HIGHER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at HIGHER numbers of realizations, we will check for LOWER number of realizations.")
        for numrealiz_i in range(numrealizations + 500, numrealizations, -1):

            if Hamiltonian == 'TFIM' or Hamiltonian == 'Heisenberg':
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                    + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'


            elif Hamiltonian == 'MHS':
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                    + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'

            if GaussianInitializer:
                ckpt_path_i = './data_VMC_GaussianInit'
            elif XavierGlorotUniform:
                ckpt_path_i = './data_VMC_XavierGlorotUniform'
            elif XavierGlorotNormal:
                ckpt_path_i = './data_VMC_XavierGlorotNormal'

            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'

            if Hamiltonian == 'TFIM':
                if SR_or_Adam == 'Adam':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                                    + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                                    + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            elif Hamiltonian == 'Heisenberg':
                if SR_or_Adam == 'Adam':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                                    + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                                    + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                                    + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                                    + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'


            elif Hamiltonian == 'MHS':
                if SR_or_Adam == 'Adam':

                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_' + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_' + SR_or_Adam \
                                                    + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'






            if (os.path.isfile(filepath_conv_time_array_ckpt_i)):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                conv_time_array = list(np.load(filepath_conv_time_array_ckpt_i))
                if len(conv_time_array) >= numrealizations:
                    length_original = len(conv_time_array)
                    conv_time_array = conv_time_array[:numrealizations]
                    print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                          ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                    if length_original < numrealiz_i:
                        print("There are actually less data points recorded than numrealiz_i =", numrealiz_i,
                              "points, which is OK. It's still more data points than numrealizations =",
                              numrealizations)

                    conv_time_array = np.array(conv_time_array)
                    np.save(filepath_conv_time_array_ckpt, conv_time_array)

                    indexWhereToStart = len(conv_time_array)  # numrealizations
                    found_higher = 1
                    break  # break out of for loop and move forward.
                # If found == 0 no data at lower number of realizations has been found.
        if found_higher == 0:
            print("No data at HIGHER number of realizations has been found. Confirmed that we will now check for data at lower number of realizations.")
    

    '''
    LOWER number of realizations. Check for data to be extracted from simulations at lower numrealizations.
    '''
    if (indexWhereToStart == 0) and (found_higher == 0):
        print()
        print("NEXT. WE NOW CHECK to see if data for same hyperparameters but LOWER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at lower numbers of realizations, we will start from scratch.")
        found = 0
        for numrealiz_i in range(numrealizations - 1, 0, -1):

            if Hamiltonian == 'TFIM' or Hamiltonian == 'Heisenberg':
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + f'_{PBC_or_OBC}' + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                    + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            elif Hamiltonian == 'MHS':
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = Hamiltonian + '_RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    + f'{positive_or_complex}/L_{L}/{exactOrMC}_nsTrain{nsamples_training}_nsFinalCalc{nsamples_final_calculation}/' \
                                    + f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                    + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'

            if GaussianInitializer:
                ckpt_path_i = './data_VMC_GaussianInit'
            elif XavierGlorotUniform:
                ckpt_path_i = './data_VMC_XavierGlorotUniform'
            elif XavierGlorotNormal:
                ckpt_path_i = './data_VMC_XavierGlorotNormal'

            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i += f'/no_horiz_corr_VMC/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i += f'/no_horiz_corr_VMC_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'


            if Hamiltonian == 'TFIM':
                if SR_or_Adam == 'Adam':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                                    + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_h' + '{:.1f}'.format(Gamma) + '_J' + '{:.1f}'.format(J) + '_' \
                                                    + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'

            elif Hamiltonian == 'Heisenberg':
                if SR_or_Adam == 'Adam':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                                    + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                                    + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_J' + '{:.1f}'.format(J_Heis) + '_' \
                                                    + f'{"totalsz0_" if constrain_total_sz else ""}' \
                                                    + SR_or_Adam + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'



            elif Hamiltonian == 'MHS':
                if SR_or_Adam == 'Adam':

                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_' + SR_or_Adam + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) \
                                                    + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) + '_varconv' + '{:.0e}'.format(var_conv) \
                                                    + '_expsmoothfactor' + '{:.2f}'.format(exp_smoothing_factor) \
                                                    + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'
                elif SR_or_Adam == 'SR':
                    filepath_conv_time_array_ckpt_i = ckpt_path_i + '/convtime_ckpt' + '_' + SR_or_Adam \
                                                    + '_diagshift' + '{:.3f}'.format(SR_diag_shift) \
                                                    + '_lr' + '{:.0e}'.format(lr) \
                                                    + f'{f"_lrDecayRate{lr_decay_rate:.2f}" if lr_schedule else ""}' \
                                                    + '_niter' + '{:.0f}'.format(n_iter) + '_relerrorconv' + '{:.0e}'.format(rel_error_conv) \
                                                    + '_varconv' + '{:.0e}'.format(var_conv) + '_expsmoothingfactor' \
                                                    + '{:.2f}'.format(exp_smoothing_factor) + f'_nsFinalCalc{nsamples_final_calculation}' + '.npy'


            if (os.path.isfile(filepath_conv_time_array_ckpt_i)):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                conv_time_array = list(np.load(filepath_conv_time_array_ckpt_i))
                indexWhereToStart = len(conv_time_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                # so we want to start at i = 3, i.e. i = len(entropy_array)
                print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                      ", (same hyperparameters otherwise). Starting at realiz =",numrealiz_i)
                # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                found += 1
                break

            # If found == 0 no data at lower number of realizations has been found.
        if found == 0:
            print("No data at lower number of realizations has been found. Confirmed that we are starting from scratch, realization = 0.")

    # ACTIVATION AND GATE FUNCTIONS TO PASS TO THE MODEL.

    if activation_fn == 'tanh':  # this will be default
        activation_function = tanh
        # print("3ainsssssssssss")
    elif activation_fn == 'sigmoid':
        activation_function = sigmoid
    elif activation_fn == 'identity':  # linear activation function
        activation_function = (lambda x: x)

    if gate_fn == 'sigmoid':
        gate_function = sigmoid
        # print("3ainsssssssssss GRU")
    elif gate_fn == 'tanh':
        gate_function = tanh
    elif gate_fn == 'identity':
        gate_function = (lambda x: x)


    if GaussianInitializer == 1 and XavierGlorotUniform == 0 and XavierGlorotNormal == 0:
        if dtype == 'jax.numpy.float16':  # Using these if statements because don't know code to convert string into a jax.numpy.float.
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=normal(stddev=width),
                          recurrent_kernel_init=normal(stddev=width),
                          bias_init=normal(stddev=width),
                          param_dtype=jax.numpy.float16,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float32':
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=normal(stddev=width),
                          recurrent_kernel_init=normal(stddev=width),
                          bias_init=normal(stddev=width),
                          param_dtype=jax.numpy.float32,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float64':
            # param_dtype = jax.numpy.float64
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=normal(stddev=width),
                          recurrent_kernel_init=normal(stddev=width),
                          bias_init=normal(stddev=width),
                          param_dtype=jax.numpy.float64,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D() =", dtype)

    elif GaussianInitializer == 0 and XavierGlorotUniform == 1 and XavierGlorotNormal == 0:
        if dtype == 'jax.numpy.float16':  # Using these if statements because don't know code to convert string into a jax.numpy.float.
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_uniform(),
                          recurrent_kernel_init=glorot_uniform(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float16,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float32':
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_uniform(),
                          recurrent_kernel_init=glorot_uniform(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float32,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float64':
            # param_dtype = jax.numpy.float64
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_uniform(),
                          recurrent_kernel_init=glorot_uniform(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float64,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D() =", dtype)

    elif GaussianInitializer == 0 and XavierGlorotUniform == 0 and XavierGlorotNormal == 1:

        if dtype == 'jax.numpy.float16':  # Using these if statements because don't know code to convert string into a jax.numpy.float.
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_normal(),
                          recurrent_kernel_init=glorot_normal(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float16,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float32':
            # param_dtype = jax.numpy.float32
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_normal(),
                          recurrent_kernel_init=glorot_normal(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float32,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D()  =", dtype)
            # print("param_dtype inside numrealizations loop =",param_dtype)
        elif dtype == 'jax.numpy.float64':
            # param_dtype = jax.numpy.float64
            model = RNN1D(hilbert=hilbert,
                          # these models can take hilbert as an input because of the AbstractARNN subclass
                          L=L,
                          numSpins_A=numSpins_A,
                          dhidden=dhidden,
                          use_bias=bias,
                          rnn_cell_type=rnn_cell,
                          kernel_init=glorot_normal(),
                          recurrent_kernel_init=glorot_normal(),
                          bias_init=constant(0),
                          param_dtype=jax.numpy.float64,
                          positive_or_complex=positive_or_complex,
                          activation_fn=activation_function,
                          gate_fn=gate_function,
                          softmaxOnOrOff=softmaxOnOrOff,
                          modulusOnOrOff=modulusOnOrOff,
                          signOfProbsIntoPhase=signOfProbsIntoPhase)
            print("dtype argument to RNN1D() =", dtype)


        

    
    DEVICE = jax.devices()[-1].device_kind # jax.devices(backend=None)[source]: Returns a list of all devices for a given backend.
    print("DEVICE (Roeland's way) =", DEVICE)
    print("jax.devices()[0].platform =",jax.devices()[0].platform)
    print("jax.devices() =", jax.devices())


    if lr_schedule:
        lr_function = optax.schedules.exponential_decay(init_value=lr, transition_steps=n_iter, decay_rate=lr_decay_rate, transition_begin=0, staircase=False, end_value=None)
        print("lr_function(np.arange(n_iter))=",lr_function(np.arange(n_iter)))
        if SR_or_Adam == 'SR':
            print("Stochastic Reconfig initiated with diag_shift =",SR_diag_shift)
            optimizer = nk.optimizer.Sgd(learning_rate=lr_function)
        elif SR_or_Adam == 'Adam':
            print("Adam optimization initiated")
            optimizer = nk.optimizer.Adam(learning_rate=lr_function)
    else:
        if SR_or_Adam == 'SR':
            print("Stochastic Reconfig initiated with diag_shift =",SR_diag_shift)
            optimizer = nk.optimizer.Sgd(learning_rate=lr)
        elif SR_or_Adam == 'Adam':
            print("Adam optimization initiated")
            optimizer = nk.optimizer.Adam(learning_rate=lr)

    # Now defining the sampler outside the loop
    if ARD_vs_MCMC == 'ARD':
        sampler = nk.sampler.ARDirectSampler(hilbert)  # autoregressive sampler of netket
    elif ARD_vs_MCMC == 'MCMC':
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, sweep_size=sweep_size)

    print("Softmax:",softmaxOnOrOff)
    print("Modulus:",modulusOnOrOff)
    print("signOfProbsIntoPhase:",signOfProbsIntoPhase)

    for i in range(indexWhereToStart,numrealizations):
        # No calculations have been done yet
        #start = time.time()
        # SEED = i

        if GaussianInitializer:
            SEED = int(L * dhidden * width * 312000 + i)
            SAMPLER_SEED = int(L * dhidden * width ** 3 * 32 * 902.48 + i)  # Also new sampler seed for every new wavefunction.
        elif XavierGlorotUniform or XavierGlorotNormal:
            SEED = int(L * dhidden * 10.234427294 * 312000 + i)
            SAMPLER_SEED = int(L * dhidden * 10.234427294 ** 3 * 32 * 902.48 + i)  # Also new sampler seed for every new wavefunction.

        if ARD_vs_MCMC == 'ARD':
            vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=nsamples_training,
                                    chunk_size=chunk_size,seed = random.key(SEED),sampler_seed=random.key(SAMPLER_SEED))
        elif ARD_vs_MCMC == 'MCMC':
            vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=nsamples_training,
                                    n_discard_per_chain=n_disc_chain,
                                    chunk_size=chunk_size, seed = random.key(SEED), sampler_seed=random.key(SAMPLER_SEED))

        if i == indexWhereToStart:
            print()
            print(f"Number of parameters {vstate.n_parameters}")
            print()

        if SR_or_Adam == 'SR':
            #optimizer = nk.optimizer.Sgd(learning_rate=lr)
            gs = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=SR_diag_shift))
        elif SR_or_Adam == 'Adam':
            #optimizer = nk.optimizer.Adam(learning_rate=lr)
            gs = nk.driver.VMC(H, optimizer, variational_state=vstate)
        # diag_shift = 0.1 in Netket tutorial.
        # nk.driver.VMC: Class that performs energy minimization using Variational Monte Carlo (VMC).
        
        # nk.logging.RuntimeLog():
        # This logger accumulates log data in a set of nested dictionaries which are stored in memory. The log data is not automatically saved to
        # the filesystem. It can be passed with keyword argument out to Monte Carlo drivers in order to serialize the output data of the simulation.
        # This logger keeps the data in memory, and does not save it to disk. To serialize the current content to a file, use the method serialize().
        print()
        print("realization =",i)
        log=nk.logging.RuntimeLog()
        #callback_r = lambda step, log, driver: print(driver.learning_rate)
        #gs.run(n_iter=n_iter,out=log, show_progress=show_progress, callback = (callback_r))
        gs.run(n_iter=n_iter,out=log, show_progress=show_progress, callback = Callback(var_conv,rel_error_conv,exp_smoothing_factor,eig_vals[0],L))
        # print("optimizer =",optimizer)
        data_NQS = log.data

        '''
        plt.errorbar(
            data_NQS["Energy"].iters,
            data_NQS["Energy"].Mean,
            yerr=data_NQS["Energy"].Sigma,
        )

        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.show()
        '''
        
        # append to array
        conv_time_array.append(data_NQS["Energy"].iters[-1]) # index of last iteration
        
        # Save checkpointing file
        np.save(filepath_conv_time_array_ckpt,conv_time_array)
        
        # Save the checkpointing file

        '''
        Ok now we are here, let us evaluate the expection values we want using a large number of samples. Matija told me how to do this.
        And let's print
        '''
        vstate.n_samples = nsamples_final_calculation # Want a big number of samples for final observable estimations
        vstate.reset() # Matija said to do this after resetting n_samples and before using expect.

        nqs_energy=vstate.expect(H)
        print()
        print("step at which VMC stops =", data_NQS["Energy"].iters[-1])
        print()
        print("eigenvalues with scipy sparse:", eig_vals)
        print("optimized ground state energy (including complex part) = ",nqs_energy)
        #print()
        print("optimized ground state energy (real) = ",nqs_energy.mean.real)
        print("optimized ground state error = ",nqs_energy.error_of_mean)
        print("optimized ground state variance = ",nqs_energy.variance/L)
        print("optimized relative error, comp to true ground state = ",abs((nqs_energy.mean.real - eig_vals[0])/eig_vals[0]))
        #print()

      
    # OK Now, we want to save the conv time data. Should I load it from the checkpointing file or not? Let's do it from the checkpointing
    # file. In case we complete the 100th simulation and the Vector cluster fucks us before saving the main conv_time_array file, let's pull
    # up the data from the checkpointing file rather than relying on the conv_time_array being modified during the above loop
    conv_time_array = np.load(filepath_conv_time_array_ckpt)
    print("avg conv_time across realizations =",np.mean(conv_time_array,axis=0))
    np.save(filepath_conv_time_array,conv_time_array)
    return 0

# Class needed to stop the optimization whenever certain criteria are met.
class Callback:
    
    def __init__(self,var_cutoff, rel_error_cutoff, exp_smoothing_factor, gs_energy_exact,L):
        '''
        var_cutoff = variance cutoff to end optimization
        
        rel_error_cutoff = relative error cutoff to end optimization
        
        x = exponential smoothing factor for exponentially moving average
        
        gs_energy_exact = desired exact ground state energy for comparison
        
        L = size of 1d chain
        '''
        
        self.var_conv = var_cutoff
        self.rel_error_conv = rel_error_cutoff
        self.x = exp_smoothing_factor
        self.gs_energy_exact = gs_energy_exact
        self.L = L
        #print("3ainssssss")
        
        
    def __call__(self,step,log_data,driver):
        '''
        FUNCTION: Pretty sure this call function will be called at every step of the VMC process. It has to return True or False.
        
        step = step in the VMC process that we are currently at
        
        log_data = nk.logging.RuntimeLog() object for now, otherwise it has to be some dictionary that logs the data
        
        driver = I guess this is a netket.driver.VMC object? Or a netket.driver object? Anyways, I think the Netket API is set up
                 to handle whatever step, log_data and driver are
        '''
        #if step == 0:
        #    print("CHIRURGIEEEYNNNN")
        energy = log_data["Energy"].Mean.real
        #arr_energy = np.reshape(arr_energy,(1))
        rel_error_energy = abs((energy - self.gs_energy_exact)/self.gs_energy_exact)
        variance = log_data["Energy"].Variance

        #print("lr @ step ", step," =", self.optimizer)
        
        if (rel_error_energy <= self.rel_error_conv) and (variance/self.L <= self.var_conv):
            print()
            print("Criteria for convergence: SATISFIED BABY")
            print("Current iteration (step):", step)
            print()
            return False # this stops the optimization
        
        # return True to keep the optimization going in case the if statement above fails.
        return True


# Class needed to stop the optimization whenever certain criteria are met... Exponential moving aaverage.
class Callback_Ema:
    
    def __init__(self,var_cutoff, rel_error_cutoff, exp_smoothing_factor, gs_energy_exact,L):
        '''
        var_cutoff = variance cutoff to end optimization
        
        rel_error_cutoff = relative error cutoff to end optimization
        
        x = exponential smoothing factor for exponentially moving average
        
        gs_energy_exact = desired exact ground state energy for comparison
        
        L = size of 1d chain
        '''
        
        self.var_conv = var_cutoff
        self.rel_error_conv = rel_error_cutoff
        self.x = exp_smoothing_factor
        self.gs_energy_exact = gs_energy_exact
        self.L = L
        print("3ainssssss")
        
        
    def __call__(self,step,log_data,driver):
        '''
        FUNCTION: Pretty sure this call function will be called at every step of the VMC process. It has to return True or False.
        
        step = step in the VMC process that we are currently at
        
        log_data = nk.logging.RuntimeLog() object for now, otherwise it has to be some dictionary that logs the data
        
        driver = I guess this is a netket.driver.VMC object? Or a netket.driver object? Anyways, I think the Netket API is set up
                 to handle whatever step, log_data and driver are
        '''

        
        '''
        Moving average for energy
        '''
        #print("CHIEYYYYYNNNN   0")
        arr_energy = log_data["Energy"].Mean
        arr_energy = np.reshape(arr_energy,(1))
        #print("arr_energy =")
        i = 1  
        # Initializing an empty list to put the EMA values  
        moving_averages_energy = []  
        
        #print("CHIEYYYYYNNNN   0.5")
            
        # Inserting the first exponential moving average in the list  
        moving_averages_energy.append(arr_energy[0])  

        # Looping through the elements of the array  
        while i < len(arr_energy):  
            
            # Calculating the exponential moving average using the formula we stated  
            average = round(self.x * (arr_energy[i] - moving_averages_energy[-1]) + moving_averages_energy[-1], 2)  
                
            # Storing the cumulative average of the current window of elements in the moving_averages list  
            moving_averages_energy.append(average)  
                
            # Shifting the window to the right by one index  
            i += 1
        
        rel_error_energy = abs((moving_averages_energy[-1] - self.gs_energy_exact) / self.gs_energy_exact)

        '''
        Moving average for variance
        '''
        arr_variance = log_data["Energy"].Variance
        arr_variance = np.reshape(arr_variance,(1))
        i = 1  
        # Initializing an empty list to put the EMA values  
        moving_averages_variance = []  
            
        # Inserting the first exponential moving average in the list  
        moving_averages_variance.append(arr_variance[0])  
            
        # Looping through the elements of the array  
        while i < len(arr_variance):  
            
            # Calculating the exponential moving average using the formula we stated  
            average = round(self.x * (arr_variance[i] - moving_averages_variance[-1]) + moving_averages_variance[-1], 2)  
                
            # Storing the cumulative average of the current window of elements in the moving_averages list  
            moving_averages_variance.append(average)  
                
            # Shifting the window to the right by one index  
            i += 1
        
        #rel_error = abs((moving_averages_variance[-1] - self.gs_energy_exact) / self.gs_energy_exact)
        
        
        if (rel_error_energy <= self.rel_error_conv) and (moving_averages_variance[-1]/self.L <= self.var_conv):
            print()
            print("Criteria for convergence: SATISFIED BABY")
            print("Current iteration (step):", step)
            print()
            return False # this stops the optimization
        
        # return True to keep the optimization going in case the if statement above fails.
        return True