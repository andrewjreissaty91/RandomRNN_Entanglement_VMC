import jax
from jax import numpy as jnp
from jax import random
from jax.nn.initializers import normal
from netket.nn.masked_linear import default_kernel_init
from functools import partial
from models.utils import states_to_local_indices
import itertools as it

import netket as nk
from netket.operator.spin import sigmax, sigmaz  # an instance of the class nk.operator.LocalOperator

import networkx

from models.rnn1d import RNN1D

import numpy as np
import string

from scipy.linalg import ishermitian
import scipy

import os

import functools as ft

import time
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

# Testing whether we can use Callable here or not.
# from typing import Any, Callable, Optional, Tuple
from flax.linen.activation import sigmoid, tanh

import pickle


# config contains: Lx, Ly,
def main(config):
    # MODEL PARAMS
    print('NQS ENTANGLEMENT STUDY WITH THE TEAM: RNN1D')  # Let's see if this is something we have to change.
    print('_' * 30)
    periodic = config.get('periodic',False)  # if the dictionary doesn't contain 'periodic' as a key, return False as the value of
    L = config['L']
    # numSpins_A = config['numSpins_A']
    numSpins_A = L // 2
    lattice = config['lattice']
    nsamples = config['nsamples']  # batch size, andrew
    chunk_size = config['chunk_size']
    L_chunk_threshold = config['L_chunk_threshold']
    model_name = config['model']
    bias = config['bias']
    dhidden = config['dhidden']
    weight_sharing = config['weight_sharing']

    # THE FOLLOWING THREE ARGUMENTS ARE ALL STRINGS
    rnn_cell = config['rnn_cell']
    activation_fn = config['activation_fn']  # new argument to be able to specify activation functions in Vanilla RNN and GRU
    gate_fn = config['gate_fn']  # new argument to be able to specify gate function in GRU

    softmaxOnOrOff = config['softmaxOnOrOff']
    modulusOnOrOff = config['modulusOnOrOff']
    signOfProbsIntoPhase = config['signOfProbsIntoPhase']
    purity_correlation_sampling_errors = config['purity_correlation_sampling_errors']
    purityLogSumExp_correlation_sampling_errors = config['purityLogSumExp_correlation_sampling_errors']
    #fullyPropagatedError = config['fullyPropagatedError']

    autoreg = config['autoreg']  # just telling us if the model is autoregressive or not
    width = config['width']
    numrealizations = config['numrealizations']
    DATA = config['DATA']  # Run training
    PLOT = config['PLOT']  # Make plots
    # SEED = config['SEED']  # Random seed
    # GET_PATH = config.get('GET_PATH', False)  # Get path only if it is specified... I think this would be for checkpointing?
    SHOW = config['SHOW']  # Show plots
    DEVICE = jax.devices()[-1].device_kind # jax.devices(backend=None)[source]: Returns a list of all devices for a given backend.
    print("DEVICE (Roeland's way) =", DEVICE)
    print("jax.devices() =",jax.devices())
    exactOrMC = config['exactOrMC']  # 'exact' or 'MC' for sample generation for entropy calculation

    # New input to specify positive or complex RNN. pRNN vs cRNN
    positive_or_complex = config['positive_or_complex']
    
    print("jax.devices()[0].platform =",jax.devices()[0].platform)

    calcEntropy = config['calcEntropy']
    calcEntropyCleanUp = config['calcEntropyCleanUp']
    if calcEntropy or calcEntropyCleanUp:
        if L <= L_chunk_threshold:
            print()
            print("chunking for entropy calculation will NOT be used")
            print()
        else:
            print()
            print("chunking for entropy calculation WILL BE used. Chunk size =",chunk_size)
            print()
    calcRankQGT = config['calcRankQGT']
    calcEntangSpectrum = config['calcEntangSpectrum']
    calcAdjacentGapRatio = config['calcAdjacentGapRatio']

    # Entanglement scaling law analysis
    calcEntropy_scaling = config['calcEntropy_scaling']

    # Correlation-related arguments:
    calcCorr_1d_specific_dist = config['calcCorr_1d_specific_dist']
    calcCorr_1d_upTo_dist = config['calcCorr_1d_upTo_dist']
    d_specific_corr = config['d_specific_corr']  # if we want to calculate a correlation function at a specific distance of d
    d_up_to_corr = config['d_up_to_corr']  # if we want to calculate all correlations up to a distance of d...
    newSamplesOrNot_corr = config['newSamplesOrNot_corr'] # produce new samples for every exp value in corr function or not. 0 or 1.

    # tolerance for rank calculation
    tol_rank_calc = config['tol_rank_calc']

    print("positive_or_complex =", positive_or_complex)
    print("tol_rank_calc inside sample file =", tol_rank_calc)
    if tol_rank_calc == 0.0:
        print("Not specifying a tolerance for the QGT rank calculation")
    print("type(tol_rank_calc inside sample file) =", type(tol_rank_calc))

    # Quantities for Fourier Transform calculation: rho (pure state density matrix), rhoA (reduced dens. matrix of subsys. A)
    calcFourierTransform_rho = config['calcFourierTransform_rho']
    calcFourierTransform_rhoA = config['calcFourierTransform_rhoA']
    max_kBody_term = config['max_kBody_term']

    # New quantity: numpy or jax
    numpy_or_jax = config['numpy_or_jax']

    # New quantity: MCMC or ARD sampler
    ARD_vs_MCMC = config['sampler']

    # New quantities needed for MCMC:
    n_chains = config['n_chains']
    n_disc_chain = config['n_discard_per_chain']  # n_discard_per_chain
    sweep_size = config['sweep_size']

    # dtype for RNN parameters,
    dtype = config['dtype']
    # dtype_for_file_naming = dtype[10:]
    if dtype == 'jax.numpy.float16':
        dtype_for_file_naming = 'float16'
    elif dtype == 'jax.numpy.float32':
        dtype_for_file_naming = 'float32'
    elif dtype == 'jax.numpy.float64':
        dtype_for_file_naming = 'float64'

    print(f'1D chain of size {L}\n')
    hilbert = nk.hilbert.Spin(1 / 2, L)  # create the Hilbert space using the netket hilbert.Spin class

    print(f'Saving data: {DATA}')  # Ahh DATA tells us if we want to save the data or not, PLOT tells us if we want to plot or not.
    print(f'Plotting: {PLOT}')
    print('_' * 30)
    print(f'Number of samples = {nsamples}')
    print(f'RNN Cell = {rnn_cell}')
    print(f'activation_fn = {activation_fn}')
    print(f'gate_fn = {gate_fn}')
    print(f'softmaxOnOrOff = {softmaxOnOrOff}')
    print(f'modulusOnOrOff = {modulusOnOrOff}')
    print(f'purity_correlation_sampling_errors = {purity_correlation_sampling_errors}')
    print(f'purityLogSumExp_correlation_sampling_errors = {purityLogSumExp_correlation_sampling_errors}')
    print(f'signOfProbsIntoPhase = {signOfProbsIntoPhase}')
    print(f'Bias = {bias}')
    print(f'RNN dhidden = {dhidden}')
    print(f'numrealizations = {numrealizations}')

    if purity_correlation_sampling_errors:
        if ARD_vs_MCMC == 'ARD':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
        elif ARD_vs_MCMC == 'MCMC':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming \
                                  + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
    elif purityLogSumExp_correlation_sampling_errors:
        if ARD_vs_MCMC == 'ARD':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
        elif ARD_vs_MCMC == 'MCMC':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming \
                                  + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
    else: # purity_correlation_sampling_errors = 0
        if ARD_vs_MCMC == 'ARD':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
        elif ARD_vs_MCMC == 'MCMC':
            if exactOrMC == 'MC':
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                    f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                    f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                  + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            elif exactOrMC == 'exact': # only doing this for entang spectrum calculation
                path_name = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                        f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                        f'numrealiz_{numrealizations}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
    
        
    

    '''
    CHECKPOINTING
    '''
    # if the files all exist, kill the execution:
    if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
        if activation_fn == 'identity' or gate_fn == 'identity':
            if softmaxOnOrOff == 'on':
                ckpt_path = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
        else:  # neither activation_fn nor gate_fn is identity
            if softmaxOnOrOff == 'on':
                ckpt_path = f'./no_hcorr_bpur_data/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
    elif rnn_cell == 'RNN':
        if activation_fn == 'identity':
            if softmaxOnOrOff == 'on':
                ckpt_path = f'./no_hcorr_actFn_{activation_fn}/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
        else:  # neither activation_fn nor gate_fn is identity
            if softmaxOnOrOff == 'on':
                ckpt_path = f'./no_hcorr_bpur_data/{path_name}'
            elif softmaxOnOrOff == 'off':
                if modulusOnOrOff == 'on':
                    ckpt_path = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name}'
                elif modulusOnOrOff == 'off':
                    ckpt_path = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name}'
                

    filepath_purity_array = ckpt_path + '/purity_array.npy'
    filepath_purity_array_checkpointing = ckpt_path + '/' + 'purity_array' + '_checkpointing' + '.npy'
    # Will only use error file paths if purity_correlation_sampling_errors == 1.
    if purity_correlation_sampling_errors:
        filepath_purity_RNN_sampling_error_array = ckpt_path + '/purity_RNN_sampling_error_array.npy'
        filepath_purity_RNN_sampling_error_array_checkpointing = ckpt_path + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
    elif purityLogSumExp_correlation_sampling_errors: # logSumExp will only compute the entropy using what is hopefully a more stable technique...
        #filepath_purity_RNN_sampling_error_array = ckpt_path + '/purity_RNN_sampling_error_array.npy'
        #filepath_purity_RNN_sampling_error_array_checkpointing = ckpt_path + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
        filepath_entropy_logSumExp_array = ckpt_path + '/entropy_logSumExp_array.npy'
        filepath_entropy_logSumExp_array_checkpointing = ckpt_path + '/entropy_logSumExp_array' + '_checkpointing' + '.npy'
        
    
    filepath_entropy_array = ckpt_path + '/entropy_array.npy'
    filepath_entropy_array_checkpointing = ckpt_path + '/entropy_array' + '_checkpointing' + '.npy'

    filepath_config_txt = ckpt_path + '/config.txt'
    # filepath_rankFullness_qgt_array = ckpt_path + '/rankFullness_qgt_array.npy'

    filepath_rankFullness_qgt_array = ckpt_path + '/' + 'tolRank' + '{:.0e}'.format(tol_rank_calc) + '_rankFullness_qgt_array.npy'
    # '{tol_rank_calc}_'
    filepath_eigenvalues_rho_A_array = ckpt_path + '/eigenvalues_rho_A_array.npy'
    filepath_eigenvalues_rho_A_array_checkpointing = ckpt_path + '/eigenvalues_rho_A_array_checkpointing.npy'
    # filepath_entanglement_spectrum_array = ckpt_path + '/entanglement_spectrum_array.npy'
    filepath_adjacent_energy_gap_ratio_array_checkpointing = ckpt_path + '/adjacent_energy_gap_ratio_array_checkpointing.npy'
    filepath_adjacent_energy_gap_ratio_array = ckpt_path + '/adjacent_energy_gap_ratio_array.npy'
    # rho
    filepath_fourier_transform_rho_checkpointing = ckpt_path + '/' + 'fourier_transform_rho_checkpointing_k' + str(max_kBody_term) + '.npy'
    filepath_fourier_transform_rho = ckpt_path + '/' + 'fourier_transform_rho_k' + str(max_kBody_term) + '.npy'
    # rhoA
    filepath_fourier_transform_rhoA_checkpointing = ckpt_path + '/' + 'fourier_transform_rhoA_checkpointing_k' + str(max_kBody_term) + '.npy'
    filepath_fourier_transform_rhoA = ckpt_path + '/' + 'fourier_transform_rhoA_k' + str(max_kBody_term) + '.npy'
    # correlations
    filepath_corr_d_specific_checkpointing = ckpt_path + '/' + 'corr_avg_specific_d' + str(d_specific_corr) + '_checkpointing' + '.npy'
    filepath_corr_d_specific = ckpt_path + '/' + 'corr_avg_specific_d' + str(d_specific_corr) + '_' + '.npy'

    filepath_corr_up_to_d_checkpointing = ckpt_path + '/' + 'corr_abs_avg_up_to_d' + str(d_up_to_corr) + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.npy'
    filepath_corr_up_to_d = ckpt_path + '/' + 'corr_abs_avg_up_to_d' + str(d_up_to_corr) + '_newSamples' + str(newSamplesOrNot_corr) + '.npy'

    #d_up_to_corr
    if purity_correlation_sampling_errors:
        
        filepath_sigmai_sigmaj_RNN_sampling_avg_dict = ckpt_path + f'/sigmai_sigmaj_d{d_up_to_corr}_RNN_sampling_avg_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '.pickle'
        filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing = ckpt_path + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_sampling_avg_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
        filepath_sigmai_sigmaj_RNN_error_of_mean_dict = ckpt_path + f'/sigmai_sigmaj_d{d_up_to_corr}_RNN_error_of_mean_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '.pickle'
        filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing = ckpt_path + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_error_of_mean_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
        filepath_sigmai_RNN_sampling_avg_array = ckpt_path + '/sigmai_RNN_sampling_avg_array' + '_newSamples' + str(newSamplesOrNot_corr) + '.npy'
        filepath_sigmai_RNN_sampling_avg_array_checkpointing = ckpt_path + '/sigmai_RNN_sampling_avg_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'
        filepath_sigmai_RNN_error_of_mean_array = ckpt_path + '/sigmai_RNN_error_of_mean_array' + '_newSamples' + str(newSamplesOrNot_corr) + '.npy'
        filepath_sigmai_RNN_error_of_mean_array_checkpointing = ckpt_path + '/sigmai_RNN_error_of_mean_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'
        
    # Entanglement scaling law (using 2-Renyi entropy)
    filepath_entropy_scaling_law_checkpointing = ckpt_path + '/' + 'entropy_scaling' + '_checkpointing' + '.npy'
    filepath_entropy_scaling_law = ckpt_path + '/' + 'entropy_scaling' + '.npy'


    shouldWeLoopOverRealizations = 0
    if purity_correlation_sampling_errors:
        if calcEntropy:
            if (not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array)) \
            or (not os.path.isfile(filepath_purity_RNN_sampling_error_array)):
                shouldWeLoopOverRealizations += 1
    elif purityLogSumExp_correlation_sampling_errors: # if this = 1, let's output all the files, including the logSumExp entropy file.
            shouldWeLoopOverRealizations += 1
        
    else: # no errors needed
        if calcEntropy:
            if (not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array)):
                shouldWeLoopOverRealizations += 1
    

    if calcRankQGT and (not os.path.isfile(filepath_rankFullness_qgt_array)):
        shouldWeLoopOverRealizations += 1

    if calcEntangSpectrum and (not os.path.isfile(filepath_eigenvalues_rho_A_array)):
        shouldWeLoopOverRealizations += 1

    if calcAdjacentGapRatio and (not os.path.isfile(filepath_adjacent_energy_gap_ratio_array)):
        shouldWeLoopOverRealizations += 1


    if calcFourierTransform_rho and (not os.path.isfile(filepath_fourier_transform_rho)):
        shouldWeLoopOverRealizations += 1

    if calcFourierTransform_rhoA and (not os.path.isfile(filepath_fourier_transform_rhoA)):
        shouldWeLoopOverRealizations += 1

    if calcCorr_1d_specific_dist and (not os.path.isfile(filepath_corr_d_specific)):
        shouldWeLoopOverRealizations += 1

    if purity_correlation_sampling_errors:
        if calcCorr_1d_upTo_dist:
            if (not os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict)) \
            or (not os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict)) \
            or (not os.path.isfile(filepath_sigmai_RNN_sampling_avg_array)) \
            or (not os.path.isfile(filepath_sigmai_RNN_error_of_mean_array)):
                shouldWeLoopOverRealizations += 1
    else:
        if calcCorr_1d_upTo_dist and (not os.path.isfile(filepath_corr_up_to_d)):
            shouldWeLoopOverRealizations += 1

    if calcEntropy_scaling and (not os.path.isfile(filepath_entropy_scaling_law)):
        shouldWeLoopOverRealizations += 1

    '''
    if calcEntropyCleanUp = 1, let's increment shouldWeLoopOverRealizations to override this checkpoint. We want to clean-up
    the completed purity and entropy files. We won't be looping over realizations but we need to override this obstacle. In the
    future we'll clean up this code and make it less disorganized.
    '''
    if calcEntropyCleanUp:
        shouldWeLoopOverRealizations += 1

    if calcEntropy and shouldWeLoopOverRealizations == 0: # this means the final files exist for the desired simulation

        if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_array_checkpointing) \
        and os.path.isfile(filepath_entropy_array_checkpointing) \
        and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing):
            #print('asdasdasdaadsdasd')
            purity_array = np.load(filepath_purity_array_checkpointing)
            entropy_array = np.load(filepath_entropy_array_checkpointing)
            purity_RNN_sampling_error_array = np.load(filepath_purity_RNN_sampling_error_array_checkpointing)

        elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array_checkpointing):
            #print("asdasd")
            #purity_RNN_sampling_error_array = np.load(filepath_purity_RNN_sampling_error_array_checkpointing)
            
            entropy_logSumExp_array = np.load(filepath_entropy_logSumExp_array_checkpointing)
            print()
            print("entropy_logSumExp_array.shape",entropy_logSumExp_array.shape)
            print("type(entropy_logSumExp_array)",type(entropy_logSumExp_array))
        else:
            purity_array = np.load(filepath_purity_array_checkpointing)
            entropy_array = np.load(filepath_entropy_array_checkpointing)
            
            #print(entropy_logSumExp_array)
        
        if purity_correlation_sampling_errors:
            if (not os.path.isfile(filepath_purity_RNN_sampling_error_array)) \
            and (not os.path.isfile(filepath_purity_array)) and (not os.path.isfile(filepath_entropy_array)):
                np.save(filepath_purity_array, purity_array)
                np.save(filepath_entropy_array, entropy_array)
                np.save(filepath_purity_RNN_sampling_error_array, purity_RNN_sampling_error_array)
        elif purityLogSumExp_correlation_sampling_errors:
            if (not os.path.isfile(filepath_entropy_logSumExp_array)):
            #np.save(filepath_purity_RNN_sampling_error_array, purity_RNN_sampling_error_array)
                np.save(filepath_entropy_logSumExp_array, entropy_logSumExp_array)
        else:
            if (not os.path.isfile(filepath_entropy_array)) and (not os.path.isfile(filepath_purity_array)):
                np.save(filepath_purity_array, purity_array)
                np.save(filepath_entropy_array, entropy_array)

            
        print()
        print("Final Statistics")
        print()
        if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing) \
        and os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
            print("average purity =", np.mean(purity_array))
            print("average entropy =", np.mean(entropy_array))
            print()
        elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array_checkpointing):
            # We could also do the mean first, absolute value second
            print("average entropy LogSumExp abs 1st mean 2nd =", np.mean(np.abs(entropy_logSumExp_array)))
            print("average entropy LogSumExp mean 1st abs 2nd =", np.abs(np.mean(entropy_logSumExp_array)))
            print("average entropy LogSumExp mean 1st real 2nd",(np.mean(entropy_logSumExp_array)).real)
            print("average entropy LogSumExp real 1st mean 2nd",np.mean(entropy_logSumExp_array.real))
            print()
        elif os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
            print("average purity =", np.mean(purity_array))
            print("average entropy =", np.mean(entropy_array))
            print()
        
        
        #if purity_correlation_sampling_errors and (not os.path.isfile(filepath_purity_RNN_sampling_error_array)):
        if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array) \
        and os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
            # Calculating standard error of the mean for entropy: see my notebook and error prop. page on Wikipedia for more
            # info
            if (dhidden == 0) or (width == 0.0):
            #if width == 0.0:
                entropy_standard_error = 0
                purity_standard_error = 0
            else:
                entropy_standard_error = 2 * np.log(purity_RNN_sampling_error_array) - np.log(nsamples) - 2 * np.log(purity_array)
                entropy_standard_error = np.sum(np.exp(entropy_standard_error))
                entropy_standard_error = np.sqrt(entropy_standard_error) / numrealizations
                
                purity_standard_error = 2 * np.log(purity_RNN_sampling_error_array) - np.log(nsamples)
                purity_standard_error = np.sum(np.exp(purity_standard_error))
                purity_standard_error = np.sqrt(purity_standard_error) / numrealizations
            

            print("fully propagated standard error on purity across all realiz =",purity_standard_error)
            print("fully propagated standard error on entropy across all realiz =",entropy_standard_error)
            purity_standard_error_unpropagated = np.std(purity_array) / np.sqrt(numrealizations)
            entropy_standard_error_unpropagated = np.std(entropy_array) / np.sqrt(numrealizations)
            print()
            print("unpropagated standard error on purity across all realiz =",purity_standard_error_unpropagated)
            print("unpropagated standard error on entropy across all realiz =",entropy_standard_error_unpropagated)
                
        #elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array) and os.path.isfile(filepath_entropy_logSumExp_array):
        elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array):

            print("logSumExp standard error on entropy across all realiz, abs 1st, std 2nd =",np.std(np.abs(entropy_logSumExp_array)) / np.sqrt(numrealizations))
            print("logSumExp standard error on entropy across all realiz, std only =",np.std(entropy_logSumExp_array) / np.sqrt(numrealizations))
            print("logSumExp standard error on entropy across all realiz, real part only =",np.std(entropy_logSumExp_array.real) / np.sqrt(numrealizations))
            print()
    
    
    
    # if the final files are present, I want to see the final stats
    if calcCorr_1d_upTo_dist and shouldWeLoopOverRealizations == 0: # and (not os.path.isfile(filepath_corr_up_to_d)):
        
        if purity_correlation_sampling_errors \
        and os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing) \
        and os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing) \
        and os.path.isfile(filepath_sigmai_RNN_sampling_avg_array_checkpointing) \
        and os.path.isfile(filepath_sigmai_RNN_error_of_mean_array_checkpointing):
            #print('asdasdasdaadsdasd')
            with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, 'rb') as handle:
                corr_unconnected_avg_up_to_d_dict_of_arrays = pickle.load(handle)
            
            with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing, 'rb') as handle:
                corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = pickle.load(handle)
            magnetization_avg_array = np.load(filepath_sigmai_RNN_sampling_avg_array_checkpointing)
            magnetization_error_of_mean_array = np.load(filepath_sigmai_RNN_error_of_mean_array_checkpointing)
            
            assert(type(corr_unconnected_avg_up_to_d_dict_of_arrays) == dict)
            assert(type(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays) == dict)
            
            # SAVE IN FINAL FILES if needed
            with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict, 'wb') as handle:
                pickle.dump(corr_unconnected_avg_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict, 'wb') as handle:
                pickle.dump(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(filepath_sigmai_RNN_sampling_avg_array,magnetization_avg_array)
            np.save(filepath_sigmai_RNN_error_of_mean_array,magnetization_error_of_mean_array)
            
            # Now let's calculate
            
            assert(magnetization_avg_array.shape[0] == numrealizations)
            assert(magnetization_error_of_mean_array.shape[0] == numrealizations)
            assert(len(corr_unconnected_avg_up_to_d_dict_of_arrays['d=1']) == numrealizations)
            assert(len(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=1']) == numrealizations)
            # Now we calculate the average correlation function across all sites for d = 1 through d = d_up_to_corr,
            # and then we calculate the error as well, in order. First, the average correlation function across all
            # sites:
            print()
            print("Final Statistics")
            print()
            
            
            
            #count = 0
            conn_corr_avg_over_all_sites_all_realiz = np.zeros((numrealizations, d_up_to_corr))
            log_conn_corr_avg_over_all_sites_all_realiz = np.zeros((numrealizations, d_up_to_corr))
            for r in range(numrealizations): # loop over wavefunctions
                for d in range(1,d_up_to_corr+1): # loop over values of d
                    # We need the list_of_edges for the loop below: this depends on k
                    list_of_edges_d = []
                    for j in range(L - d):
                        list_of_edges_d.append((j, j + d))
                    #print("d =",d,"; len(list_of_edges) =",len(list_of_edges))
                    assert(len(list_of_edges_d) == L-d)
                
                    #connected_corr_avg_over_all_m_n_pairs = 0
                    conn_r_d_avg_over_sites = 0
                    log_conn_r_d_avg_over_sites = 0
                    for k in range(L-d): # loop over all bonds (site pairs) a distance d apart (there are L-d of them)
                        edge_k = list_of_edges_d[k]
                        m = edge_k[0]
                        n = edge_k[1]
                        # First extract list or (array) corresponding to the unconn corr function at separation d for
                        # wavefunction/realiz r, from which we take the k^th term to get the k^th unconn corr function.
                        unconn_r_d_k = corr_unconnected_avg_up_to_d_dict_of_arrays['d=%s'%d][r][k] # This should be a list
                        # Next we subtract the right product of magnetization
                        conn_r_d_k = abs(unconn_r_d_k - magnetization_avg_array[r,m] * magnetization_avg_array[r,n])
                        conn_r_d_avg_over_sites += conn_r_d_k
                        log_conn_r_d_avg_over_sites += np.log(conn_r_d_k)

                    conn_corr_avg_over_all_sites_all_realiz[r,d-1] += conn_r_d_avg_over_sites / (L-d) 
                    log_conn_corr_avg_over_all_sites_all_realiz[r,d-1] += log_conn_r_d_avg_over_sites / (L-d)
                    
                        
            
            assert(conn_corr_avg_over_all_sites_all_realiz.shape == (numrealizations, d_up_to_corr))
            assert(log_conn_corr_avg_over_all_sites_all_realiz.shape == (numrealizations, d_up_to_corr))
                        
            print("avg corr_abs_avg_up_to_d across realizations =", np.mean(conn_corr_avg_over_all_sites_all_realiz,axis=0))
            print("unpropagated errors for corr across all realiz =",np.std(conn_corr_avg_over_all_sites_all_realiz,axis=0) / np.sqrt(numrealizations))
            
            print("avg log_corr_abs_avg_up_to_d across realizations =", np.mean(log_conn_corr_avg_over_all_sites_all_realiz,axis=0))
            print("unpropagated errors for log corr across all realiz =",np.std(log_conn_corr_avg_over_all_sites_all_realiz,axis=0) / np.sqrt(numrealizations))
            # Last bit is we need code for the errors
            
            

        else:
            print("adsdasdasdasdasd")
            if os.path.isfile(filepath_corr_up_to_d_checkpointing):
                corr_abs_avg_up_to_d_array = np.load(filepath_corr_up_to_d_checkpointing)
                np.save(filepath_corr_up_to_d, corr_abs_avg_up_to_d_array)
                print()
                print("Final Statistics")
                print()
                print("avg corr_abs_avg_up_to_d across realizations =", np.mean(corr_abs_avg_up_to_d_array,axis=0))
                print("log of avg corr_abs_avg_up_to_d across realizations =", np.log(np.mean(corr_abs_avg_up_to_d_array,axis=0)))
                print("unpropagated errors for corr across all realiz =",np.std(corr_abs_avg_up_to_d_array,axis=0) / np.sqrt(numrealizations))
    
    
    
    
    
    
    
    # KILL EXECUTION HERE IF ALL FINAL FILES ARE PRESENT.
    if shouldWeLoopOverRealizations == 0:
        print("The desired final files are all present. Enjoy.")
        return ckpt_path

    # if directory doesn't exist, create it
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # if the config file doesn't exist yet, 
    if not os.path.isfile(filepath_config_txt):
        with open(filepath_config_txt, 'w') as file:
            for k, v in config.items():
                file.write(k + f'={v}\n')  # saving the dictionary items of "config" in a file... \n is new line object.

    # It's the same "samples" for every realization of the wavefunction if we are doing this exactly, so we only need
    # to generate the set of samples one time.
    # ...
    # There is a smarter way to do the partial trace in the exact case, and not use the swap operator trick. We want to
    # calculate the reduced density matrix exactly.
    # ...
    # Before starting the loop over numrealizations, if we want to calculate the entropy exactly, then produce the samples
    # We only do this calcEntropy == True and if the relevant files don't already exist. Checking for the existence of the
    # files is what accomplishes the checkpointing.
    # ...
    # calcEntropy = True means "calculate the entropy". So if calcEntropy == True AND the relevant files storing
    # the entropy / purity arrays (containing entropies and purities associated with each wavefunction realization
    # at a given dhidden and width) don't exist, then go ahead and produce those samples.
    if calcEntropy and (not os.path.isfile(filepath_purity_array)) and (not os.path.isfile(filepath_entropy_array)):
        # Produce the exact samples if exactOrMC == 'exact'.
        if exactOrMC == 'exact':
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    if (calcEntangSpectrum and (not os.path.isfile(filepath_eigenvalues_rho_A_array))):
        if not ('configs_full' in locals()):
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    if calcAdjacentGapRatio and (not os.path.isfile(filepath_adjacent_energy_gap_ratio_array)):
        if not ('configs_full' in locals()):
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    if calcFourierTransform_rho and (not os.path.isfile(filepath_fourier_transform_rho)):
        if not ('configs_full' in locals()):
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    if calcFourierTransform_rhoA and (not os.path.isfile(filepath_fourier_transform_rhoA)):
        if not ('configs_full' in locals()):
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    # Entanglement Scaling analysis requires the same if statements as above if we want the reduced density matrix exactly
    if calcEntropy_scaling and (not os.path.isfile(filepath_entropy_scaling_law)):
        # Produce the exact samples if exactOrMC == 'exact'.
        if exactOrMC == 'exact':
            configs = []
            for k in range(L):
                configs.append((-1, 1))
            configs = list(it.product(*configs))  # list of configurations
            configs_full = np.array(configs)  # store the full set of configurations as a numpy array, then proceed

    # Initialize arrays in case we want to calculate the relevant quantities
    purity_array = []
    if purity_correlation_sampling_errors:
        purity_RNN_sampling_error_array = [] # new error array that we will use for the error propagation
    elif purityLogSumExp_correlation_sampling_errors:
        #purity_RNN_sampling_error_array = [] # new error array that we will use for the error propagation
        entropy_logSumExp_array = []
    entropy_array = []
    rankFullness_qgt_array = []
    eigenvalues_rho_A = []  # initialize array
    adjacent_energy_gap_ratio_array = []
    sum_abs_Op_rho_array = []  # this is going to be a list of lists or a list of tuples, with each individual
    # element being a list containing the k values associated with each realization
    sum_abs_Op_rhoA_array = []  # this is going to be a list of lists or a list of tuples, with each individual
    # element being a list containing the k values associated with each realization

    if purity_correlation_sampling_errors or purityLogSumExp_correlation_sampling_errors:
        print()

    # Entropy scaling functions
    entropy_scaling_array = []
    
    # Correlation functions
    corr_avg_specific_d_array = []
    corr_abs_avg_up_to_d_array = []
    
    # Initialize the dictionaries and arrays needed for the correlation function
    corr_unconnected_avg_up_to_d_dict_of_arrays = {} # keys 'd = 1', 'd = 2', etc. up to d =L/2
    corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = {} # corresponding errors of mean
    magnetization_avg_array = []
    magnetization_error_of_mean_array = []
    


    '''
    Defining special quantity to do checkpointing for the entropy calculation. This means we should only calculate the
    entropy in simulations where that's all we are calculating.
    '''
    indexWhereToStart = 0  # start the loop from scratch unless some of the below if statements are satisfied
    if purity_correlation_sampling_errors:
        # Here, we want the sampling errors for the purity calculation. So we look for the relevant error files as well.
        if calcEntropy and ((not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array)) \
        or (not os.path.isfile(filepath_purity_RNN_sampling_error_array))):
            # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
            # function. So now let's check where to start
            # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
            if os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing) \
            and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                purity_array = list(np.load(filepath_purity_array_checkpointing))
                purity_RNN_sampling_error_array = list(np.load(filepath_purity_RNN_sampling_error_array_checkpointing))
                entropy_array = list(np.load(filepath_entropy_array_checkpointing))
                   
                    
                if len(purity_array) == len(entropy_array) and len(purity_array) == len(purity_RNN_sampling_error_array):
                    indexWhereToStart = len(entropy_array)
                    # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(conv_time_array)
                    print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
                    print("Entropy/purity calculation initiated.")
                else:
                    print("checkpointing for entropy, purity and/or purity error has resulted in ckpt files of different lengths. Restarting simulation.")
                    purity_array = []  # will store correlation function for all the realizations
                    purity_RNN_sampling_error_array = [] # will store sampling error for purity of each wavefunction
                    entropy_array = []
                    indexWhereToStart = 0
                    #return 0
                
            else:  # No simulations have been completed yet
                purity_array = []  # will store correlation function for all the realizations
                purity_RNN_sampling_error_array = [] # will store sampling error for purity of each wavefunction
                entropy_array = []
                indexWhereToStart = 0  # no need to rewrite this but that's ok
                print("No simulations have been performed that have resulted in all checkpointing files being present. Start from scratch, realization = 0.")
                print("Entropy/purity calculation initiated.")
                
    elif purityLogSumExp_correlation_sampling_errors:
        # Here, we want the sampling errors for the purity calculation. So we look for the relevant error files as well.
        if calcEntropy and (not os.path.isfile(filepath_entropy_logSumExp_array)):
            # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
            # function. So now let's check where to start
            # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
            if os.path.isfile(filepath_entropy_logSumExp_array_checkpointing):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                entropy_logSumExp_array = list(np.load(filepath_entropy_logSumExp_array_checkpointing))
                   
                    
                indexWhereToStart = len(entropy_logSumExp_array)
                    # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(conv_time_array)
                
                print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
                print("Entropy logSumExp calculation initiated.")

            else:  # No simulations have been completed yet
                entropy_logSumExp_array = []
                indexWhereToStart = 0  # no need to rewrite this but that's ok
                print("No simulations have been performed that have resulted in all checkpointing file being present. Start from scratch, realization = 0.")
                print("Entropy logSumExp calculation initiated.")

    else: # not calculating errors
        # Here, we DO NOT want the sampling errors for the purity calculation.
        if calcEntropy and ((not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array))):
            # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
            # function. So now let's check where to start
            # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
            if os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                purity_array = list(np.load(filepath_purity_array_checkpointing))
                entropy_array = list(np.load(filepath_entropy_array_checkpointing))
                   
                if len(purity_array) == len(entropy_array):
                    indexWhereToStart = len(entropy_array)
                    # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(conv_time_array)
                    print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
                    print("Entropy/purity calculation initiated.")
                else:
                    print("checkpointing for entropy and purity and purity error has resulted in files of different lengths. Restarting simulation")
                    purity_array = []  # will store correlation function for all the realizations
                    entropy_array = []
                    indexWhereToStart = 0
                    #return 0
                
            else:  # No simulations have been completed yet
                purity_array = []  # will store correlation function for all the realizations
                entropy_array = []
                indexWhereToStart = 0  # no need to rewrite this but that's ok
                print("No simulations have been performed that have resulted in checkpointing files being present. Start from scratch, realization = 0.")
                print("Entropy/purity calculation initiated.")
    


    '''
    Defining special quantity to do checkpointing for the correlation function calculation. This means we should only calculate the
    correlation function in simulations where that's all we are calculating.
    '''
    if purity_correlation_sampling_errors:
        if calcCorr_1d_upTo_dist and ((not os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict)) \
        or (not os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict)) \
        or (not os.path.isfile(filepath_sigmai_RNN_sampling_avg_array)) \
        or (not os.path.isfile(filepath_sigmai_RNN_error_of_mean_array))):
            # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
            # function. So now let's check where to start
            # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
            if os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing) \
            and os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing) \
            and os.path.isfile(filepath_sigmai_RNN_sampling_avg_array_checkpointing) \
            and os.path.isfile(filepath_sigmai_RNN_error_of_mean_array_checkpointing):  # if statement to check if ckpt file exists (it should if simulations have
                # been completed already
                with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, 'rb') as handle:
                    corr_unconnected_avg_up_to_d_dict_of_arrays = pickle.load(handle)
                
                with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing, 'rb') as handle:
                    corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = pickle.load(handle)
                
                assert(type(corr_unconnected_avg_up_to_d_dict_of_arrays) == dict)
                assert(type(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays) == dict)

                magnetization_avg_array = list(np.load(filepath_sigmai_RNN_sampling_avg_array_checkpointing))
                magnetization_error_of_mean_array = list(np.load(filepath_sigmai_RNN_error_of_mean_array_checkpointing))
                
                keys = list(corr_unconnected_avg_up_to_d_dict_of_arrays.keys())
                
                if len(magnetization_avg_array) == len(magnetization_error_of_mean_array) \
                and len(magnetization_avg_array) == len(corr_unconnected_avg_up_to_d_dict_of_arrays[keys[0]]) \
                and len(magnetization_avg_array) == len(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays[keys[0]]):
                    indexWhereToStart = len(magnetization_avg_array)
                    # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(conv_time_array)
                    print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
                    print("Correlation Function calculation UP TO d (with errors) initiated.")
                else:
                    print("checkpointing for correlation function has resulted in ckpt files of different lengths. Restarting simulation.")
                    corr_unconnected_avg_up_to_d_dict_of_arrays = {} # keys 'd = 1', 'd = 2', etc. up to d =L/2
                    corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = {} # corresponding errors of mean
                    magnetization_avg_array = []
                    magnetization_error_of_mean_array = []
                    indexWhereToStart = 0
                    #return 0
                
            
            else:  # No simulations have been completed yet
                corr_unconnected_avg_up_to_d_dict_of_arrays = {} # keys 'd = 1', 'd = 2', etc. up to d =L/2
                corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = {} # corresponding errors of mean
                magnetization_avg_array = []
                magnetization_error_of_mean_array = []
                
                indexWhereToStart = 0
                print("No simulations have been performed that have resulted in all checkpointing files being present. Start from scratch, realization = 0.")
                print("Correlation Function UP TO d calculation initiated.")
                #print("No simulations have been performed. Start from scratch, realization = 0.")
    else:
        if calcCorr_1d_upTo_dist and (not os.path.isfile(filepath_corr_up_to_d)):
            # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
            # function. So now let's check where to start
            # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
            if os.path.isfile(filepath_corr_up_to_d_checkpointing):  # if statement to check if ckpt file exists (it should if simulations have
                # been completed already
                corr_abs_avg_up_to_d_array = list(np.load(filepath_corr_up_to_d_checkpointing))
                indexWhereToStart = len(corr_abs_avg_up_to_d_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                # so we want to start at i = 3, i.e. i = len(conv_time_array)
                print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
                print("Correlation Function UP TO d calculation initiated.")
            else:  # No simulations have been completed yet
                corr_abs_avg_up_to_d_array = []  # will store correlation function for all the realizations
                indexWhereToStart = 0
                print("No simulations have been performed that have resulted in checkpointing file being present. Start from scratch, realization = 0.")
                print("Correlation Function UP TO d calculation initiated.")
    

    '''
    Defining special quantity to do checkpointing for the entropy scaling calculation. This means we should only calculate the
    entropy scaling in simulations where that's all we are calculating.
    '''
    if calcEntropy_scaling and (not os.path.isfile(filepath_entropy_scaling_law)):
        # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
        # function. So now let's check where to start
        # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
        if os.path.isfile(filepath_entropy_scaling_law_checkpointing):  # if statement to check if ckpt file exists (it should if simulations have
            # been completed already
            entropy_scaling_array = list(np.load(filepath_entropy_scaling_law_checkpointing))
            indexWhereToStart = len(entropy_scaling_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
            np.load()
            # so we want to start at i = 3, i.e. i = len(conv_time_array)
            print("Entropy scaling calculation initiated.")
            print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
        else:  # No simulations have been completed yet
            entropy_scaling_array = []  # will store convergence time for all the realizations
            indexWhereToStart = 0
            print("Entropy scaling calculation initiated.")
            print("No simulations have been performed. Start from scratch, realization = 0.")

    '''
    Defining special quantity to do checkpointing for the entang spectrum calculation. This means we should only calculate the
    gap ratio in simulations where that's all we are calculating.
    '''
    if calcEntangSpectrum and (not os.path.isfile(filepath_eigenvalues_rho_A_array)):
        # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
        # function. So now let's check where to start
        # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
        if os.path.isfile(filepath_eigenvalues_rho_A_array_checkpointing):  # if statement to check if ckpt file exists (it should if simulations have
            # been completed already
            eigenvalues_rho_A = list(np.load(filepath_eigenvalues_rho_A_array_checkpointing))
            indexWhereToStart = len(eigenvalues_rho_A) // 2**(L//2)  # if list length is 3 * 2**(L//2), then we want to start at realization 3.
            print("Entang spec (eigenvalues of rho_A) calculation initiated.")
            print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",
                  indexWhereToStart)
            print("len(eigenvalues_rho_A) =", len(eigenvalues_rho_A))
        else:  # No simulations have been completed yet
            eigenvalues_rho_A = []  # will store convergence time for all the realizations
            indexWhereToStart = 0
            print("Entang spec (eigenvalues of rho_A) calculation initiated.")
            print("No simulations have been performed. Start from scratch, realization = 0.")



    
    '''
    Defining special quantity to do checkpointing for the adjacent gap ratio calculation. This means we should only calculate the
    gap ratio in simulations where that's all we are calculating.
    '''
    if calcAdjacentGapRatio and (not os.path.isfile(filepath_adjacent_energy_gap_ratio_array)):
        # If we are here, there are at least some realizations / simulations we haven't completed yet for the calculation of the correlation
        # function. So now let's check where to start
        # We check for where to start the loop (i.e. what simulation we are at.) by studying the checkpointing file
        if os.path.isfile(filepath_adjacent_energy_gap_ratio_array_checkpointing):  # if statement to check if ckpt file exists (it should if simulations have
            # been completed already
            adjacent_energy_gap_ratio_array = list(np.load(filepath_adjacent_energy_gap_ratio_array_checkpointing))
            indexWhereToStart = len(adjacent_energy_gap_ratio_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
            # so we want to start at i = 3, i.e. i = len(conv_time_array)
            print("Adjacent gap ratio calculation initiated.")
            print("Data already exists for some realizations (but not all). Loading the data. Continuing at realiz = ",indexWhereToStart)
        else:  # No simulations have been completed yet
            adjacent_energy_gap_ratio_array = []  # will store convergence time for all the realizations
            indexWhereToStart = 0
            print("Adjacent gap ratio calculation initiated.")
            print("No simulations have been performed. Start from scratch, realization = 0.")

        
    
    
    '''
    HIGHER # of numrealizations: First we're gonna check if the same simulation has already been performed at a HIGHER number of realizations.
    '''
    # Check if simulations have already been performed and possibly completed at higher number of realizations.
    # No simulations have been conducted in the present folder, so let's check for higher number of realizations with all other
    # hyperparameters held equal.
    found_higher = 0
    if (indexWhereToStart == 0) and calcEntropy and (not calcEntropyCleanUp):
        print()
        print("Actually wait. First checking to see if data for same hyperparameters but HIGHER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at HIGHER numbers of realizations, we will check for LOWER number of realizations.")
        
        #found_higher = 0
        for numrealiz_i in range(numrealizations + 500, numrealizations, -1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations + 500 to numrealizations + 1.
            
            # Re-initialize arrays (in the event arrays of different lengths were found at a previous iteration of the loop, and so we want
            # to discard those realizations.)
            
            if purity_correlation_sampling_errors:
                purity_RNN_sampling_error_array = []
                purity_array = []
                entropy_array = []
            elif purityLogSumExp_correlation_sampling_errors:
                #purity_RNN_sampling_error_array = []
                entropy_logSumExp_array = []
            else:
                purity_array = []
                entropy_array = []
            
            
            
            if purity_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            elif purityLogSumExp_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            else: # purity_correlation_sampling_errors = 0
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                    
                    
                    
            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                            
            
            
            if purity_correlation_sampling_errors:
                # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly
                # (the checkpointing files should always exist)
                filepath_purity_array_checkpointing_i = ckpt_path_i + '/' + 'purity_array' + '_checkpointing' + '.npy'
                filepath_entropy_array_checkpointing_i = ckpt_path_i + '/entropy_array' + '_checkpointing' + '.npy'
                filepath_purity_RNN_sampling_error_array_checkpointing_i = ckpt_path_i + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
            elif purityLogSumExp_correlation_sampling_errors:
                #filepath_purity_RNN_sampling_error_array_checkpointing_i = ckpt_path_i + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
                filepath_entropy_logSumExp_array_checkpointing_i = ckpt_path_i + '/entropy_logSumExp_array' + '_checkpointing' + '.npy'
            else:
                filepath_purity_array_checkpointing_i = ckpt_path_i + '/' + 'purity_array' + '_checkpointing' + '.npy'
                filepath_entropy_array_checkpointing_i = ckpt_path_i + '/entropy_array' + '_checkpointing' + '.npy'
            
            if purity_correlation_sampling_errors:
                if os.path.isfile(filepath_purity_array_checkpointing_i) and os.path.isfile(filepath_entropy_array_checkpointing_i) \
                and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    purity_array = list(np.load(filepath_purity_array_checkpointing_i))
                    purity_RNN_sampling_error_array = list(np.load(filepath_purity_RNN_sampling_error_array_checkpointing_i))
                    entropy_array = list(np.load(filepath_entropy_array_checkpointing_i))
                    
                    
                    # We are assuming the simulations at higher numrealizations values have been performed and reached a higher number
                    # of realizations than our present simulation, even if they haven't been completed yet.

                    if len(purity_array) == len(entropy_array) and len(purity_array) == len(purity_RNN_sampling_error_array): # good sign, sanity check, check if length of arrays is the same
                        #indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        if len(purity_array) >= numrealizations:
                            length_original = len(purity_array)
                            purity_array = purity_array[:numrealizations]
                            purity_RNN_sampling_error_array = purity_RNN_sampling_error_array[:numrealizations]
                            entropy_array = entropy_array[:numrealizations]
                            print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                                  ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                            if length_original < numrealiz_i:
                                print("There are actually less data points recorded than numrealiz_i =",numrealiz_i, "points, which is OK. It's still more data points than numrealizations =",numrealizations)
                                
                            purity_array = np.array(purity_array)
                            np.save(filepath_purity_array_checkpointing, purity_array)
                            
                            purity_RNN_sampling_error_array = np.array(purity_RNN_sampling_error_array)
                            np.save(filepath_purity_RNN_sampling_error_array_checkpointing,purity_RNN_sampling_error_array)

                            entropy_array = np.array(entropy_array)  # I think the size of the array is (numrealizations, d)
                            np.save(filepath_entropy_array_checkpointing, entropy_array)
                            
                            
                            indexWhereToStart = len(entropy_array) # numrealizations
                            found_higher = 1
                            break  # break out of for loop and move forward.
                        else:
                            print("checkpointing for entropy & purity for same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                                numrealiz_i," has resulted in purity, entropy and error ckpt files of same length but of NOT ENOUGH LENGTH for numrealizations = ",numrealizations,
                                ". Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
                    else:
                        print("checkpointing for entropy & purity for same simulations AT LOWER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in purity, entropy and/or error ckpt files of different lengths. Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
            elif purityLogSumExp_correlation_sampling_errors:
                if os.path.isfile(filepath_entropy_logSumExp_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    entropy_logSumExp_array = list(np.load(filepath_entropy_logSumExp_array_checkpointing_i))
                    
                    
                    # We are assuming the simulations at higher numrealizations values have been performed and reached a higher number
                    # of realizations than our present simulation, even if they haven't been completed yet.

                    #indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(entropy_array)
                    if len(entropy_logSumExp_array) >= numrealizations:
                        length_original = len(entropy_logSumExp_array)
                        entropy_logSumExp_array = entropy_logSumExp_array[:numrealizations]
                        print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                              ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                        if length_original < numrealiz_i:
                            print("There are actually less data points recorded than numrealiz_i =",numrealiz_i, "points, which is OK. It's still more data points than numrealizations =",numrealizations)
                            
                        
                        entropy_logSumExp_array = np.array(entropy_logSumExp_array)
                        np.save(filepath_entropy_logSumExp_array_checkpointing, entropy_logSumExp_array)
                        
                        
                        indexWhereToStart = len(entropy_logSumExp_array) # numrealizations
                        found_higher = 1
                        break  # break out of for loop and move forward.
                    else:
                        print("checkpointing for logSumExpfor same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in logSumExp ckpt file of NOT ENOUGH LENGTH for numrealizations = ",numrealizations,
                            ". Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
            else:
                if os.path.isfile(filepath_purity_array_checkpointing_i) and os.path.isfile(filepath_entropy_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    purity_array = list(np.load(filepath_purity_array_checkpointing_i))
                    entropy_array = list(np.load(filepath_entropy_array_checkpointing_i))
                    
                    # We are assuming the simulations at higher numrealizations values have been performed and reached a higher number
                    # of realizations than our present simulation, even if they haven't been completed yet.

                    if len(purity_array) == len(entropy_array): # good sign, sanity check, check if length of arrays is the same
                        #indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        if len(purity_array) >= numrealizations:
                            length_original = len(purity_array)
                            purity_array = purity_array[:numrealizations]
                            entropy_array = entropy_array[:numrealizations]
                            print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                                  ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                            if length_original < numrealiz_i:
                                print("There are actually less data points recorded than numrealiz_i =",numrealiz_i, "points, which is OK. It's still more data points than numrealizations =",numrealizations)
                                
                            purity_array = np.array(purity_array)
                            np.save(filepath_purity_array_checkpointing, purity_array)

                            entropy_array = np.array(entropy_array)  # I think the size of the array is (numrealizations, d)
                            np.save(filepath_entropy_array_checkpointing, entropy_array)
                            
                            indexWhereToStart = len(entropy_array) # numrealizations
                            found_higher = 1
                            break  # break out of for loop and move forward.
                        else:
                            print("checkpointing for entropy & purity for same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                                numrealiz_i," has resulted in purity and entropy ckpt files of same length but of NOT ENOUGH LENGTH for numrealizations = ",numrealizations,
                                ". Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
                    else:
                        print("checkpointing for entropy & purity for same simulations AT LOWER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in purity and entropy ckpt files of different lengths. Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)

        # If found == 0 no data at lower number of realizations has been found.
        if found_higher == 0:
            print("No data at HIGHER number of realizations has been found. Confirmed that we will now check for data at lower number of realizations.")
    
    
    
    
    
    
    
    
    
    
    
    

    # NOW WE KNOW IF SIMULATIONS HAVE ALREADY BEEN PERFORMED AT THE STIPULATED HYPERPARAMETERS, e.g. at a well-defined value of numrealizations.
    # We want to do one more thing. We want to know if simulations have been performed at the same exact hyperparameters but at a lower value
    # of numrealizations, so that we don't have to repeat any simulations that don't need to be repeated.
    # We want to test all values from 1 to (numrealizations-1) to see if simulations have already been performed.
    # ...
    # if indexWhereToStart = 0 after the if statements above, that means that no simulations have been performed as of yet at the given
    # hyperparameters, including at the desired value of numrealizations. So now we look at smaller values of numrealizations and see if
    # we have already performed simulations at those values.
    # ...
    # For now, we are just applying this to the entropy/purity calculations. We can apply this to other quantities later.
    # ...
    # UPDATE: if calcEntropyCleanUp = 1 (or True), then not calcEntropyCleanUp = False, and this if statement doesn't get executed.
    # In other words, if we are performing "clean-up" due to numerical issues on the cluster or wherever, let's not perform
    # ...
    '''
    LOWER # of numrealizations:
    '''
    if (indexWhereToStart == 0) and (found_higher == 0) and calcEntropy and (not calcEntropyCleanUp):  # means we found nothing in the present folder, so let's check folders with lower values of numrealizations
        print()
        print("NEXT. WE NOW CHECK to see if data for same hyperparameters but LOWER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at lower numbers of realizations, we will start from scratch.")
        found = 0
        for numrealiz_i in range(numrealizations - 1, 0, -1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations-1 to 1.
            
            if purity_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            elif purityLogSumExp_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_logSumExp_only/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            else: # purity_correlation_sampling_errors = 0
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                    
                    
                    
            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                        
            
            
            if purity_correlation_sampling_errors:
                # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly
                # (the checkpointing files should always exist)
                filepath_purity_array_checkpointing_i = ckpt_path_i + '/' + 'purity_array' + '_checkpointing' + '.npy'
                filepath_entropy_array_checkpointing_i = ckpt_path_i + '/entropy_array' + '_checkpointing' + '.npy'
                filepath_purity_RNN_sampling_error_array_checkpointing_i = ckpt_path_i + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
            elif purityLogSumExp_correlation_sampling_errors:
                #filepath_purity_RNN_sampling_error_array_checkpointing_i = ckpt_path_i + '/purity_' + 'RNN_sampling_error_array' + '_checkpointing' + '.npy'
                filepath_entropy_logSumExp_array_checkpointing_i = ckpt_path_i + '/entropy_logSumExp_array' + '_checkpointing' + '.npy'
            else:
                filepath_purity_array_checkpointing_i = ckpt_path_i + '/' + 'purity_array' + '_checkpointing' + '.npy'
                filepath_entropy_array_checkpointing_i = ckpt_path_i + '/entropy_array' + '_checkpointing' + '.npy'
            
            
            if purity_correlation_sampling_errors:
                # include error check
                if os.path.isfile(filepath_purity_array_checkpointing_i) and os.path.isfile(filepath_entropy_array_checkpointing_i) and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    purity_array = list(np.load(filepath_purity_array_checkpointing_i))
                    purity_RNN_sampling_error_array = list(np.load(filepath_purity_RNN_sampling_error_array_checkpointing_i))
                    entropy_array = list(np.load(filepath_entropy_array_checkpointing_i))

                    if len(purity_array) == len(entropy_array) and len(purity_array) == len(purity_RNN_sampling_error_array):
                        indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                              ", (same hyperparameters otherwise). Starting at realiz =",numrealiz_i)  # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                        found += 1
                        break  # break out of for loop and move forward.
                    else:
                        print("checkpointing for entropy & purity for same simulations AT LOWER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in purity, entropy and/or error ckpt files of different lengths. Continue checking for folders at lower numbers of realizations.")
            elif purityLogSumExp_correlation_sampling_errors:
                # include error check
                if os.path.isfile(filepath_entropy_logSumExp_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    
                    entropy_logSumExp_array = list(np.load(filepath_entropy_logSumExp_array_checkpointing_i))
                    indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    # so we want to start at i = 3, i.e. i = len(entropy_array)
                    print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                          ", (same hyperparameters otherwise). Starting at realiz =",numrealiz_i)  # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                    found += 1
                    break
                    
            
            else:
                if os.path.isfile(filepath_purity_array_checkpointing_i) and os.path.isfile(filepath_entropy_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    purity_array = list(np.load(filepath_purity_array_checkpointing_i))
                    entropy_array = list(np.load(filepath_entropy_array_checkpointing_i))

                    if len(purity_array) == len(entropy_array):
                        indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                              ", (same hyperparameters otherwise). Starting at realiz =",numrealiz_i)  # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                        found += 1
                        break  # break out of for loop and move forward.
                    else:
                        print("checkpointing for entropy & purity for same simulations AT LOWER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in purity and entropy ckpt files of different lengths. Continue checking for folders at lower numbers of realizations.")

        # If found == 0 no data at lower number of realizations has been found.
        if found == 0:
            print("No data at lower number of realizations has been found. Confirmed that we are starting from scratch, realization = 0.")

    

    
    '''
    HIGHER # of numrealizations: CORRELATION FUNCTION
    '''
    # Check if simulations have already been performed and possibly completed at higher number of realizations.
    # No simulations have been conducted in the present folder, so let's check for higher number of realizations with all other
    # hyperparameters held equal.
    found_higher = 0
    if (indexWhereToStart == 0) and calcCorr_1d_upTo_dist:
        print()
        print("Actually wait. First checking to see if data for same hyperparameters but HIGHER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at HIGHER numbers of realizations, we will check for LOWER number of realizations.")
        
        #found_higher = 0
        for numrealiz_i in range(numrealizations + 500, numrealizations, -1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations + 500 to numrealizations + 1.
            
            # Re-initialize arrays (in the event arrays of different lengths were found at a previous iteration of the loop, and so we want
            # to discard those realizations.)
            
            
            if purity_correlation_sampling_errors:
                corr_unconnected_avg_up_to_d_dict_of_arrays = {} # keys 'd = 1', 'd = 2', etc. up to d =L/2
                corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = {} # corresponding errors of mean
                magnetization_avg_array = []
                magnetization_error_of_mean_array = []
            else:
                corr_abs_avg_up_to_d_array = []
            
            
            
            if purity_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'

            else: # purity_correlation_sampling_errors = 0
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                    
                    
                    
            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                            
            
            
            if purity_correlation_sampling_errors:
                # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly
                # (the checkpointing files should always exist)
                filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i = ckpt_path_i + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_sampling_avg_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
                filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i = ckpt_path_i + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_error_of_mean_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
                filepath_sigmai_RNN_sampling_avg_array_checkpointing_i = ckpt_path_i + '/sigmai_RNN_sampling_avg_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'
                filepath_sigmai_RNN_error_of_mean_array_checkpointing_i = ckpt_path_i + '/sigmai_RNN_error_of_mean_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'
            else:
                filepath_corr_up_to_d_checkpointing_i = ckpt_path_i + '/' + 'corr_abs_avg_up_to_d' + str(d_up_to_corr) + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.npy'
            
            
            
            
            if purity_correlation_sampling_errors:
                if os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_RNN_sampling_avg_array_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_RNN_error_of_mean_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i, 'rb') as handle:
                        corr_unconnected_avg_up_to_d_dict_of_arrays = pickle.load(handle)
                    
                    with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i, 'rb') as handle:
                        corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = pickle.load(handle)
                    
                    assert(type(corr_unconnected_avg_up_to_d_dict_of_arrays) == dict)
                    assert(type(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays) == dict)
                    
                    
                    #corr_unconnected_avg_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i,allow_pickle=True).item() # .item() needed to convert from numpy array to dictionary
                    #corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i,allow_pickle=True).item()
                    
                    
                    magnetization_avg_array = list(np.load(filepath_sigmai_RNN_sampling_avg_array_checkpointing_i))
                    magnetization_error_of_mean_array = list(np.load(filepath_sigmai_RNN_error_of_mean_array_checkpointing_i))

                    #indexWhereToStart = len(magnetization_avg_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                    
                    
                    # We are assuming the simulations at higher numrealizations values have been performed and reached a higher number
                    # of realizations than our present simulation, even if they haven't been completed yet.

                    if len(magnetization_avg_array) == len(magnetization_error_of_mean_array) \
                    and len(magnetization_avg_array) == len(corr_unconnected_avg_up_to_d_dict_of_arrays['d=1']) \
                    and len(magnetization_avg_array) == len(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=1']): # good sign, sanity check, check if length of arrays is the same
                        #indexWhereToStart = len(entropy_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        if len(magnetization_avg_array) >= numrealizations:
                            length_original = len(magnetization_avg_array)
                            
                            # Each dictionary needs a loop
                            keys = list(corr_unconnected_avg_up_to_d_dict_of_arrays.keys())
                            for i in range(len(keys)):
                                corr_unconnected_avg_up_to_d_dict_of_arrays[keys[i]] = corr_unconnected_avg_up_to_d_dict_of_arrays[keys[i]][0:numrealizations]
                                corr_unconnected_error_of_mean_up_to_d_dict_of_arrays[keys[i]] = corr_unconnected_error_of_mean_up_to_d_dict_of_arrays[keys[i]][0:numrealizations]
                            
                            magnetization_avg_array = np.array(magnetization_avg_array) # shape (numrealizations, L)
                            magnetization_avg_array = magnetization_avg_array[:numrealizations]
                            magnetization_error_of_mean_array = np.array(magnetization_error_of_mean_array) # shape (numrealizations, L)
                            magnetization_error_of_mean_array = magnetization_error_of_mean_array[:numrealizations]
                            

                            print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                                  ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                            if length_original < numrealiz_i:
                                print("There are actually less data points recorded than numrealiz_i =",numrealiz_i, "points, which is OK. It's still more data points than numrealizations =",numrealizations)
                            
                            with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, 'wb') as handle:
                                pickle.dump(corr_unconnected_avg_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing, 'wb') as handle:
                                pickle.dump(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            #np.save(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, corr_unconnected_avg_up_to_d_dict_of_arrays)
                            #np.save(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing,corr_unconnected_error_of_mean_up_to_d_dict_of_arrays
                            np.save(filepath_sigmai_RNN_sampling_avg_array_checkpointing, magnetization_avg_array)
                            np.save(filepath_sigmai_RNN_error_of_mean_array_checkpointing, magnetization_error_of_mean_array)
                            
                            
                            indexWhereToStart = len(magnetization_avg_array) # numrealizations
                            found_higher = 1
                            break  # break out of for loop and move forward.
                        else:
                            print("checkpointing for correlation files (with error) for same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                                numrealiz_i," has resulted in ckpt files of same length but of NOT ENOUGH LENGTH for numrealizations = ",numrealizations,
                                ". Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
                    else:
                        print("checkpointing for correlation files (with error) for same simulations AT HIGHER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in ckpt files of different lengths. Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
            
            else:
                if os.path.isfile(filepath_corr_up_to_d_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    corr_abs_avg_up_to_d_array = list(np.load(filepath_corr_up_to_d_checkpointing_i))
                    
                    if len(corr_abs_avg_up_to_d_array) >= numrealizations:
                        length_original = len(corr_abs_avg_up_to_d_array)
                        corr_abs_avg_up_to_d_array = corr_abs_avg_up_to_d_array[:numrealizations]
                        
                        print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                              ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                        if length_original < numrealiz_i:
                            print("There are actually less data points recorded than numrealiz_i =",numrealiz_i, "points, which is OK. It's still more data points than numrealizations =",numrealizations)
                            
                        corr_abs_avg_up_to_d_array = np.array(corr_abs_avg_up_to_d_array)
                        np.save(filepath_corr_up_to_d, corr_abs_avg_up_to_d_array)

                      
                        indexWhereToStart = len(corr_abs_avg_up_to_d_array) # numrealizations
                        found_higher = 1
                        break  # break out of for loop and move forward.
                    else:
                        print("checkpointing for correlation file (no error) for same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                            numrealiz_i," has resulted in ckpt file of same length but of NOT ENOUGH LENGTH for numrealizations = ",numrealizations,
                            ". Continue checking for folders at lower numbers of realizations than realiz =",numrealiz_i)
                    
                   

        # If found == 0 no data at lower number of realizations has been found.
        if found_higher == 0:
            print("No data at HIGHER number of realizations has been found. Confirmed that we will now check for data at lower number of realizations.")
    
    
    
    

    
    
    
    '''
    LOWER # of Realizations, Correlation Function
    '''
    # REPEAT THE ABOVE IF STATEMENT FOR THE CORRELATION FUNCTION. Check if data exists at lower number of realizations.
    if (indexWhereToStart == 0) and calcCorr_1d_upTo_dist:  # means we found nothing in the present folder, so let's check folders with lower values of numrealizations
        print()
        print("Actually wait. First checking to see if data for same hyperparameters but LOWER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at lower numbers of realizations, we will start from scratch.")
        found = 0
        for numrealiz_i in range(numrealizations - 1, 0, -1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations-1 to 1.
            if purity_correlation_sampling_errors:
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/results_with_errors/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
            else: # purity_correlation_sampling_errors = 0
                if ARD_vs_MCMC == 'ARD':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                    elif exactOrMC == 'exact':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif ARD_vs_MCMC == 'MCMC':
                    if exactOrMC == 'MC':
                        path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                            f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                            f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                          + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                    
                    
                    
            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            
            
            if purity_correlation_sampling_errors:
                # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly
                # (the checkpointing files should always exist)
                filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i = ckpt_path_i + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_sampling_avg_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
                filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i = ckpt_path_i + f'/sigmai_sigmaj_d{d_up_to_corr}' + '_RNN_error_of_mean_dict' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.pickle'
                filepath_sigmai_RNN_sampling_avg_array_checkpointing_i = ckpt_path_i + '/sigmai_RNN_sampling_avg_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'
                filepath_sigmai_RNN_error_of_mean_array_checkpointing_i = ckpt_path_i + '/sigmai_RNN_error_of_mean_array' + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing.npy'

                if os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_RNN_sampling_avg_array_checkpointing_i) \
                and os.path.isfile(filepath_sigmai_RNN_error_of_mean_array_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already
                    with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i, 'rb') as handle:
                        corr_unconnected_avg_up_to_d_dict_of_arrays = pickle.load(handle)
                    
                    with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i, 'rb') as handle:
                        corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = pickle.load(handle)
                    
                    assert(type(corr_unconnected_avg_up_to_d_dict_of_arrays) == dict)
                    assert(type(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays) == dict)
                    #corr_unconnected_avg_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing_i,allow_pickle=True).item() # .item() needed to convert from numpy array to dictionary
                    #corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing_i,allow_pickle=True).item()
                    magnetization_avg_array = list(np.load(filepath_sigmai_RNN_sampling_avg_array_checkpointing_i))
                    magnetization_error_of_mean_array = list(np.load(filepath_sigmai_RNN_error_of_mean_array_checkpointing_i))
                    
                    
                    
                    if len(magnetization_avg_array) == len(magnetization_error_of_mean_array) \
                    and len(magnetization_avg_array) == len(corr_unconnected_avg_up_to_d_dict_of_arrays['d=1']) \
                    and len(magnetization_avg_array) == len(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=1']): # good sign, sanity check, check if length of arrays is the same
                        indexWhereToStart = len(magnetization_avg_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                        # so we want to start at i = 3, i.e. i = len(entropy_array)
                        print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                              ", (same hyperparameters otherwise). Starting at realiz =", numrealiz_i)
                              # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                        found += 1
                        break  # break out of for loop and move forward.
                    else:
                        print("checkpointing for corr files for same simulations AT LOWER NUMBER OF REALIZ, realiz = ",
                              numrealiz_i," has resulted in ckpt files of different lengths. Continue checking for folders at lower numbers of realizations.")
                    

                
            else:
                # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly                                                             
                # (the checkpointing files should always exist)                                                                                                                                       
                filepath_corr_up_to_d_checkpointing_i = ckpt_path_i + '/' + 'corr_abs_avg_up_to_d' + str(d_up_to_corr) + '_newSamples' + str(newSamplesOrNot_corr) + '_checkpointing' + '.npy'
                if os.path.isfile(filepath_corr_up_to_d_checkpointing_i):
                    # if statement to check if ckpt file exists (it should if simulations have been completed already                                                                                 
                    corr_abs_avg_up_to_d_array = list(np.load(filepath_corr_up_to_d_checkpointing_i))
                    
                    indexWhereToStart = len(corr_abs_avg_up_to_d_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done                                                    
                    # so we want to start at i = 3, i.e. i = len(entropy_array)                                                                                                                       
                    print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                          ", (same hyperparameters otherwise). Starting at realiz =", numrealiz_i)
                          # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.                                                                                            
                    found += 1
                    break  # break out of for loop and move forward. 

        # If found == 0 no data at lower number of realizations has been found.
        if found == 0:
            print(
                "No data at lower number of realizations has been found. Confirmed that we are starting from scratch, realization = 0.")

    '''
    HIGHER # of numrealizations: ENTANG SPECTRUM
    '''
    # Check if simulations have already been performed and possibly completed at higher number of realizations.
    # No simulations have been conducted in the present folder, so let's check for higher number of realizations with all other
    # hyperparameters held equal.
    found_higher = 0
    if (indexWhereToStart == 0) and calcEntangSpectrum:
        print()
        print("Actually wait. First checking to see if data for same hyperparameters but HIGHER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at HIGHER numbers of realizations, we will check for LOWER number of realizations.")

        # found_higher = 0
        for numrealiz_i in range(numrealizations + 10000, numrealizations,-1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations + 500 to numrealizations + 1.

            # Re-initialize arrays (in the event arrays of different lengths were found at a previous iteration of the loop, and so we want
            # to discard those realizations.)

            eigenvalues_rho_A = []

            if ARD_vs_MCMC == 'ARD':
                if exactOrMC == 'MC':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif exactOrMC == 'exact':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
            elif ARD_vs_MCMC == 'MCMC':
                if exactOrMC == 'MC':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                        + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                elif exactOrMC == 'exact':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming


            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'

            filepath_eigenvalues_rho_A_array_checkpointing_i = ckpt_path_i + '/eigenvalues_rho_A_array_checkpointing.npy'

            if os.path.isfile(filepath_eigenvalues_rho_A_array_checkpointing_i):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                eigenvalues_rho_A = list(np.load(filepath_eigenvalues_rho_A_array_checkpointing_i))
                print("len(eigenvalues_rho_A) HIGHER =", len(eigenvalues_rho_A))
                n_completed = len(eigenvalues_rho_A) // 2**(L//2)
                #completed = ",n_completed)

                # We are assuming the simulations at higher numrealizations values have been performed and reached a higher number
                # of realizations than our present simulation, even if they haven't been completed yet.

                if n_completed >= numrealizations:
                    length_original = n_completed
                    index_end = numrealizations*(2**(L//2))
                    eigenvalues_rho_A = eigenvalues_rho_A[:index_end]
                    print("Data was found at a HIGHER NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                          ", (same hyperparameters otherwise). No need to perform this simulation after all.")
                    print()
                    print("len(eigenvalues_rho_A) =", len(eigenvalues_rho_A))
                    if length_original < numrealiz_i:
                        print()
                        print("There are actually less data points recorded than numrealiz_i =", numrealiz_i,
                              "points, which is OK. It's still more data points than numrealizations =",
                              numrealizations)

                    eigenvalues_rho_A = np.array(eigenvalues_rho_A)
                    np.save(filepath_eigenvalues_rho_A_array_checkpointing, eigenvalues_rho_A)



                    indexWhereToStart = n_completed  # numrealizations
                    found_higher = 1
                    break  # break out of for loop and move forward.
                else:
                    print(
                        "checkpointing for entang spec for same simulation AT HIGHER NUMBER OF REALIZ, realiz = ",
                        numrealiz_i,
                        " has resulted in entang spec ckpt files of NOT ENOUGH LENGTH for numrealizations = ",
                        numrealizations,
                        ". Continue checking for folders at lower numbers of realizations than realiz =",
                        numrealiz_i)

        # If found == 0 no data at lower number of realizations has been found.
        if found_higher == 0:
            print("No data at HIGHER number of realizations has been found. Confirmed that we will now check for data at lower number of realizations.")

    '''
    LOWER # of numrealizations: ENTANG SPEC
    '''
    if (indexWhereToStart == 0) and (found_higher == 0) and calcEntangSpectrum:
        # means we found nothing in the present folder and at higher realizations, so let's check folders with lower values of numrealizations
        print()
        print("NEXT. WE NOW CHECK to see if data for same hyperparameters but LOWER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at lower numbers of realizations, we will start from scratch.")
        found = 0
        for numrealiz_i in range(numrealizations - 1, 0,-1):  # we want to find the largest value of numrealizations at which we have already performed simulations.

            if ARD_vs_MCMC == 'ARD':
                if exactOrMC == 'MC':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
                elif exactOrMC == 'exact':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
            elif ARD_vs_MCMC == 'MCMC':
                if exactOrMC == 'MC':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming \
                                        + f'/nchains_{n_chains}_ndiscchain_{n_disc_chain}_sweepsize_{sweep_size}'
                elif exactOrMC == 'exact':
                    path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}_MCMC/rnn_cell_{rnn_cell}/' \
                                        f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                        f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming


            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'

            filepath_eigenvalues_rho_A_array_checkpointing_i = ckpt_path_i + '/eigenvalues_rho_A_array_checkpointing.npy'

            if os.path.isfile(filepath_eigenvalues_rho_A_array_checkpointing_i):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                eigenvalues_rho_A = list(np.load(filepath_eigenvalues_rho_A_array_checkpointing_i))
                indexWhereToStart = len(eigenvalues_rho_A) // 2**(L//2)
                print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                      ", (same hyperparameters otherwise). Starting at realiz =",
                      numrealiz_i)  # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                print("len(eigenvalues_rho_A) LOWER =", len(eigenvalues_rho_A))
                found += 1
                break



        # If found == 0 no data at lower number of realizations has been found.
        if found == 0:
            print("No data at lower number of realizations has been found. Confirmed that we are starting ENTANG SPEC calc from scratch, realization = 0.")








    # REPEAT THE ABOVE IF STATEMENT FOR THE Adjacent Gap Ratio. Check if data exists at lower number of realizations.
    if (indexWhereToStart == 0) and calcAdjacentGapRatio:  # means we found nothing in the present folder, so let's check folders with lower values of numrealizations
        print()
        print("Actually wait. First checking to see if data for same hyperparameters but LOWER NUMBER OF REALIZATIONS exists in other folders.")
        print("If we find nothing at lower numbers of realizations, we will start from scratch.")
        found = 0
        for numrealiz_i in range(numrealizations - 1, 0, -1):  # we want to find the largest value of numrealizations at which we have already performed
            # simulations... loop goes from numrealizations-1 to 1.
            if exactOrMC == 'MC':
                path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                      f'{positive_or_complex}/L_{L}/{exactOrMC}_ns_{nsamples}/' \
                                      f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming
            elif exactOrMC == 'exact':
                path_name_i = 'RNN' + f'{lattice}{"_WS" if weight_sharing else ""}{"_AR" if autoreg else "_MCMC"}/rnn_cell_{rnn_cell}/' \
                                      f'{positive_or_complex}/L_{L}/{exactOrMC}/' \
                                      f'numrealiz_{numrealiz_i}/dhidden_{dhidden}_width_{width:.2f}/' + dtype_for_file_naming

            if rnn_cell == 'GRU_Mohamed' or rnn_cell == 'GRU':
                if activation_fn == 'identity' or gate_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_gateFn_{gate_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            elif rnn_cell == 'RNN':
                if activation_fn == 'identity':
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_actFn_{activation_fn}_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
                else:  # neither activation_fn nor gate_fn is identity
                    if softmaxOnOrOff == 'on':
                        ckpt_path_i = f'./no_hcorr_bpur_data/{path_name_i}'
                    elif softmaxOnOrOff == 'off':
                        if modulusOnOrOff == 'on':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}/{path_name_i}'
                        elif modulusOnOrOff == 'off':
                            ckpt_path_i = f'./no_hcorr_bpur_data_softmax_{softmaxOnOrOff}_modulus_{modulusOnOrOff}_signOfProbsIntoPhase_{signOfProbsIntoPhase}/{path_name_i}'
            
            # Pull up the checkpointing files instead of the non-checkpointing files, in case the final files weren't saved correctly
            # (the checkpointing files should always exist)
            filepath_adjacent_energy_gap_ratio_array_checkpointing_i = ckpt_path_i + '/adjacent_energy_gap_ratio_array_checkpointing.npy'
            #filepath_adjacent_energy_gap_ratio_array = ckpt_path + '/adjacent_energy_gap_ratio_array.npy'
            
            
            
            if os.path.isfile(filepath_adjacent_energy_gap_ratio_array_checkpointing_i):
                # if statement to check if ckpt file exists (it should if simulations have been completed already
                adjacent_energy_gap_ratio_array = list(np.load(filepath_adjacent_energy_gap_ratio_array_checkpointing_i))

                indexWhereToStart = len(adjacent_energy_gap_ratio_array)  # if 3 simulations have been completed, that means i = 0, 1, 2 have been done
                # so we want to start at i = 3, i.e. i = len(entropy_array)
                print("Data was found at NEXT HIGHEST NUMBER OF REALIZATIONS, realiz = ", numrealiz_i,
                      ", (same hyperparameters otherwise). Starting at realiz =", numrealiz_i)
                        # indexWhereToStart = len(entropy_array) should equal numrealiz_i, in this case.
                found += 1
                break  # break out of for loop and move forward.

        # If found == 0 no data at lower number of realizations has been found.
        if found == 0:
            print(
                "No data at lower number of realizations has been found. Confirmed that we are starting from scratch, realization = 0.")

    # ACTIVATION AND GATE FUNCTIONS TO PASS TO THE MODEL.

    if activation_fn == 'tanh':  # this will be default
        activation_function = tanh
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

    # NEW SPOT FOR MODEL BUILDING
    if dtype == 'jax.numpy.float16':  # Using these if statements because don't know code to convert string into a jax.numpy.float.
        # param_dtype = jax.numpy.float32
        model = RNN1D(hilbert=hilbert,  # these models can take hilbert as an input because of the AbstractARNN subclass
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
                      L=L,
                      numSpins_A=numSpins_A,
                      # we don't actually use numSpins_A in RNN1D anymore... so can drop this in the future.
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

    if ARD_vs_MCMC == 'ARD':
        sampler = nk.sampler.ARDirectSampler(hilbert)#,dtype=jax.numpy.complex64)  # autoregressive sampler of netket
        phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=nsamples,chunk_size=chunk_size)  # , sampler_seed=random.key(0))
    elif ARD_vs_MCMC == 'MCMC':
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, sweep_size=sweep_size)
        phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=nsamples, n_discard_per_chain=n_disc_chain,
                             chunk_size=chunk_size)  # , sampler_seed=random.key(1))

    '''
    OK we will write a code block for the entropy clean up calculations. This is a bit disorganized but we have had dumb numerical issues.
    So let's go with it like this for now. We will only "clean up" the entropy/purity results when all the other calc... variables are 0.
    '''

    if calcEntropyCleanUp:  # if calcEntropyCleanUp == 1, none of the other calc variables will be 1 (all will be 0).
        print("Performing purity/entropy clean-up. First checking to see if there is anything to clean up")
        # We only want to clean up purity/entropy values if the final files of the original simulations already exist. So check
        # if they exist and then afterwards we will check if there are values to clean up.
        if os.path.isfile(filepath_purity_array) and os.path.isfile(filepath_entropy_array):
            print("Final files exist. Search for unphysical purities.")
            purity_array = np.load(filepath_purity_array)
            entropy_array = np.load(filepath_entropy_array)
            assert (len(purity_array) == numrealizations)
            # if the assertion yields no errors, move forward
            indices_unphysical_purities = np.where(purity_array > 1.0001)  # let's call anything above 1.0001 unphysical. The
            # negative purity values that sometimes (rarely) result are
            # being dealt with elsewhere. Sometimes, you get a purity
            # > 1 like 1.00000000006 due to machine precision.
            # Convert to lists because jax is giving me issues with np.arrays
            purity_array = list(purity_array)
            entropy_array = list(entropy_array)

            indices_unphysical_purities = indices_unphysical_purities[0]  # because np.where produces array of shape (1,number of indices)
            if len(indices_unphysical_purities) == 0:
                print("None of the purity values are > than 1.0001. Ending clean-up.")
                return ckpt_path  # end simulation
            elif len(indices_unphysical_purities) > 0:
                # So there is some shit to clean-up. Let's calculate the purity/entropy at the relevant indices (model already) defined
                # and then let's update the checkpointing and final files. Let's print shit along the way.
                print("At least one purity value is > than 1.0001. Starting clean-up.")
                for ind in indices_unphysical_purities:
                    print()
                    # start = time.time()
                    print("realiz =", ind, ", purity_array[realiz] =", purity_array[ind])
                    SEED = int(L * dhidden * width * 100000 + ind)  # Seed for parameter initialization
                    # ROELAND: Call the init function with the seed:
                    phi.init(random.key(SEED))  # This initializes the parameters
                    # Hack to reset the sampler seed, which Jannes taught me.
                    SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + ind)
                    phi._sampler_seed = random.key(SAMPLER_SEED)
                    phi.sampler = sampler
                    # Parameters
                    parameters = phi.parameters
                    # Print
                    # print("realization to be cleaned up = ",ind)
                    new_nsamples = phi.n_samples
                    assert (exactOrMC == 'MC')  # make sure that this clean-up is only happening when importance sampling is used...
                    # It is not needed for exact sampling / exact summation.
                    # Entropy calculation
                    samples_jax_1 = phi.sample()
                    samples_jax_2 = phi.sample()

                    if numpy_or_jax == 'numpy':  # we are unlikely to ever "clean up" using the numpy method. But have it here in case.
                        '''
                        NUMPY
                        '''
                        # start_np = time.time()
                        samples1 = np.reshape(samples_jax_1, (new_nsamples, L))
                        samples2 = np.reshape(samples_jax_2, (new_nsamples, L))

                        # For some realizations of disorder, we get a negative estimate of purity. It's probably cause we need more samples. Instead, I'm
                        # going to try generating new samples until we get a positive estimate of purity. Hopefully this doesn't happen too often.
                        
                        if purity_correlation_sampling_errors:
                            # Case purity_correlation_sampling_errors = 1 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            # Calculate purity and entropy and save them
                            purity, purity_error, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                                parameters=parameters,
                                                                                numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                                purity_sampling_error=purity_correlation_sampling_errors,
                                                                                entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if purity <= 0:
                                while purity <= 0:
                                    print("purity estimate is negative. Trying again with a different set of samples.")
                                    samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                    samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                    
                                    purity, purity_error, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                                        parameters=parameters,
                                                                                        numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                                        purity_sampling_error=purity_correlation_sampling_errors,
                                                                                        entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)

                                        

                                    num_purity_calculations += 1
                        elif purityLogSumExp_correlation_sampling_errors:
                            # Case purity_correlation_sampling_errors = 0 
                            # purityLogSumExp_correlation_sampling_errors = 1
                            entropy_logSumExp = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                            parameters=parameters,
                                                                            numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                        else:
                            # Case purity_correlation_sampling_errors = 0 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            
                            purity, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                              parameters=parameters,
                                                              numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                              purity_sampling_error=purity_correlation_sampling_errors,
                                                              entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if purity <= 0:
                                while purity <= 0:
                                    print("purity estimate is negative. Trying again with a different set of samples.")
                                    samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                    samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                    purity, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                                        parameters=parameters,
                                                                                        numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                                        purity_sampling_error=purity_correlation_sampling_errors,
                                                                                        entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                    num_purity_calculations += 1
                            
                        
                        # This technique helps and seems to work.
                    elif numpy_or_jax == 'jax':
                        '''
                        JAX
                        '''
                        # start_jax = time.time()
                        if L > L_chunk_threshold and chunk_size is not None:
                            print(f"chunking renyi entropy calculation in chunks of {chunk_size}")
                            if purity_correlation_sampling_errors:
                                # Case purity_correlation_sampling_errors = 1, 
                                # purityLogSumExp_correlation_sampling_errors = 0
                                purity, purity_error, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                            numSpins_A,
                                                                                            nsamples,
                                                                                            chunk_size,
                                                                                            phi.model_state,
                                                                                            phi.parameters,
                                                                                            inputs1=samples_jax_1,
                                                                                            inputs2=samples_jax_2,
                                                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations = 1
                                if purity <= 0:
                                    while purity <= 0:
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                        samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                        samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                        purity, purity_error, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                                    numSpins_A,
                                                                                                    nsamples,
                                                                                                    chunk_size,
                                                                                                    phi.model_state,
                                                                                                    phi.parameters,
                                                                                                    inputs1=samples_jax_1,
                                                                                                    inputs2=samples_jax_2,
                                                                                                    purity_sampling_error=purity_correlation_sampling_errors,
                                                                                                    entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        num_purity_calculations += 1
                                        
                            elif purityLogSumExp_correlation_sampling_errors:
                                # Case purity_correlation_sampling_errors = 0, 
                                # purityLogSumExp_correlation_sampling_errors = 1
                                entropy_logSumExp = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                numSpins_A,
                                                                                nsamples,
                                                                                chunk_size,
                                                                                phi.model_state,
                                                                                phi.parameters,
                                                                                inputs1=samples_jax_1,
                                                                                inputs2=samples_jax_2,
                                                                                purity_sampling_error=purity_correlation_sampling_errors,
                                                                                entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations = 1
                            else:
                                # Case purity_correlation_sampling_errors = 0, 
                                # purityLogSumExp_correlation_sampling_errors = 0
                                purity, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                              numSpins_A,
                                                                              nsamples,
                                                                              chunk_size,
                                                                              phi.model_state,
                                                                              phi.parameters,
                                                                              inputs1=samples_jax_1,
                                                                              inputs2=samples_jax_2,
                                                                              purity_sampling_error=purity_correlation_sampling_errors,
                                                                              entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                
                                num_purity_calculations = 1
                                if purity <= 0:
                                    while purity <= 0:
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                        samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                        samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                        purity, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                      numSpins_A,
                                                                                      nsamples,
                                                                                      chunk_size,
                                                                                      phi.model_state,
                                                                                      phi.parameters,
                                                                                      inputs1=samples_jax_1,
                                                                                      inputs2=samples_jax_2,
                                                                                      purity_sampling_error=purity_correlation_sampling_errors,
                                                                                      entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        num_purity_calculations += 1

                        else:
                            # Case purity_correlation_sampling_errors = 1, 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            if purity_correlation_sampling_errors:
                                purity, purity_error, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                                    phi.parameters, inputs1=samples_jax_1,
                                                                                    inputs2=samples_jax_2,
                                                                                    purity_sampling_error=purity_correlation_sampling_errors,
                                                                                    entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations = 1
                                if purity <= 0:
                                    while purity <= 0:
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                        samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                        samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                        purity, purity_error, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                                            phi.parameters, inputs1=samples_jax_1,
                                                                                            inputs2=samples_jax_2,
                                                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        num_purity_calculations += 1
                                        
                            elif purityLogSumExp_correlation_sampling_errors:
                                # Case purity_correlation_sampling_errors = 0, 
                                # purityLogSumExp_correlation_sampling_errors = 1
                                entropy_logSumExp = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                        phi.parameters, inputs1=samples_jax_1,
                                                                        inputs2=samples_jax_2,
                                                                        purity_sampling_error=purity_correlation_sampling_errors,
                                                                        entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations = 1
                            else:
                                # Case purity_correlation_sampling_errors = 0, 
                                # purityLogSumExp_correlation_sampling_errors = 0
                                purity, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                      phi.parameters, inputs1=samples_jax_1,
                                                                      inputs2=samples_jax_2,
                                                                      purity_sampling_error=purity_correlation_sampling_errors,
                                                                      entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations = 1
                                if purity <= 0:
                                    while purity <= 0:
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                        samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                        samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                        purity, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                              phi.parameters, inputs1=samples_jax_1,
                                                                              inputs2=samples_jax_2,
                                                                              purity_sampling_error=purity_correlation_sampling_errors,
                                                                              entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        num_purity_calculations += 1
                                
                        # assert np.allclose(purity, purity_np)
                        # assert np.allclose(entropy, entropy_np)
                        # print(f"Jax time: {time.time() - start_jax}")

                        # For some realizations of disorder, we get a negative estimate of purity. It's probably cause we need more samples. Instead, I'm
                        # going to try generating new samples until we get a positive estimate of purity. Hopefully this doesn't happen too often.
                        
                        # This technique helps and works.

                    # Save
                    print("num_purity_calculations = ", num_purity_calculations)
                    if purityLogSumExp_correlation_sampling_errors == 0:
                        print("old purity =", purity_array[ind])
                        print("old entropy =", purity_array[ind])
                        print("new purity =", purity)
                        print("new entropy =", entropy)
                        print()
                        purity_array[ind] = purity  # replace unphysical purity in purity_array with latest purity estimate, hopefully a physically reasonable one
                        entropy_array[ind] = entropy  # same as for purity...
                        '''
                        SAVE AFTER EVERY REALIZATION ITERATION (Checkpointing)
                        '''
                        # purity_array = np.array(purity_array)
                        np.save(filepath_purity_array_checkpointing, purity_array)

                        # entropy_array = np.array(entropy_array) # I think the size of the array is (numrealizations, d)
                        np.save(filepath_entropy_array_checkpointing, entropy_array)
                        # Now the clean up is done. Time to save the final files.
                        # Store / save the clean-up arrays in final files (not checkpointing).
                        print()
                        print("Final statistics")
                        print()
                        # In theory no need to re-load from the checkpointing files, but do it anyways.
                        purity_array = np.load(filepath_purity_array_checkpointing)
                        entropy_array = np.load(filepath_entropy_array_checkpointing)

                        print("average purity =", np.mean(purity_array))
                        print("average entropy =", np.mean(entropy_array))

                        # purity_std = np.std(purity_array)
                        # entropy_std = np.std(entropy_array)

                        np.save(filepath_purity_array, purity_array)
                        np.save(filepath_entropy_array, entropy_array)
                        return ckpt_path


        else:
            print("The final files don't already exist. Exiting clean-up simulation.")
            return ckpt_path

    '''
    The loop over realizations below will only be run when calcEntropyCleanUp is not run.
    '''

    # vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=SEED, sampler_seed=SEED)
    
    print("Starting from realization =", indexWhereToStart)
    # indexWhereToStart = 0
    print()
    print("numpy_or_jax: ", numpy_or_jax)
    print("Calculating errors:",bool(purity_correlation_sampling_errors)) # True or False
    print("Calculating logSumExp of log(swap) values:",bool(purityLogSumExp_correlation_sampling_errors))
    # New quantities needed for MCMC:
    if ARD_vs_MCMC == 'ARD':
        print("sampler: ", ARD_vs_MCMC)
    elif ARD_vs_MCMC == 'MCMC':
        print("sampler: ", ARD_vs_MCMC)
        print("n_chains = ", n_chains)
        print("n_discard_per_chain = ", n_disc_chain)
        print("sweep_size = ", sweep_size)

    for i in range(indexWhereToStart, numrealizations):
        start = time.time()
        # SEED = i
        SEED = int(L * dhidden * width * 100000 + i)

        # ROELAND: Call the init function with the seed:
        phi.init(random.key(SEED))#,dtype=jax.numpy.complex64)  # This initializes the parameters

        # Hack to reset the sampler seed, which Jannes taught me.
        SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + i)
        # SAMPLER_SEED = int(0)
        phi._sampler_seed = random.key(SAMPLER_SEED)
        phi.sampler = sampler
        phi.reset() # reset variational state just in case. Eliminate samples from previous realization.

        # phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=nsamples, seed = random.key(SEED), chunk_size = chunk_size, sampler_seed=random.key(7))

        # netket variational state
        '''
        I've just discovered that defining the variational state automatically initializes the parameters. So let's go.
        '''
        parameters = phi.parameters
        if i == indexWhereToStart:
            print(f"Number of parameters {phi.n_parameters}")

        print()
        print("realization =", i)

        new_nsamples = phi.n_samples  # this is only if we use MC.
        '''
        LEARNINGS. the "seed" of nk.vqs.MCSTATE (not sampler_seed) is the same seed as the seed that goes into model.init(seed,...). Both
        these seeds seed the random number generators used to initialize the variational parameters of the model. As long as these two seeds
        are the same, the parameters are initialized to the same values using the initializers inherent to the model.
        '''

        '''
        CALCULATE THE ENTROPY IF AND ONLY IF calcEntropy == True AND THE RELEVANT FILES DON'T ALREADY EXIST
        '''
        if calcEntropy and ((not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array))):
            
            if purity_correlation_sampling_errors:
                purity_array = list(purity_array)
                entropy_array = list(entropy_array)
                purity_RNN_sampling_error_array = list(purity_RNN_sampling_error_array)
            elif purityLogSumExp_correlation_sampling_errors:
                entropy_logSumExp_array = list(entropy_logSumExp_array)
            else:
                purity_array = list(purity_array)
                entropy_array = list(entropy_array)
                
            

            if (exactOrMC == 'exact') and (dhidden != 0) and (width != 0.0):
                # if we are here, then embark on the entropy calculations.
                # print("configs =",configs,type(configs))
                shape = tuple(np.full(L, 2))
                shape = tuple(np.append(1, shape))
                # print("shape =",shape)
                log_psi = model.apply({'params': parameters}, configs_full)
                psi = jnp.exp(log_psi)

                if ARD_vs_MCMC == 'MCMC': # unnormalized wavefunction
                    norm_sq = jnp.sum(psi * jnp.conj(psi))
                    norm = jnp.sqrt(norm_sq)
                    psi = psi / norm

                psi = np.reshape(psi, shape)

                # print(psi,psi.shape)
                keep = list(np.arange(0, numSpins_A))
                dims = list(np.full(L, 2))
                dims = [1] + dims
                # print("dims =",dims)
                rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]
                # print(rho_A)
                # print("psi_A =",psi_A,psi_A.shape)
                rho_A_sq = jnp.einsum('ij,jk->ik', rho_A, rho_A)
                purity = jnp.trace(rho_A_sq)

                purity = purity.real
                entropy = -jnp.log(purity)
                
                num_purity_calculations = 1

                # In order to make the QGT calculations exactly equal for exactOrMC=='MC' or 'exact', we need
                # to sample from phi the same number of times before the QGT function is called. Because, ultimately
                # the elements of the QGT are calculated by Monte Carlo sampling, and MC sampling is done using
                # a variational state with random number generators that require seeding- so if phi was already used
                # to generate samples before the QGT calculation, then the random number is seeded differently for the MC
                # case compared to the Exact case where no sample generation is required a priori, when it comes to
                # producing the QGT and calculating its rank.
                # ...
                # samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                # samples2 = np.reshape(phi.sample(), (new_nsamples, L))

            elif (exactOrMC == 'MC') and (dhidden != 0) and (width != 0.0):

                # ROELAND: You're converting things from jaxlib.xla_extension.ArrayImpl to numpy arrays by calling
                # np.reshape on phi.sample. This moves things from GPU->CPU, which you may not want.
                # I've written a Jax function that calculates the Renyi entropy on the GPU
                samples_jax_1 = phi.sample()
                print("samples_jax_1[0:5] =",samples_jax_1[0,0:5])
                print("samples_jax_1[1000] =", samples_jax_1[0,999])
                samples_jax_2 = phi.sample()


                if numpy_or_jax == 'numpy':  
                    '''
                    NUMPY
                    '''
                    # start_np = time.time()
                    samples1 = np.reshape(samples_jax_1, (new_nsamples, L))
                    samples2 = np.reshape(samples_jax_2, (new_nsamples, L))

                    
                    # print(f"Numpy time: {time.time() - start_np}")

                    # For some realizations of disorder, we get a negative estimate of purity. It's probably cause we need more samples. Instead, I'm
                    # going to try generating new samples until we get a positive estimate of purity. Hopefully this doesn't happen too often.
                    
                    if purity_correlation_sampling_errors:
                        # Case purity_correlation_sampling_errors = 1 
                        # purityLogSumExp_correlation_sampling_errors = 0
                        # Calculate purity and entropy and save them
                        purity, purity_error, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                            parameters=parameters,
                                                                            numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                        num_purity_calculations = 1
                        if purity <= 0:
                            while purity <= 0:
                                print("purity estimate is negative. Trying again with a different set of samples.")
                                samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                purity, purity_error, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                                    parameters=parameters,
                                                                                    numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                                    purity_sampling_error=purity_correlation_sampling_errors,
                                                                                    entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)

                                    

                                num_purity_calculations += 1
                    elif purityLogSumExp_correlation_sampling_errors:
                        # Case purity_correlation_sampling_errors = 0 
                        # purityLogSumExp_correlation_sampling_errors = 1
                        entropy_logSumExp = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                            parameters=parameters,
                                                            numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                        num_purity_calculations = 1
                    else:
                        # Case purity_correlation_sampling_errors = 0 
                        # purityLogSumExp_correlation_sampling_errors = 0
                        purity, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                          parameters=parameters,
                                                          numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                          purity_sampling_error=purity_correlation_sampling_errors,
                                                          entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                        num_purity_calculations = 1
                        if purity <= 0:
                            while purity <= 0:
                                print("purity estimate is negative. Trying again with a different set of samples.")
                                samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                                samples2 = np.reshape(phi.sample(), (new_nsamples, L))
                                purity, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                  parameters=parameters,
                                                                  numSpins_A=numSpins_A, seed=SEED, exactOrMC=exactOrMC,
                                                                  purity_sampling_error=purity_correlation_sampling_errors,
                                                                  entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                num_purity_calculations += 1
                    # This technique helps and seems to work.

                elif numpy_or_jax == 'jax':
                    '''
                    JAX
                    '''
                    # start_jax = time.time()
                    if L > L_chunk_threshold and chunk_size is not None:
                        print(f"chunking renyi entropy calculation in chunks of {chunk_size}")
                        if purity_correlation_sampling_errors:
                            # Case purity_correlation_sampling_errors = 1, 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            purity, purity_error, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                        numSpins_A,
                                                                                        nsamples,
                                                                                        chunk_size,
                                                                                        phi.model_state,
                                                                                        phi.parameters,
                                                                                        inputs1=samples_jax_1,
                                                                                        inputs2=samples_jax_2,
                                                                                        purity_sampling_error=purity_correlation_sampling_errors,
                                                                                        entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if (purity <= 0 or purity > 1.0000001) or jnp.isnan(purity):
                                while (purity <= 0 or purity > 1.0000001 or jnp.isnan(purity)):
                                    if num_purity_calculations >= 2:


                                        # OK nothing has worked so far. So try re-initializing the state at the exact same parameters.
                                        # Next step will be to re-initialize to different parameters.
                                        SEED += 3123149  # Change the seed used to initialize the param
                                        # += 1 means you have the wavefunction from the next realization
                                        # SEED = int(L * dhidden * width * 100000 + i)

                                        # ROELAND: Call the init function with the seed:
                                        phi.init(random.key(SEED))  # This initializes the parameters

                                        # Hack to reset the sampler seed, which Jannes taught me.
                                        SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + i)
                                        # SAMPLER_SEED = int(0)
                                        phi._sampler_seed = random.key(SAMPLER_SEED)
                                        phi.sampler = sampler
                                        print("TOO MANY PURITY CALCULATIONS. LET's TRY Re-Initializing the state, with different parameters.")
                                        # print("phi.n_samples =",phi.n_samples)

                                    if purity <= 0:
                                        print("negative purity = ", purity)
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                    elif purity > 1.0000001:
                                        print("unphysical purity grtr than 1.0000001 =", purity)
                                        print("purity estimate is much greater than 1. Repeat with the same variational state and diff. samples")
                                    elif jnp.isnan(purity):
                                        print("purity =", purity)
                                        print("nan value produced for purity. Trying again with a different set of samples.")
                                    samples_jax_1 = phi.sample()  # now you're doing it with more
                                    samples_jax_2 = phi.sample()
                                    
                                    purity, purity_error, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                                numSpins_A,
                                                                                                nsamples,
                                                                                                chunk_size,
                                                                                                phi.model_state,
                                                                                                phi.parameters,
                                                                                                inputs1=samples_jax_1,
                                                                                                inputs2=samples_jax_2,
                                                                                                purity_sampling_error=purity_correlation_sampling_errors,
                                                                                                entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        
                                    num_purity_calculations += 1
                                    # This technique helps and works.
                                # once we are out of the while loop, check to see if we have changed the number of samples. If so, change it
                                # back, to prepare for the subsequent calculations.
                                # we need to reset the number of samples to the original nsamples.
                                if phi.n_samples != nsamples:
                                    phi.n_samples = nsamples  # add 5000 samples
                                    phi.reset()
                                    #print("We are out of the while loop. number of samples changed back to ", phi.n_samples)
                            
                            
                            
                                    
                        elif purityLogSumExp_correlation_sampling_errors:
                            # Case purity_correlation_sampling_errors = 0, 
                            # purityLogSumExp_correlation_sampling_errors = 1
                            entropy_logSumExp = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                            numSpins_A,
                                                                            nsamples,
                                                                            chunk_size,
                                                                            phi.model_state,
                                                                            phi.parameters,
                                                                            inputs1=samples_jax_1,
                                                                            inputs2=samples_jax_2,
                                                                            purity_sampling_error=purity_correlation_sampling_errors,
                                                                            entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                        else:
                            # Case purity_correlation_sampling_errors = 0, 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            purity, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                          numSpins_A,
                                                                          nsamples,
                                                                          chunk_size,
                                                                          phi.model_state,
                                                                          phi.parameters,
                                                                          inputs1=samples_jax_1,
                                                                          inputs2=samples_jax_2,
                                                                          purity_sampling_error=purity_correlation_sampling_errors,
                                                                          entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if (purity <= 0 or purity > 1.0000001) or jnp.isnan(purity):
                                while (purity <= 0 or purity > 1.0000001 or jnp.isnan(purity)):
                                    if num_purity_calculations >= 2:


                                        # OK nothing has worked so far. So try re-initializing the state at the exact same parameters.
                                        # Next step will be to re-initialize to different parameters.
                                        SEED += 3123149  # Change the seed used to initialize the param
                                        # += 1 means you have the wavefunction from the next realization
                                        # SEED = int(L * dhidden * width * 100000 + i)

                                        # ROELAND: Call the init function with the seed:
                                        phi.init(random.key(SEED))  # This initializes the parameters

                                        # Hack to reset the sampler seed, which Jannes taught me.
                                        SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + i)
                                        # SAMPLER_SEED = int(0)
                                        phi._sampler_seed = random.key(SAMPLER_SEED)
                                        phi.sampler = sampler
                                        print("TOO MANY PURITY CALCULATIONS. LET's TRY Re-Initializing the state, with different parameters.")
                                        # print("phi.n_samples =",phi.n_samples)

                                    if purity <= 0:
                                        print("negative purity = ", purity)
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                    elif purity > 1.0000001:
                                        print("unphysical purity grtr than 1.0000001 =", purity)
                                        print("purity estimate is much greater than 1. Repeat with the same variational state and diff. samples")
                                    elif jnp.isnan(purity):
                                        print("purity =", purity)
                                        print("nan value produced for purity. Trying again with a different set of samples.")
                                    samples_jax_1 = phi.sample()  # now you're doing it with more
                                    samples_jax_2 = phi.sample()
                                    
                                    purity, entropy = renyi_2_entropy_jax_chunked(phi._apply_fun, L,
                                                                                  numSpins_A,
                                                                                  nsamples,
                                                                                  chunk_size,
                                                                                  phi.model_state,
                                                                                  phi.parameters,
                                                                                  inputs1=samples_jax_1,
                                                                                  inputs2=samples_jax_2,
                                                                                  purity_sampling_error=purity_correlation_sampling_errors,
                                                                                  entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        
                                    num_purity_calculations += 1
                                    # This technique helps and works.
                                # once we are out of the while loop, check to see if we have changed the number of samples. If so, change it
                                # back, to prepare for the subsequent calculations.
                                # we need to reset the number of samples to the original nsamples.
                                if phi.n_samples != nsamples:
                                    phi.n_samples = nsamples  # add 5000 samples
                                    phi.reset()
                                    #print("We are out of the while loop. number of samples changed back to ", phi.n_samples)

                    else: # unchunked, but still Jax
                        # Case purity_correlation_sampling_errors = 1, 
                        # purityLogSumExp_correlation_sampling_errors = 0
                        if purity_correlation_sampling_errors:
                            purity, purity_error, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                                phi.parameters, inputs1=samples_jax_1,
                                                                                inputs2=samples_jax_2,
                                                                                purity_sampling_error=purity_correlation_sampling_errors,
                                                                                entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if (purity <= 0 or purity > 1.0000001) or jnp.isnan(purity):
                                while (purity <= 0 or purity > 1.0000001 or jnp.isnan(purity)):
                                    if num_purity_calculations >= 2:


                                        # OK nothing has worked so far. So try re-initializing the state at the exact same parameters.
                                        # Next step will be to re-initialize to different parameters.
                                        SEED += 3123149  # Change the seed used to initialize the param
                                        # += 1 means you have the wavefunction from the next realization
                                        # SEED = int(L * dhidden * width * 100000 + i)

                                        # ROELAND: Call the init function with the seed:
                                        phi.init(random.key(SEED))  # This initializes the parameters

                                        # Hack to reset the sampler seed, which Jannes taught me.
                                        SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + i)
                                        # SAMPLER_SEED = int(0)
                                        phi._sampler_seed = random.key(SAMPLER_SEED)
                                        phi.sampler = sampler
                                        print("TOO MANY PURITY CALCULATIONS. LET's TRY Re-Initializing the state, with different parameters.")
                                        # print("phi.n_samples =",phi.n_samples)

                                    if purity <= 0:
                                        print("negative purity = ", purity)
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                    elif purity > 1.0000001:
                                        print("unphysical purity grtr than 1.0000001 =", purity)
                                        print("purity estimate is much greater than 1. Repeat with the same variational state and diff. samples")
                                    elif jnp.isnan(purity):
                                        print("purity =", purity)
                                        print("nan value produced for purity. Trying again with a different set of samples.")
                                    samples_jax_1 = phi.sample()  # now you're doing it with more
                                    samples_jax_2 = phi.sample()
                                    
                                    purity, purity_error, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                                        phi.parameters, inputs1=samples_jax_1,
                                                                                        inputs2=samples_jax_2,
                                                                                        purity_sampling_error=purity_correlation_sampling_errors,
                                                                                        entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        
                                    num_purity_calculations += 1
                                    # This technique helps and works.
                                # once we are out of the while loop, check to see if we have changed the number of samples. If so, change it
                                # back, to prepare for the subsequent calculations.
                                # we need to reset the number of samples to the original nsamples.
                                if phi.n_samples != nsamples:
                                    phi.n_samples = nsamples  # add 5000 samples
                                    phi.reset()
                                    #print("We are out of the while loop. number of samples changed back to ", phi.n_samples)
                            
                            
                                    
                        elif purityLogSumExp_correlation_sampling_errors:
                            # Case purity_correlation_sampling_errors = 0, 
                            # purityLogSumExp_correlation_sampling_errors = 1
                            entropy_logSumExp = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                    phi.parameters, inputs1=samples_jax_1,
                                                                    inputs2=samples_jax_2,
                                                                    purity_sampling_error=purity_correlation_sampling_errors,
                                                                    entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                        else:
                            # Case purity_correlation_sampling_errors = 0, 
                            # purityLogSumExp_correlation_sampling_errors = 0
                            purity, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                  phi.parameters, inputs1=samples_jax_1,
                                                                  inputs2=samples_jax_2,
                                                                  purity_sampling_error=purity_correlation_sampling_errors,
                                                                  entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                            num_purity_calculations = 1
                            if (purity <= 0 or purity > 1.0000001) or jnp.isnan(purity):
                                while (purity <= 0 or purity > 1.0000001 or jnp.isnan(purity)):
                                    if num_purity_calculations >= 2:


                                        # OK nothing has worked so far. So try re-initializing the state at the exact same parameters.
                                        # Next step will be to re-initialize to different parameters.
                                        SEED += 3123149  # Change the seed used to initialize the param
                                        # += 1 means you have the wavefunction from the next realization
                                        # SEED = int(L * dhidden * width * 100000 + i)

                                        # ROELAND: Call the init function with the seed:
                                        phi.init(random.key(SEED))  # This initializes the parameters

                                        # Hack to reset the sampler seed, which Jannes taught me.
                                        SAMPLER_SEED = int(L * dhidden * width ** 2 * 32 * 400.98 + i)
                                        # SAMPLER_SEED = int(0)
                                        phi._sampler_seed = random.key(SAMPLER_SEED)
                                        phi.sampler = sampler
                                        print("TOO MANY PURITY CALCULATIONS. LET's TRY Re-Initializing the state, with different parameters.")
                                        # print("phi.n_samples =",phi.n_samples)

                                    if purity <= 0:
                                        print("negative purity = ", purity)
                                        print("purity estimate is negative. Trying again with a different set of samples.")
                                    elif purity > 1.0000001:
                                        print("unphysical purity grtr than 1.0000001 =", purity)
                                        print("purity estimate is much greater than 1. Repeat with the same variational state and diff. samples")
                                    elif jnp.isnan(purity):
                                        print("purity =", purity)
                                        print("nan value produced for purity. Trying again with a different set of samples.")
                                    samples_jax_1 = phi.sample()  # now you're doing it with more
                                    samples_jax_2 = phi.sample()
                                    
                                    purity, entropy = renyi_2_entropy_jax(phi._apply_fun, L, numSpins_A, phi.model_state,
                                                                          phi.parameters, inputs1=samples_jax_1,
                                                                          inputs2=samples_jax_2,
                                                                          purity_sampling_error=purity_correlation_sampling_errors,
                                                                          entropyLogSumExp=purityLogSumExp_correlation_sampling_errors)
                                        
                                    num_purity_calculations += 1
                                    # This technique helps and works.
                                # once we are out of the while loop, check to see if we have changed the number of samples. If so, change it
                                # back, to prepare for the subsequent calculations.
                                # we need to reset the number of samples to the original nsamples.
                                if phi.n_samples != nsamples:
                                    phi.n_samples = nsamples  # add 5000 samples
                                    phi.reset()
                                    #print("We are out of the while loop. number of samples changed back to ", phi.n_samples)




                '''
                DEBUGGING, why I'm getting huge entropy values for L = 56, dhidden = 40, width = 5.1. The issue happens severely at
                numsamples = 2*10^4, numsamples = 2*10^5 but also numsamples = 2000, or 200. It happens at the 56th realization using
                the latest SEED scheme, for all these values of numsamples. Final purity associated with subsystem half the size of the
                full system, for numsamples = 200, is average purity = 15527101.38579892, average entropy = -16.55809753158705
                
                Crazy right? I think whenever you get a pair of samples that produce a humungus value for the swap operator, it means you need
                infinite samples to get an accurate, physical read of the purity.
                '''
            
            elif (dhidden == 0) or (width == 0.0):
            #elif width == 0.0:
                print("No need to generate samples. We know the result a priori.")
                # not calculating logSumExp in this case, no need.
                num_purity_calculations = 0
                if exactOrMC == 'exact':
                    purity = 1.0
                    entropy = 0.0

                elif exactOrMC == 'MC':
                    if purity_correlation_sampling_errors:
                        purity = 1.0
                        entropy = 0.0
                        purity_error = 0.0
                    elif purityLogSumExp_correlation_sampling_errors:
                        # purity_error = 0.0
                        entropy_logSumExp = 0.0
                    else:
                        purity = 1.0
                        entropy = 0.0


            print("num_purity_calculations = ", num_purity_calculations)

            if exactOrMC == 'exact':
                print("purity =", purity)
                print("entropy =", entropy)
                purity_array.append(purity)
                entropy_array.append(entropy)

            elif exactOrMC == 'MC':
                if purity_correlation_sampling_errors:
                    print("purity =", purity)
                    print("entropy =", entropy)
                    print("purity_std_sampling =", purity_error)
                    purity_array.append(purity)
                    entropy_array.append(entropy)
                    purity_RNN_sampling_error_array.append(purity_error)
                elif purityLogSumExp_correlation_sampling_errors:
                    entropy_logSumExp_array.append(entropy_logSumExp)
                    print("entropy_logSumExp =", entropy_logSumExp)
                else:
                    print("purity =", purity)
                    print("entropy =", entropy)
                    purity_array.append(purity)
                    entropy_array.append(entropy)
            

            '''
            SAVE AFTER EVERY REALIZATION ITERATION (Checkpointing)
            '''
            
            if exactOrMC == 'exact':
                purity_array = np.array(purity_array)
                np.save(filepath_purity_array_checkpointing, purity_array)

                entropy_array = np.array(entropy_array)  # I think the size of the array is (numrealizations, d)
                np.save(filepath_entropy_array_checkpointing, entropy_array)

            elif exactOrMC == 'MC':
                if purity_correlation_sampling_errors:
                    purity_array = np.array(purity_array)
                    np.save(filepath_purity_array_checkpointing, purity_array)

                    entropy_array = np.array(entropy_array)  # I think the size of the array is (numrealizations, d)
                    np.save(filepath_entropy_array_checkpointing, entropy_array)

                    purity_RNN_sampling_error_array = np.array(purity_RNN_sampling_error_array)
                    np.save(filepath_purity_RNN_sampling_error_array_checkpointing,purity_RNN_sampling_error_array)

                elif purityLogSumExp_correlation_sampling_errors:
                    entropy_logSumExp_array = np.array(entropy_logSumExp_array)
                    np.save(filepath_entropy_logSumExp_array_checkpointing,entropy_logSumExp_array)

                else:
                    purity_array = np.array(purity_array)
                    np.save(filepath_purity_array_checkpointing, purity_array)

                    entropy_array = np.array(entropy_array)  # I think the size of the array is (numrealizations, d)
                    np.save(filepath_entropy_array_checkpointing, entropy_array)

            

        '''
        CALCULATE THE rank of the QGT IF AND ONLY IF calcRankQGT == True AND THE RELEVANT FILE DOESN'T ALREADY EXIST
        '''
        if calcRankQGT and (not os.path.isfile(filepath_rankFullness_qgt_array)):
            # "not ___" essentially performs the checkpointing for us

            # Calculate the rank of the QGT
            # print("model =",model_name)
            # qgt = nk.optimizer.qgt.QGTJacobianPyTree(vstate=phi,holomorphic=(model == 'rbm'))
            # qgt = nk.optimizer.qgt.QGTJacobianPyTree(vstate=phi,holomorphic=(model == 'rnn'))
            # qgt = nk.optimizer.qgt.QGTJacobianPyTree(vstate=phi)
            qgt = nk.optimizer.qgt.QGTJacobianDense(vstate=phi)
            qgt_mat = qgt.to_dense()
            # rank_qgt = jax.numpy.linalg.matrix_rank(qgt_mat)
            '''
            We want the rank fullness, not the rank, because we will be changing the number of parameters by modifying
            dhidden.
            '''
            if tol_rank_calc == 0.0:
                # If we set tol_rank_calc to 0, then we want jax to use its algorithm to decide the tolerance for
                # every simulation, and we do not explicitly specify a tolerance in the matrix_rank function.
                rankFullness_qgt = jnp.linalg.matrix_rank(qgt_mat) / phi.n_parameters
            else:
                rankFullness_qgt = jnp.linalg.matrix_rank(qgt_mat, tol=tol_rank_calc) / phi.n_parameters
            # rankFullness_qgt = jnp.linalg.matrix_rank(qgt_mat) / phi.n_parameters
            print("rankFullness_qgt =", rankFullness_qgt)
            rankFullness_qgt_array.append(rankFullness_qgt)


        # Now it's time to calculate the entanglement spectrum for each realization, and store the data in an array
        if calcEntangSpectrum and (not os.path.isfile(filepath_eigenvalues_rho_A_array)):
            # ccheck if rho_A exists. If it already does no need to calculate it. Also if it does, then we know configs_full
            # also exists. If rho_A doesn't exist, then calculate it here.
            eigenvalues_rho_A = list(eigenvalues_rho_A)  # we converted this array to an np.array at the end
            # of the previous realization
            if 'rho_A' in locals():
                eigenvalues, eigenvectors = jax.scipy.linalg.eigh(rho_A)

            else:
                # then we have not yet calculated rho_A, and
                shape = tuple(np.full(L, 2))
                shape = tuple(np.append(1, shape))
                # print("shape =",shape)
                log_psi = model.apply({'params': parameters}, configs_full)
                psi = jnp.exp(log_psi)

                if ARD_vs_MCMC == 'MCMC':
                    norm_sq = jnp.sum(psi * jnp.conj(psi))
                    norm = jnp.sqrt(norm_sq)
                    psi = psi / norm

                psi = np.reshape(psi, shape)

                # print(psi,psi.shape)
                keep = list(np.arange(0, numSpins_A))
                dims = list(np.full(L, 2))
                dims = [1] + dims
                # print("dims =",dims)
                rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]
                print("jnp.trace(rho_A) =", jnp.trace(rho_A))
                eigenvalues, eigenvectors = jax.scipy.linalg.eigh(rho_A)
                print("np.max(eigenvalues) =", np.max(eigenvalues))

            eigenvalues_rho_A = eigenvalues_rho_A + list(eigenvalues)
            print("eigenvalues_rho_A calculated successfully")

            eigenvalues_rho_A = np.array(eigenvalues_rho_A)
            np.save(filepath_eigenvalues_rho_A_array_checkpointing, eigenvalues_rho_A)

        if calcAdjacentGapRatio and (not os.path.isfile(filepath_adjacent_energy_gap_ratio_array)):
            # First check if we've calculated the density matrix or not and if so, have we done the SVD
            # Right now what we will do is do linalg.eigh for the Adjacent Gap Ratio
            adjacent_energy_gap_ratio_array = list(adjacent_energy_gap_ratio_array) # we converted this array to an np.array at the end
                                                                                    # of the previous realization
            
            if 'rho_A' in locals():
                eigenvalues, eigenvectors = scipy.linalg.eigh(rho_A)
                # Discarding negative values produced due to numerical precision issues
                print("eigenvalues of rho_A =",eigenvalues)
                s = eigenvalues[eigenvalues > 0]  # I think all the negative values would be listed in the first positions because
                # eigenvalues is a list of eigenvalues of rho_A in ascending order.

            else:  # no rho_A, which means no s
                # then we have not yet calculated rho_A, and
                shape = tuple(np.full(L, 2))
                shape = tuple(np.append(1, shape))
                # print("shape =",shape)
                log_psi = model.apply({'params': parameters}, configs_full)
                psi = jnp.exp(log_psi) # this was always jnp, everything else was np I think

                if ARD_vs_MCMC == 'MCMC':
                    norm_sq = jnp.sum(psi * jnp.conj(psi))
                    norm = jnp.sqrt(norm_sq)
                    psi = psi / norm

                psi = np.reshape(psi, shape)

                # print(psi,psi.shape)
                keep = list(np.arange(0, numSpins_A))
                dims = list(np.full(L, 2))
                dims = [1] + dims
                # print("dims =",dims)
                rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]
                eigenvalues, eigenvectors = scipy.linalg.eigh(rho_A)
                print("eigenvalues of rho_A =",eigenvalues)

                # Discarding negative values produced due to numerical precision issues
                s = eigenvalues[eigenvalues > 0]

            # Now we have s, containing the positive eigenvalues of rho_A. First we want the entanglement spectrum.
            s_entang_H = -np.log(s)
            s_gap = np.abs(np.diff(s_entang_H))
            # Calculate ratios and avoid numerical issues by taking the exponential of the logarithmic difference
            s_gap_ratios = np.exp(np.diff(np.log(s_gap)))

            s_gap_ratios_reciprocal = 1 / s_gap_ratios
            s_gap_ratios_final = np.minimum(s_gap_ratios, s_gap_ratios_reciprocal)
            # Now we have our adjacent gap ratios, correctly calculated (inshallah). We want their average.
            adjacent_energy_gap_ratio_array.append(np.mean(s_gap_ratios_final))
            print("average ratio_adjacent_energy_gap (current realization) =", adjacent_energy_gap_ratio_array[-1])
            
            adjacent_energy_gap_ratio_array = np.array(adjacent_energy_gap_ratio_array)
            np.save(filepath_adjacent_energy_gap_ratio_array_checkpointing, adjacent_energy_gap_ratio_array)

        # rho
        if calcFourierTransform_rho and (not os.path.isfile(filepath_fourier_transform_rho)):
            # Now bring up the checkpointing file and see where we are: see if it exists first
            if not os.path.isfile(filepath_fourier_transform_rho_checkpointing):
                # No calculations have been done yet
                print("starting Fourier Transform calculationsfor rho at realization = 0")
                rho = full_density_matrix(model, configs_full, parameters)
                sum_abs_Op_realiz_array = []
                for j in range(max_kBody_term):
                    k = j + 1
                    print("k =", k, " body terms")
                    sum_abs_Op, sum_abs_Op_squared = FT_k_terms(rho=rho, k=k, N=L)
                    sum_abs_Op_realiz_array.append(sum_abs_Op)
                    sum_abs_Op_realiz_array.append(sum_abs_Op_squared)
                    print("sum_abs_Op_realiz_array =", sum_abs_Op_realiz_array)

                sum_abs_Op_realiz_array = np.array(sum_abs_Op_realiz_array)
                sum_abs_Op_rho_array.append(sum_abs_Op_realiz_array)

                '''
                SAVE AFTER EVERY REALIZATION ITERATION
                '''
                sum_abs_Op_rho_array = np.array(sum_abs_Op_rho_array)
                # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                np.save(filepath_fourier_transform_rho_checkpointing, sum_abs_Op_rho_array)
                # print("sum_abs_Op_squared_rho_array =",sum_abs_Op_squared_rho_array)

            elif os.path.isfile(filepath_fourier_transform_rho_checkpointing):
                # The checkpointing file exists. Only start calculating when we reach the correct realization
                sum_abs_Op_rho_array = np.load(filepath_fourier_transform_rho_checkpointing)
                if i + 1 > sum_abs_Op_rho_array.shape[0]:  # i is the index associated with the present realization
                    print("continuing Fourier Transform calculations for rho at realization =", i)
                    sum_abs_Op_rho_array = list(sum_abs_Op_rho_array)  # code to append to lists is nicer

                    rho = full_density_matrix(model, configs_full, parameters)
                    sum_abs_Op_realiz_array = []
                    for j in range(max_kBody_term):
                        k = j + 1
                        print("k =", k, " body terms")
                        sum_abs_Op, sum_abs_Op_squared = FT_k_terms(rho=rho, k=k, N=L)
                        sum_abs_Op_realiz_array.append(sum_abs_Op)
                        sum_abs_Op_realiz_array.append(sum_abs_Op_squared)
                        print("sum_abs_Op_realiz_array =", sum_abs_Op_realiz_array)

                    sum_abs_Op_realiz_array = np.array(sum_abs_Op_realiz_array)
                    sum_abs_Op_rho_array.append(sum_abs_Op_realiz_array)
                    '''
                    SAVE AFTER EVERY REALIZATION ITERATION
                    '''
                    sum_abs_Op_rho_array = np.array(sum_abs_Op_rho_array)
                    # print("sum_abs_Op_squared_array =",sum_abs_Op_squared_array)
                    # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                    np.save(filepath_fourier_transform_rho_checkpointing, sum_abs_Op_rho_array)

        # rhoA
        if calcFourierTransform_rhoA and (not os.path.isfile(filepath_fourier_transform_rhoA)):
            # print("JE SUIS CHARLIE")
            # Now bring up the checkpointing file and see where we are: see if it exists first
            if not os.path.isfile(filepath_fourier_transform_rhoA_checkpointing):
                # No calculations have been done yet
                print("starting Fourier Transform calculations for rho_A at realization = 0")

                '''
                IF DOING THIS CALCULATION FOR RHO_A.... ACTUALLY NEED NEW CODE FOR THIS.
                '''
                if not 'rho_A' in locals():
                    # then we have not yet calculated rho_A
                    shape = tuple(np.full(L, 2))
                    shape = tuple(np.append(1, shape))
                    # print("shape =",shape)
                    log_psi = model.apply({'params': parameters}, configs_full)
                    psi = jnp.exp(log_psi)
                    if ARD_vs_MCMC == 'MCMC':
                        norm_sq = jnp.sum(psi * jnp.conj(psi))
                        norm = jnp.sqrt(norm_sq)
                        psi = psi / norm
                    psi = np.reshape(psi, shape)

                    # print(psi,psi.shape)
                    keep = list(np.arange(0, numSpins_A))
                    dims = list(np.full(L, 2))
                    dims = [1] + dims
                    # print("dims =",dims)
                    rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]

                sum_abs_Op_realiz_array = []
                for j in range(max_kBody_term):
                    k = j + 1
                    print("k =", k, " body terms")
                    sum_abs_Op, sum_abs_Op_squared = FT_k_terms(rho=rho_A, k=k, N=numSpins_A)
                    sum_abs_Op_realiz_array.append(sum_abs_Op)
                    sum_abs_Op_realiz_array.append(sum_abs_Op_squared)
                    print("sum_abs_Op_realiz_array =", sum_abs_Op_realiz_array)

                sum_abs_Op_realiz_array = np.array(sum_abs_Op_realiz_array)
                sum_abs_Op_rhoA_array.append(sum_abs_Op_realiz_array)

                '''
                SAVE AFTER EVERY REALIZATION ITERATION
                '''
                sum_abs_Op_rhoA_array = np.array(sum_abs_Op_rhoA_array)
                # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                np.save(filepath_fourier_transform_rhoA_checkpointing, sum_abs_Op_rhoA_array)
                # print("sum_abs_Op_squared_rhoA_array =",sum_abs_Op_squared_rhoA_array)


            elif os.path.isfile(filepath_fourier_transform_rhoA_checkpointing):
                # The checkpointing file exists. Only start calculating when we reach the correct realization
                sum_abs_Op_rhoA_array = np.load(filepath_fourier_transform_rhoA_checkpointing)
                if i + 1 > sum_abs_Op_rhoA_array.shape[0]:
                    print("continuing Fourier Transform calculations for rho_A at realization =", i)
                    sum_abs_Op_rhoA_array = list(sum_abs_Op_rhoA_array)  # code to append to lists is nicer

                    if not 'rho_A' in locals():
                        # then we have not yet calculated rho_A
                        shape = tuple(np.full(L, 2))
                        shape = tuple(np.append(1, shape))
                        # print("shape =",shape)
                        log_psi = model.apply({'params': parameters}, configs_full)
                        psi = jnp.exp(log_psi)
                        if ARD_vs_MCMC == 'MCMC':
                            norm_sq = jnp.sum(psi * jnp.conj(psi))
                            norm = jnp.sqrt(norm_sq)
                            psi = psi / norm
                        psi = np.reshape(psi, shape)

                        # print(psi,psi.shape)
                        keep = list(np.arange(0, numSpins_A))
                        dims = list(np.full(L, 2))
                        dims = [1] + dims
                        # print("dims =",dims)
                        rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]

                    sum_abs_Op_realiz_array = []
                    for j in range(max_kBody_term):
                        k = j + 1
                        print("k =", k, " body terms")
                        sum_abs_Op, sum_abs_Op_squared = FT_k_terms(rho=rho_A, k=k, N=numSpins_A)
                        sum_abs_Op_realiz_array.append(sum_abs_Op)
                        sum_abs_Op_realiz_array.append(sum_abs_Op_squared)
                        print("sum_abs_Op_realiz_array =", sum_abs_Op_realiz_array)

                    sum_abs_Op_realiz_array = np.array(sum_abs_Op_realiz_array)
                    sum_abs_Op_rhoA_array.append(sum_abs_Op_realiz_array)
                    '''
                    SAVE AFTER EVERY REALIZATION ITERATION
                    '''
                    sum_abs_Op_rhoA_array = np.array(sum_abs_Op_rhoA_array)
                    # print("sum_abs_Op_squared_array =",sum_abs_Op_squared_array)
                    # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                    np.save(filepath_fourier_transform_rhoA_checkpointing, sum_abs_Op_rhoA_array)
        
        
        
        
        
        
        
        # Correlation function up to distance d
        if calcCorr_1d_upTo_dist: # and (not os.path.isfile(filepath_corr_up_to_d)):
            
            if purity_correlation_sampling_errors:
                
                
                #phi.reset()
                # First calculate the average spin at all sites, with corresponding errors
                m_avg_array, m_error_array, m_error_of_mean_array = sigma_z_all_sites(L, hilbert, phi, nsamples, newSamplesOrNot_corr)
                # Change type to list to append cleanly.
                magnetization_avg_array = list(magnetization_avg_array)
                magnetization_avg_array.append(m_avg_array)
                magnetization_error_of_mean_array = list(magnetization_error_of_mean_array)
                magnetization_error_of_mean_array.append(m_error_of_mean_array)

                corr_connected_avg_over_sites_array = [] # here we will output just for printing purposes the average across all sites
                                                         # of the full connected correlation function for the current wavefunction
                #phi.reset()
                for d in range(1,d_up_to_corr+1):
                    print("d (for this realiz) =", d)
                    #phi.reset()
                    avg_array, error_array, error_of_mean_array = unconnected_corr_avg_error_ij_d(L, hilbert, d, phi, nsamples, newSamplesOrNot_corr)
                    #print("phi._samples AFTER CORR for d calc complete =", phi._samples)
                    if i == 0: # i.e. we are at the first realization, so the dictionary is still empty
                        corr_unconnected_avg_up_to_d_dict_of_arrays['d=%s'%d] = [avg_array] # avg_array is a list, so this is a list of lists
                        corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=%s'%d] = [error_of_mean_array] # store error of mean, not error --> easier
                    else: # i > 0
                        corr_unconnected_avg_up_to_d_dict_of_arrays['d=%s'%d].append(avg_array)
                        #print('corr KEYS =',corr_unconnected_avg_up_to_d_dict_of_arrays.keys())
                        corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=%s'%d].append(error_of_mean_array)
                    
                    assert(len(avg_array) == L-d)
                    assert(len(error_array) == L-d)
                    assert(len(error_of_mean_array) == L-d)
                    
                    # Ok, so now we have the unnconnected correlation function at all (m,n) pairs of sites separated by
                    # a distance d for the current wavefunction. It's L-d values. Let's substract the correct
                    # sigma_m * sigma_n values from each pair, and calculate the average value of the absolute value
                    # of the connected correlation function across all sites separated by a distance d for this
                    # specific wavefunction
                    
                    
                    # Generate set of tuples corresponding to each pair
                    list_of_edges = []
                    for j in range(L - d):
                        list_of_edges.append((j, j + d))
                    assert(len(list_of_edges) == L-d)
                    
                    count = 0
                    connected_corr_avg_over_all_m_n_pairs = 0
                    for (m,n) in list_of_edges:
                        connected_corr_avg_over_all_m_n_pairs += abs(avg_array[count] - m_avg_array[m] * m_avg_array[n])
                        count += 1
                        
                    connected_corr_avg_over_all_m_n_pairs /= len(list_of_edges)
                    corr_connected_avg_over_sites_array.append(connected_corr_avg_over_all_m_n_pairs)
                    #print("corr_connected_avg_over_sites_array =", corr_connected_avg_over_sites_array)
                
                print("corr_connected_avg_over_sites_array =", corr_connected_avg_over_sites_array)
                
                with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, 'wb') as handle:
                    pickle.dump(corr_unconnected_avg_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing, 'wb') as handle:
                    pickle.dump(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                #np.save(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing,corr_unconnected_avg_up_to_d_dict_of_arrays)
                #np.save(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing,corr_unconnected_error_of_mean_up_to_d_dict_of_arrays)
                np.save(filepath_sigmai_RNN_sampling_avg_array_checkpointing,magnetization_avg_array)
                np.save(filepath_sigmai_RNN_error_of_mean_array_checkpointing,magnetization_error_of_mean_array)
                
                        
            else:
                # Technically we know that filepath_corr_up_to_d doesn't exist because of the whole indexWhereToStart analysis and the fact we are
                # in the loop. But we can still do the if statement here...

                # Ok so the full calculation of the correlation function up to distance d for this realization is not finished. Let's then
                # calculate the expection value of sigma_z at every site, because we will need this for the correlation function calculation at any d.
                sigma_z_avg_array, sigma_z_err_array, sigma_z_errOfmean_array = sigma_z_all_sites(L, hilbert, phi, nsamples, newSamplesOrNot_corr)

                # ensure corr_abs_avg_up_to_d_array is a list. This caused errors in the simulations.
                corr_abs_avg_up_to_d_array = list(corr_abs_avg_up_to_d_array)

                # corr_abs_avg_up_to_d_array has already been loaded assuming the checkpointing file exists. So there is no need to check again
                # if the checkpointing file exists or not. We have already pre-selected indexWhereToStart to know exactly where to start the
                # loop. If indexWhereToStart == 0, then corr_abs_avg_up_to_d_array = [] has already been defined.
                corr_abs_avg_up_to_d_realiz_array = []  # store the correlation function for this realization
                for d in range(1, d_up_to_corr + 1):
                    print("d (for this realiz) =", d)
                    corr_abs_avg_up_to_d_realiz_array.append(connected_corr_avg_all_sites(L, hilbert, d, phi, nsamples, sigma_z_avg_array, newSamplesOrNot_corr))
                    print("corr_abs_avg_up_to_d_realiz_array =", corr_abs_avg_up_to_d_realiz_array)

                corr_abs_avg_up_to_d_realiz_array = np.array(corr_abs_avg_up_to_d_realiz_array)
                corr_abs_avg_up_to_d_array.append(corr_abs_avg_up_to_d_realiz_array)

                '''
                SAVE AFTER EVERY REALIZATION ITERATION
                '''
                corr_abs_avg_up_to_d_array = np.array(corr_abs_avg_up_to_d_array)  # I think the size of the array is (numrealizations, d)
                # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                np.save(filepath_corr_up_to_d_checkpointing, corr_abs_avg_up_to_d_array)
                # print("sum_abs_Op_squared_rho_array =",sum_abs_Op_squared_rho_array)
            


        # Correlation function AT A SPECIFIC distance d
        if calcCorr_1d_specific_dist and (not os.path.isfile(filepath_corr_d_specific)):
            # Ok so the full calculation of the correlation function AT distance d for this realization is not finished. Let's then
            # calculate the expection value of sigma_z at every site, because we will need this for the correlation function calculation at any d.
            sigma_z_avg_array, sigma_z_err_array, sigma_z_errOfmean_array = sigma_z_all_sites(L, hilbert, phi, nsamples, newSamplesOrNot_corr)
            # Now bring up the checkpointing file and see where we are: see if it exists first
            if not os.path.isfile(filepath_corr_d_specific_checkpointing):
                # No calculations have been done yet
                print("starting correlation function calculation for SPECIFIC d at realization = 0")
                corr_avg_specific_d_array.append(connected_corr_avg_all_sites(L, hilbert, d_specific_corr, phi, nsamples, sigma_z_avg_array, newSamplesOrNot_corr))

                '''
                SAVE AFTER EVERY REALIZATION ITERATION
                '''
                corr_avg_specific_d_array = np.array(corr_avg_specific_d_array)  # I think the size of the array is (numrealizations, d)
                # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                np.save(filepath_corr_d_specific_checkpointing, corr_avg_specific_d_array)
                # print("sum_abs_Op_squared_rho_array =",sum_abs_Op_squared_rho_array)

            elif os.path.isfile(filepath_corr_d_specific_checkpointing):
                # The checkpointing file exists. Only start calculating when we reach the correct realization
                corr_avg_specific_d_array = np.load(filepath_corr_d_specific_checkpointing)
                if i + 1 > corr_avg_specific_d_array.shape[0]:  # i is the index associated with the present realization
                    print("continuing correlation function calculation for SPECIFIC d at realization =", i)
                    corr_avg_specific_d_array = list(corr_avg_specific_d_array)  # code to append to lists is nicer
                    corr_avg_specific_d_array.append(connected_corr_avg_all_sites(L, hilbert, d_specific_corr, phi, nsamples, sigma_z_avg_array, newSamplesOrNot_corr))

                    '''
                    SAVE AFTER EVERY REALIZATION ITERATION
                    '''
                    corr_avg_specific_d_array = np.array(corr_avg_specific_d_array)  # I think the size of the array is (numrealizations, d)
                    # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
                    np.save(filepath_corr_d_specific_checkpointing, corr_avg_specific_d_array)
                    # print("sum_abs_Op_squared_rho_array =",sum_abs_Op_squared_rho_array)

        # ENTROPY SCALING
        if calcEntropy_scaling and (not os.path.isfile(filepath_entropy_scaling_law)):
            # Technically we know that filepath_corr_up_to_d doesn't exist because of the whole indexWhereToStart analysis and the fact we are
            # in the loop. But we can still do the if statement here...

            # Ok so the full calculation of the entropy (changing numSpins_A) for this realization is not finished

            # ensure corr_abs_avg_up_to_d_array is a list. This caused errors in the simulations.
            entropy_scaling_array = list(entropy_scaling_array)

            # entropy_scaling_array has already been loaded assuming the checkpointing file exists. So there is no need to check again
            # if the checkpointing file exists or not. We have already pre-selected indexWhereToStart to know exactly where to start the
            # loop. If indexWhereToStart == 0, then entropy_scaling_array = [] has already been defined.
            entropy_scaling_realiz_array = []  # store the entropy at varying numSpins_A function for this realization

            # Calculate full pure state density matrix if exact calculation is being done, and generate samples for the swap operator
            # if importance sampling (MC) is used. Do this outside of the loop over L_A to only have to do it one time per realization.
            if exactOrMC == 'exact':
                # if we are here, then embark on the entropy calculations.
                # print("configs =",configs,type(configs))
                shape = tuple(np.full(L, 2))
                shape = tuple(np.append(1, shape))
                # print("shape =",shape)
                log_psi = model.apply({'params': parameters}, configs_full)
                psi = jnp.exp(log_psi)

                if ARD_vs_MCMC == 'MCMC':
                    norm_sq = jnp.sum(psi * jnp.conj(psi))
                    norm = jnp.sqrt(norm_sq)
                    psi = psi / norm

                psi = np.reshape(psi, shape)
            elif exactOrMC == 'MC':
                samples1 = np.reshape(phi.sample(), (new_nsamples, L))
                samples2 = np.reshape(phi.sample(), (new_nsamples, L))

            for L_A in range(1, L):  # numSpins_A (i.e. L_A) = 1,2,...,L-1
                if exactOrMC == 'exact':
                    # print(psi,psi.shape)
                    keep = list(np.arange(0, L_A))
                    dims = list(np.full(L, 2))
                    dims = [1] + dims
                    # print("dims =",dims)
                    rho_A = partial_trace_np(psi=psi, keep=keep, dims=dims)[0]
                    # print(rho_A)
                    # print("psi_A =",psi_A,psi_A.shape)
                    rho_A_sq = jnp.einsum('ij,jk->ik', rho_A, rho_A)
                    purity = jnp.trace(rho_A_sq)

                    purity = purity.real
                    entropy = -jnp.log(purity)
                elif exactOrMC == 'MC':
                    # use the same pair of samples sets
                    if purity_correlation_sampling_errors:
                        purity, purity_error, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                                        parameters=parameters,
                                                                        numSpins_A=L_A, seed=SEED, exactOrMC=exactOrMC,
                                                                        purity_sampling_error=purity_correlation_sampling_errors)
                    else:
                        purity, entropy = renyi_2_entropy(model=model, inputs1=samples1, inputs2=samples2,
                                                          parameters=parameters,
                                                          numSpins_A=L_A, seed=SEED, exactOrMC=exactOrMC,
                                                          purity_sampling_error=purity_correlation_sampling_errors)
                        

                # Now we have the entanglement entropy associated with a subsystem of size L_A
                entropy_scaling_realiz_array.append(entropy)
                if L_A == L - 1:
                    print("L_A (for this realiz) =", L_A)
                    print("entropy_scaling_realiz_array =", entropy_scaling_realiz_array)

            entropy_scaling_realiz_array = np.array(entropy_scaling_realiz_array)
            entropy_scaling_array.append(entropy_scaling_realiz_array)

            '''
            SAVE AFTER EVERY REALIZATION ITERATION (Checkpointing)
            '''
            entropy_scaling_array = np.array(entropy_scaling_array)  # I think the size of the array is (numrealizations, d)
            # print("avg sum_abs_Op_squared across realizations =",np.mean(sum_abs_Op_squared_array,axis=0))
            np.save(filepath_entropy_scaling_law_checkpointing, entropy_scaling_array)
            # print("sum_abs_Op_squared_rho_array =",sum_abs_Op_squared_rho_array)


        if 'rho_A' in locals():
            del rho_A  # delete local variable because we use it in "if 'rho_A' in locals()" if statements, and we want a new
            # rho_A for every realization

    if calcEntropy: # this means the final files exist for the desired simulation
        if exactOrMC == 'exact':
            if os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
                # print('asdasdasdaadsdasd')
                purity_array = np.load(filepath_purity_array_checkpointing)
                entropy_array = np.load(filepath_entropy_array_checkpointing)

            if (not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array)):
                np.save(filepath_purity_array, purity_array)
                np.save(filepath_entropy_array, entropy_array)

            print()
            print("Final Statistics")
            print()
            if os.path.isfile(filepath_purity_array) and os.path.isfile(filepath_entropy_array):
                print("average purity =", np.mean(purity_array))
                print("average entropy =", np.mean(entropy_array))
                print()
                purity_standard_error_unpropagated = np.std(purity_array) / np.sqrt(numrealizations)
                entropy_standard_error_unpropagated = np.std(entropy_array) / np.sqrt(numrealizations)
                print("unpropagated standard error on purity across all realiz =", purity_standard_error_unpropagated)
                print("unpropagated standard error on entropy across all realiz =", entropy_standard_error_unpropagated)




        elif exactOrMC == 'MC':
            if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_array_checkpointing) \
            and os.path.isfile(filepath_entropy_array_checkpointing) \
            and os.path.isfile(filepath_purity_RNN_sampling_error_array_checkpointing):
                #print('asdasdasdaadsdasd')
                purity_array = np.load(filepath_purity_array_checkpointing)
                entropy_array = np.load(filepath_entropy_array_checkpointing)
                purity_RNN_sampling_error_array = np.load(filepath_purity_RNN_sampling_error_array_checkpointing)

            elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array_checkpointing):
                #print("asdasd")
                #purity_RNN_sampling_error_array = np.load(filepath_purity_RNN_sampling_error_array_checkpointing)

                entropy_logSumExp_array = np.load(filepath_entropy_logSumExp_array_checkpointing)
                #print("entropy_logSumExp_array =",entropy_logSumExp_array)
                print("entropy_logSumExp_array.shape",entropy_logSumExp_array.shape)
                print("type(entropy_logSumExp_array)",type(entropy_logSumExp_array))
            else:
                purity_array = np.load(filepath_purity_array_checkpointing)
                entropy_array = np.load(filepath_entropy_array_checkpointing)

                #print(entropy_logSumExp_array)

            if purity_correlation_sampling_errors:
                if (not os.path.isfile(filepath_purity_RNN_sampling_error_array)) \
                or (not os.path.isfile(filepath_purity_array)) or (not os.path.isfile(filepath_entropy_array)):
                    np.save(filepath_purity_array, purity_array)
                    np.save(filepath_entropy_array, entropy_array)
                    np.save(filepath_purity_RNN_sampling_error_array, purity_RNN_sampling_error_array)
            elif purityLogSumExp_correlation_sampling_errors:
                if (not os.path.isfile(filepath_entropy_logSumExp_array)):
                #np.save(filepath_purity_RNN_sampling_error_array, purity_RNN_sampling_error_array)
                    np.save(filepath_entropy_logSumExp_array, entropy_logSumExp_array)
            else:
                if (not os.path.isfile(filepath_entropy_array)) or (not os.path.isfile(filepath_purity_array)):
                    np.save(filepath_purity_array, purity_array)
                    np.save(filepath_entropy_array, entropy_array)



            print()
            print("Final Statistics")
            print()
            if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array) \
            and os.path.isfile(filepath_purity_array) and os.path.isfile(filepath_entropy_array):
                print("average purity =", np.mean(purity_array))
                print("average entropy =", np.mean(entropy_array))
                print()
            elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array_checkpointing):
                # We could also do the mean first, absolute value second
                print("average entropy LogSumExp abs 1st mean 2nd =", np.mean(np.abs(entropy_logSumExp_array)))
                print("average entropy LogSumExp mean 1st abs 2nd =", np.abs(np.mean(entropy_logSumExp_array)))
                print("average entropy LogSumExp mean 1st real 2nd",(np.mean(entropy_logSumExp_array)).real)
                print("average entropy LogSumExp real 1st mean 2nd",np.mean(entropy_logSumExp_array.real))
                print()
            elif os.path.isfile(filepath_purity_array_checkpointing) and os.path.isfile(filepath_entropy_array_checkpointing):
                print("average purity =", np.mean(purity_array))
                print("average entropy =", np.mean(entropy_array))
                print()


            #if purity_correlation_sampling_errors and (not os.path.isfile(filepath_purity_RNN_sampling_error_array)):
            if purity_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array) \
            and os.path.isfile(filepath_purity_array) and os.path.isfile(filepath_entropy_array):
                # Calculating standard error of the mean for entropy: see my notebook and error prop. page on Wikipedia for more
                # info
                if (dhidden == 0) or (width == 0.0):
                #if width == 0.0:
                    entropy_standard_error = 0
                    purity_standard_error = 0
                else:
                    entropy_standard_error = 2 * np.log(purity_RNN_sampling_error_array) - np.log(nsamples) - 2 * np.log(purity_array)
                    entropy_standard_error = np.sum(np.exp(entropy_standard_error))
                    entropy_standard_error = np.sqrt(entropy_standard_error) / numrealizations

                    purity_standard_error = 2 * np.log(purity_RNN_sampling_error_array) - np.log(nsamples)
                    purity_standard_error = np.sum(np.exp(purity_standard_error))
                    purity_standard_error = np.sqrt(purity_standard_error) / numrealizations

                print("fully propagated standard error on purity across all realiz =",purity_standard_error)
                print("fully propagated standard error on entropy across all realiz =",entropy_standard_error)
                purity_standard_error_unpropagated = np.std(purity_array) / np.sqrt(numrealizations)
                entropy_standard_error_unpropagated = np.std(entropy_array) / np.sqrt(numrealizations)
                print()
                print("unpropagated standard error on purity across all realiz =",purity_standard_error_unpropagated)
                print("unpropagated standard error on entropy across all realiz =",entropy_standard_error_unpropagated)

            #elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_purity_RNN_sampling_error_array) and os.path.isfile(filepath_entropy_logSumExp_array):
            elif purityLogSumExp_correlation_sampling_errors and os.path.isfile(filepath_entropy_logSumExp_array):

                print("logSumExp standard error on entropy across all realiz, abs 1st, std 2nd =",np.std(np.abs(entropy_logSumExp_array)) / np.sqrt(numrealizations))
                print("logSumExp standard error on entropy across all realiz, std only =",np.std(entropy_logSumExp_array) / np.sqrt(numrealizations))
                print("logSumExp standard error on entropy across all realiz, real part only =",np.std(entropy_logSumExp_array.real) / np.sqrt(numrealizations))
                print()
    
            
        # quick calculation of error across realizations only just for comparison

    if calcRankQGT and (not os.path.isfile(filepath_rankFullness_qgt_array)):
        rankFullness_qgt_array = np.array(rankFullness_qgt_array)
        print("average rankFullness_qgt =", np.mean(rankFullness_qgt_array))
        np.save(filepath_rankFullness_qgt_array, rankFullness_qgt_array)

    if calcEntangSpectrum and (not os.path.isfile(filepath_eigenvalues_rho_A_array)):
        eigenvalues_rho_A = np.load(filepath_eigenvalues_rho_A_array_checkpointing)
        print("len(eigenvalues_rho_A) =",len(eigenvalues_rho_A))
        np.save(filepath_eigenvalues_rho_A_array, eigenvalues_rho_A)
        print("eigenvalues_rho_A for entang spectrum SAVED")

    if calcAdjacentGapRatio and (not os.path.isfile(filepath_adjacent_energy_gap_ratio_array)):
        adjacent_energy_gap_ratio_array = np.load(filepath_adjacent_energy_gap_ratio_array_checkpointing)
        #adjacent_energy_gap_ratio_array = np.array(adjacent_energy_gap_ratio_array)  # contains average value for each realization
        print("avg ratio_adjacent_energy_gap across realizations =", np.mean(adjacent_energy_gap_ratio_array))
        np.save(filepath_adjacent_energy_gap_ratio_array, adjacent_energy_gap_ratio_array)

    if calcFourierTransform_rho and (not os.path.isfile(filepath_fourier_transform_rho)):
        # Suppose we haven't saved the final file containing the mean values across the realization. Then
        # pull up the array from the checkpointing file which contains the data from all realizations, and average
        # over the realizations at each value of k.

        # sum_abs_Op_squared_array = np.array(sum_abs_Op_squared_array)
        sum_abs_Op_rho_array = np.load(filepath_fourier_transform_rho_checkpointing)
        print("avg sum_abs_Op_rho across realizations =", np.mean(sum_abs_Op_rho_array, axis=0))
        np.save(filepath_fourier_transform_rho, sum_abs_Op_rho_array)

    if calcFourierTransform_rhoA and (not os.path.isfile(filepath_fourier_transform_rhoA)):
        # Suppose we haven't saved the final file containing the mean values across the realization. Then
        # pull up the array from the checkpointing file which contains the data from all realizations, and average
        # over the realizations at each value of k.

        # sum_abs_Op_squared_array = np.array(sum_abs_Op_squared_array)
        sum_abs_Op_rhoA_array = np.load(filepath_fourier_transform_rhoA_checkpointing)
        print("avg sum_abs_Op_rhoA across realizations =", np.mean(sum_abs_Op_rhoA_array, axis=0))
        np.save(filepath_fourier_transform_rhoA, sum_abs_Op_rhoA_array)
        
        



    if calcCorr_1d_upTo_dist: # and (not os.path.isfile(filepath_corr_up_to_d)):
        # Suppose we haven't saved the final file containing all the values across the realizations. Then
        # pull up the array from the checkpointing file which contains the data from all realizations, and average
        # over the realizations just for the output, so we get a nice printed set of numbers in the slurm file.

        if purity_correlation_sampling_errors \
        and os.path.isfile(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing) \
        and os.path.isfile(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing) \
        and os.path.isfile(filepath_sigmai_RNN_sampling_avg_array_checkpointing) \
        and os.path.isfile(filepath_sigmai_RNN_error_of_mean_array_checkpointing):
            #print('asdasdasdaadsdasd')
            with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing, 'rb') as handle:
                corr_unconnected_avg_up_to_d_dict_of_arrays = pickle.load(handle)
            
            with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing, 'rb') as handle:
                corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = pickle.load(handle)
            #corr_unconnected_avg_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_sampling_avg_dict_checkpointing,allow_pickle=True).item()
            #corr_unconnected_error_of_mean_up_to_d_dict_of_arrays = np.load(filepath_sigmai_sigmaj_RNN_error_of_mean_dict_checkpointing,allow_pickle=True).item()
            magnetization_avg_array = np.load(filepath_sigmai_RNN_sampling_avg_array_checkpointing)
            magnetization_error_of_mean_array = np.load(filepath_sigmai_RNN_error_of_mean_array_checkpointing)
            
            assert(type(corr_unconnected_avg_up_to_d_dict_of_arrays) == dict)
            assert(type(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays) == dict)
            
            # SAVE IN FINAL FILES
            with open(filepath_sigmai_sigmaj_RNN_sampling_avg_dict, 'wb') as handle:
                pickle.dump(corr_unconnected_avg_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(filepath_sigmai_sigmaj_RNN_error_of_mean_dict, 'wb') as handle:
                pickle.dump(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #np.save(filepath_sigmai_sigmaj_RNN_sampling_avg_dict,corr_unconnected_avg_up_to_d_dict_of_arrays)
            #np.save(filepath_sigmai_sigmaj_RNN_error_of_mean_dict,corr_unconnected_error_of_mean_up_to_d_dict_of_arrays)
            np.save(filepath_sigmai_RNN_sampling_avg_array,magnetization_avg_array)
            np.save(filepath_sigmai_RNN_error_of_mean_array,magnetization_error_of_mean_array)
            
            # Now let's calculate
            
            assert(magnetization_avg_array.shape[0] == numrealizations)
            assert(magnetization_error_of_mean_array.shape[0] == numrealizations)
            assert(len(corr_unconnected_avg_up_to_d_dict_of_arrays['d=1']) == numrealizations)
            assert(len(corr_unconnected_error_of_mean_up_to_d_dict_of_arrays['d=1']) == numrealizations)
            # Now we calculate the average correlation function across all sites for d = 1 through d = d_up_to_corr,
            # and then we calculate the error as well, in order. First, the average correlation function across all
            # sites:
            print()
            print("Final Statistics")
            print()
            
            
            
            #count = 0
            conn_corr_avg_over_all_sites_all_realiz = np.zeros((numrealizations, d_up_to_corr))
            log_conn_corr_avg_over_all_sites_all_realiz = np.zeros((numrealizations, d_up_to_corr))
            for r in range(numrealizations): # loop over wavefunctions
                for d in range(1,d_up_to_corr+1): # loop over values of d
                    # We need the list_of_edges for the loop below: this depends on k
                    list_of_edges_d = []
                    for j in range(L - d):
                        list_of_edges_d.append((j, j + d))
                    #print("d =",d,"; len(list_of_edges) =",len(list_of_edges))
                    assert(len(list_of_edges_d) == L-d)
                
                    #connected_corr_avg_over_all_m_n_pairs = 0
                    conn_r_d_avg_over_sites = 0
                    log_conn_r_d_avg_over_sites = 0
                    for k in range(L-d): # loop over all bonds (site pairs) a distance d apart (there are L-d of them)
                        edge_k = list_of_edges_d[k]
                        m = edge_k[0]
                        n = edge_k[1]
                        # First extract list or (array) corresponding to the unconn corr function at separation d for
                        # wavefunction/realiz r, from which we take the k^th term to get the k^th unconn corr function.
                        unconn_r_d_k = corr_unconnected_avg_up_to_d_dict_of_arrays['d=%s'%d][r][k] # This should be a list
                        # Next we subtract the right product of magnetization
                        conn_r_d_k = abs(unconn_r_d_k - magnetization_avg_array[r,m] * magnetization_avg_array[r,n])
                        conn_r_d_avg_over_sites += conn_r_d_k
                        log_conn_r_d_avg_over_sites += np.log(conn_r_d_k)
                    conn_corr_avg_over_all_sites_all_realiz[r,d-1] += conn_r_d_avg_over_sites / (L-d) 
                    log_conn_corr_avg_over_all_sites_all_realiz[r,d-1] += log_conn_r_d_avg_over_sites / (L-d)
                    
                        
            
            assert(conn_corr_avg_over_all_sites_all_realiz.shape == (numrealizations, d_up_to_corr))
            assert(log_conn_corr_avg_over_all_sites_all_realiz.shape == (numrealizations, d_up_to_corr))
                        
            print("avg corr_abs_avg_up_to_d across realizations =", np.mean(conn_corr_avg_over_all_sites_all_realiz,axis=0))
            print("unpropagated errors for corr across all realiz =",np.std(conn_corr_avg_over_all_sites_all_realiz,axis=0) / np.sqrt(numrealizations))
            
            print("avg log_corr_abs_avg_up_to_d across realizations =", np.mean(log_conn_corr_avg_over_all_sites_all_realiz,axis=0))
            print("unpropagated errors for log corr across all realiz =",np.std(log_conn_corr_avg_over_all_sites_all_realiz,axis=0) / np.sqrt(numrealizations))
            # Last bit is we need code for the errors
            
            

        else:
            if os.path.isfile(filepath_corr_up_to_d_checkpointing):
                corr_abs_avg_up_to_d_array = np.load(filepath_corr_up_to_d_checkpointing)
                np.save(filepath_corr_up_to_d, corr_abs_avg_up_to_d_array)
                print()
                print("Final Statistics")
                print()
                print("avg corr_abs_avg_up_to_d across realizations =", np.mean(corr_abs_avg_up_to_d_array,axis=0))
                print("log of avg corr_abs_avg_up_to_d across realizations =", np.log(np.mean(corr_abs_avg_up_to_d_array,axis=0)))
                print("unpropagated errors for corr across all realiz =",np.std(corr_abs_avg_up_to_d_array,axis=0) / np.sqrt(numrealizations))
            
            

    if calcCorr_1d_specific_dist and (not os.path.isfile(filepath_corr_d_specific)):
        corr_avg_specific_d_array = np.array(corr_avg_specific_d_array)  # contains average value for each realization
        print("avg corr_avg_specific_d across realizations =", np.mean(corr_avg_specific_d_array))
        np.save(filepath_corr_d_specific, corr_avg_specific_d_array)

    # ENTROPY SCALING
    if calcEntropy_scaling and (not os.path.isfile(filepath_entropy_scaling_law)):
        # Suppose we haven't saved the final file containing all the values across the realizations. Then
        # pull up the array from the checkpointing file which contains the data from all realizations, and average
        # over the realizations just for the output, so we get a nice printed set of numbers in the slurm file.

        # sum_abs_Op_squared_array = np.array(sum_abs_Op_squared_array)
        entropy_scaling_array = np.load(filepath_entropy_scaling_law_checkpointing)
        print("avg entropy array (changing numSpins_A) across realizations =",np.mean(entropy_scaling_array, axis=0))  # shape (numrealizations,L-1)
        np.save(filepath_entropy_scaling_law, entropy_scaling_array)

    return ckpt_path


def sigma_z_all_sites(L, hilbert, vstate, nsamples, newSamplesOrNot_corr):
    '''
    Computes the expectation value of sigma_z (the z-component of magnetization) at all sites for a given state.
    
    L: size of chain
    hilbert: netket object defining the hilbert space
    vstate: variational state (contains model, parameters, etc.)
    n_samples: number of samples for correlation function calculation
    newSamplesOrNot_corr: if == 1, produce new samples for every expectation value.
    '''
    if vstate.n_samples != nsamples:
        vstate.n_samples = nsamples
    # https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html:
    # To obtain a new set of samples either use reset() or sample().

    sigma_z_avg_array = []
    sigma_z_sampling_error_array = []
    sigma_z_sampling_error_of_mean_array = []
    for i in range(L):
        if newSamplesOrNot_corr: # if this is True, produce new samples for every expectation value.
            vstate.reset()
        sigma_z_expect = vstate.expect(sigmaz(hilbert, i))
        # avg
        sigma_z_avg = sigma_z_expect.mean.real
        sigma_z_avg_array.append(sigma_z_avg)
        # sampling error
        sigma_z_sampling_error = (sigma_z_expect.variance)**0.5
        sigma_z_sampling_error_array.append(sigma_z_sampling_error)
        # error of mean
        sigma_z_sampling_error_of_mean = sigma_z_expect.error_of_mean
        sigma_z_sampling_error_of_mean_array.append(sigma_z_sampling_error_of_mean)
    return sigma_z_avg_array, sigma_z_sampling_error_array, sigma_z_sampling_error_of_mean_array

def unconnected_corr_avg_error_ij_d(L, hilbert, d, vstate, nsamples, newSamplesOrNot_corr):
    '''
    Computes the unconnected correlation function associated with a given state, for all spins i and j a distance d apart.
    The function returns a list (or array) of all values, the length of which will be L-d, and associated errors and
    variances using NetKet.
    
    L: size of chain
    hilbert: netket object defining the hilbert space
    d: distance of spins for which correlations will be calculated.
    vstate: variational state (contains model, parameters, etc.)
    n_samples: number of samples for correlation function calculation
    newSamplesOrNot_corr: if == 1, produce new samples for every expectation value.
    '''
    list_of_edges = []
    for i in range(L - d):
        list_of_edges.append((i, i + d))
    
    unconnected_corr_avg_array = []
    unconnected_corr_sampling_error_array = []
    unconnected_corr_sampling_error_of_mean_array = []
    #vstate.reset()
    for (i,j) in list_of_edges:
        corr = sigmaz(hilbert, i) * sigmaz(hilbert, j)
        if vstate.n_samples != nsamples:
            vstate.n_samples = nsamples

        '''
        Before we were resetting the vstate to force a new set of samples to be drawn every time
        unconnected_corr_avg_error_ij_d is called, which is for every pair of sites in every wavefunction...
        This is a lot of samples man... Let's use the same set of samples as the ones we use for the
        magnetization. No need to do any resetting here.
        ...
        UPDATE: new hyperparameter that tells us whether or not to reset the state or not
        Resetting the state means producing new samples.
        '''
        if newSamplesOrNot_corr:
            vstate.reset()
        
        corr_expect = vstate.expect(corr)
        
        corr_avg = corr_expect.mean.real
        unconnected_corr_avg_array.append(corr_avg) # initially I had abs(corr_avg) being appended --> WRONG.
        
        corr_sampling_error = (corr_expect.variance)**0.5
        unconnected_corr_sampling_error_array.append(corr_sampling_error)
        
        corr_sampling_error_of_mean = corr_expect.error_of_mean
        unconnected_corr_sampling_error_of_mean_array.append(corr_sampling_error_of_mean)
        

    # FOR NOW WE WON'T Calculate the RNN error in estimate corr. It's complicated. Let's calculate the averages separately.
    return unconnected_corr_avg_array, unconnected_corr_sampling_error_array, unconnected_corr_sampling_error_of_mean_array


'''
This unconnected function (unconnected_corr_avg_all_sites) is wrong...
Taking the absolute value at the end is wrong. If I ever use it, I will need to
correct it, but for now, not using it.
'''
def unconnected_corr_avg_all_sites(L, hilbert, d, vstate, nsamples):
    '''
    Computes the unconnected correlation function (abs value) associated with a given state, and averages over all sitesrelative to which the
    function can be computed. This is not going to be easy to calculate.
    
    L: size of chain
    hilbert: netket object defining the hilbert space
    d: distance of spins for which correlations will be calculated.
    vstate: variational state (contains model, parameters, etc.)
    n_samples: number of samples for correlation function calculation
    '''
    list_of_edges = []
    for i in range(L - d):
        list_of_edges.append((i, i + d))

    corr = sum([sigmaz(hilbert, i) * sigmaz(hilbert, j) for (i, j) in list_of_edges])
    # But remember we also need the expectation value of sigma_z at site i, and the exp. value of sigma_z at site j
    if vstate.n_samples != nsamples:
        vstate.n_samples = nsamples
    # https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html:
    # To obtain a new set of samples either use reset() or sample().
    corr = vstate.expect(corr)
    corr_mean_avg_over_bonds = corr.mean.real / len(list_of_edges)  # dividing by the number of correlation functions we are calculating

    # FOR NOW WE WON'T Calculate the RNN error in estimate corr. It's complicated. Let's calculate the averages separately.
    return abs(corr_mean_avg_over_bonds)


def connected_corr_avg_all_sites(L, hilbert, d, vstate, nsamples, sigma_z_array, newSamplesOrNot_corr):
    '''
    Computes the connected correlation function associated with a given state, and averages over all sites relative to which the
    function can be computed. This is not going to be easy to calculate.
    
    L: size of chain
    hilbert: netket object defining the hilbert space
    d: distance of spins for which correlations will be calculated.
    vstate: variational state (contains model, parameters, etc.)
    n_samples: number of samples for correlation function calculation
    sigma_z_array: expectation value of sigmaz_i at all sites
    newSamplesOrNot_corr: if == 1, produce new samples for every expectation value.
    '''
    list_of_edges = []
    for i in range(L - d):
        list_of_edges.append((i, i + d))
    
    corr_conn_avg_over_all_sites = 0
    for (i,j) in list_of_edges:
        corr = sigmaz(hilbert, i) * sigmaz(hilbert, j)
        if vstate.n_samples != nsamples:
            vstate.n_samples = nsamples

        if newSamplesOrNot_corr:
            vstate.reset()
        
        corr_expect = vstate.expect(corr)
        corr_conn_ij = corr_expect.mean.real - sigma_z_array[i] * sigma_z_array[j]
        corr_conn_avg_over_all_sites += abs(corr_conn_ij)
    
    corr_conn_avg_over_all_sites /= len(list_of_edges)

    # FOR NOW WE WON'T Calculate the RNN error in estimate corr. It's complicated. Let's calculate the averages separately.
    return corr_conn_avg_over_all_sites


@partial(jax.jit, static_argnums=(0, 1, 2, 7, 8,))
def renyi_2_entropy_jax(apply_fn, L, numSpins_A, model_state, parameters, inputs1, inputs2, purity_sampling_error, entropyLogSumExp):
    """
    Computes the 2nd Renyi entropy using the swap operator technique and importance sampling (MC) JAX

    Args:
      model: variational state that will allow us to compute the log_psi values we need for the entropy. Type nk.vqs.MCState.
      inputs1: first set of samples, configurations with dimensions (batch, Hilbert.size).
      inputs2: second set of samples, configurations with dimensions (batch, Hilbert.size).
      parameters: variational parameters, that were initialized when the variational state was defined.
                  type nk.vqs.MCState.parameters
      numSpins_A: number of spins in region A (subsystem defined in partitioning of system)
      seed: seed needed to initialize the state via the initializer functions we define in the model. I have to see how
            that works again.
      exactOrMC: calculating the entropy via MC (importance sampling) or exactly.
      purity_sampling_error: if = 1, let's calculate the RNN sampling error for the purity.
      entropyLogSumExp: if = 1, let's calculate the purity, entropy and purity_std the normal way but also entropy using logsumexp
      purity_sampling_error will only be 1 when entropyLogSumExp = 0, and vice-versa. Also, both can be 0.
      
    Returns:
      The second Renyi entropy
    """

    '''
    dhidden = config['dhidden']
    bias = config['bias'] # True or False
    rnn_cell = config['rnn_cell'] # GRU, Vanilla, Tensorized (Vanilla) Cell
    weight_sharing = config['weight_sharing']
    autoreg = config['autoreg'] # just telling us if the model is autoregressive or not
    numSpins_A = config['numSpins_A']
    '''

    def logpsi(w, σ):
        return apply_fn({"params": w, **model_state}, σ)

    # print("parameters =",parameters)
    inputs1 = jnp.reshape(inputs1, (-1, L))
    inputs2 = jnp.reshape(inputs2, (-1, L))
    ns = inputs1.shape[0] # numsamples
    # print("samples1, jax =",np.array(inputs1))
    # print("samples2, jax =",np.array(inputs2))
    inputs_1A_2B, inputs_2A_1B = swap_A(inputs1, inputs2, numSpins_A)
    log_psi1 = logpsi(parameters, inputs1)
    log_psi2 = logpsi(parameters, inputs2)
    log_psi_1A_2B = logpsi(parameters, inputs_1A_2B)
    log_psi_2A_1B = logpsi(parameters, inputs_2A_1B)

    # jax.debug.print("my_array = {}", log_psi1)

    log_swap = log_psi_1A_2B + log_psi_2A_1B - log_psi1 - log_psi2
    
    if purity_sampling_error:
        swap = jnp.exp(log_swap)
        purity = jnp.mean(swap)
        purity = purity.real
        entropy = -jnp.log(purity)
        
        purity_error = jnp.std(swap)
    elif entropyLogSumExp:
        #purity_error = jnp.std(swap)
        entropy_logSumExp = -1. * jax.scipy.special.logsumexp(a=log_swap,b=1./ns)
    else:
        swap = jnp.exp(log_swap)
        purity = jnp.mean(swap)
        purity = purity.real
        entropy = -jnp.log(purity)
        
    
    #jax.debug.print("purity (complex): {}", purity)
    

    if purity_sampling_error:
        return purity, purity_error, entropy
    #elif entropyLogSumExp:
    #    return purity, purity_error, entropy, entropy_logSumExp
    elif entropyLogSumExp:
        return entropy_logSumExp
    else:
        return purity, entropy


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 9, 10,))
def renyi_2_entropy_jax_chunked(apply_fn, L, numSpins_A, ns, chunk_size, model_state, parameters, inputs1, inputs2, purity_sampling_error, entropyLogSumExp):
    """
    Computes the 2nd Renyi entropy using the swap operator technique and importance sampling (MC) JAX

    Args:
      model: variational state that will allow us to compute the log_psi values we need for the entropy. Type nk.vqs.MCState.
      inputs1: first set of samples, configurations with dimensions (batch, Hilbert.size).
      inputs2: second set of samples, configurations with dimensions (batch, Hilbert.size).
      parameters: variational parameters, that were initialized when the variational state was defined.
                  type nk.vqs.MCState.parameters
      numSpins_A: number of spins in region A (subsystem defined in partitioning of system)
      seed: seed needed to initialize the state via the initializer functions we define in the model. I have to see how
            that works again.
      exactOrMC: calculating the entropy via MC (importance sampling) or exactly.
      purity_sampling_error: if = 1, let's calculate the RNN sampling error for the purity.
      entropyLogSumExp: if = 1, let's calculate the purity, entropy and purity_std the normal way but also entropy using logsumexp
      purity_sampling_error will only be 1 when entropyLogSumExp = 0, and vice-versa. Also, both can be 0.

    Returns:
      The second Renyi entropy
    """

    '''
    dhidden = config['dhidden']
    bias = config['bias'] # True or False
    rnn_cell = config['rnn_cell'] # GRU, Vanilla, Tensorized (Vanilla) Cell
    weight_sharing = config['weight_sharing']
    autoreg = config['autoreg'] # just telling us if the model is autoregressive or not
    numSpins_A = config['numSpins_A']
    '''

    def logpsi(w, σ):
        return apply_fn({"params": w, **model_state}, σ)

    # print("parameters =",parameters)                                                                                                                                                        
    inputs1_chunks = jnp.reshape(inputs1, (ns // chunk_size, chunk_size, L))
    inputs2_chunks = jnp.reshape(inputs2, (ns // chunk_size, chunk_size, L))

    def purity_entropy_chunked(carry, x):
        idx = carry[0]
        in1, in2 = x
        in_1A_2B, in_2A_1B = swap_A(in1, in2, numSpins_A)
        log_psi1 = logpsi(parameters, in1)
        log_psi2 = logpsi(parameters, in2)
        log_psi_1A_2B = logpsi(parameters, in_1A_2B)
        log_psi_2A_1B = logpsi(parameters, in_2A_1B)

        log_swap_chunk = log_psi_1A_2B + log_psi_2A_1B - log_psi1 - log_psi2

        return (idx + 1,), (log_swap_chunk,)

    _, (log_swap_chunks,) = jax.lax.scan(purity_entropy_chunked, init=(0,), xs=(inputs1_chunks, inputs2_chunks))
    print("-----------------")
    print("log_swap_chunks size =",log_swap_chunks.shape)
    print("ns // chunk_size =", ns // chunk_size, chunk_size)
    print("--------------------")
    
    
    if purity_sampling_error:
        purity_chunks = jnp.exp(log_swap_chunks)
        purity = jnp.mean(purity_chunks)
        purity = purity.real
        entropy = -jnp.log(purity)
        
        purity_error = jnp.std(purity_chunks)
    elif entropyLogSumExp:
        #purity_error = jnp.std(purity_chunks)
        entropy_logSumExp = -1. * jax.scipy.special.logsumexp(a=log_swap_chunks,b=1./ns)
    else:
        purity_chunks = jnp.exp(log_swap_chunks)
        purity = jnp.mean(purity_chunks)
        purity = purity.real
        entropy = -jnp.log(purity)
        
    
    # Now let's return everything including the error
    if purity_sampling_error:
        return purity, purity_error, entropy
    #elif entropyLogSumExp:
    #    return purity, purity_error, entropy, entropy_logSumExp
    elif entropyLogSumExp:
        return entropy_logSumExp
    else:
        return purity, entropy
        


def renyi_2_entropy(model, inputs1, inputs2, parameters, numSpins_A, seed, exactOrMC, purity_sampling_error, entropyLogSumExp):
    """
    Computes the 2nd Renyi entropy using the swap operator technique and importance sampling (MC)

    Args:
      model: variational state that will allow us to compute the log_psi values we need for the entropy. Type nk.vqs.MCState.
      inputs1: first set of samples, configurations with dimensions (batch, Hilbert.size).
      inputs2: second set of samples, configurations with dimensions (batch, Hilbert.size).
      parameters: variational parameters, that were initialized when the variational state was defined.
                  type nk.vqs.MCState.parameters
      numSpins_A: number of spins in region A (subsystem defined in partitioning of system)
      seed: seed needed to initialize the state via the initializer functions we define in the model. I have to see how
            that works again.
      exactOrMC: calculating the entropy via MC (importance sampling) or exactly.
      purity_sampling_error: if = 1, let's calculate the RNN sampling error for the purity.
      entropyLogSumExp: if = 1, let's calculate the purity, entropy and purity_std the normal way but also entropy using logsumexp
      purity_sampling_error will only be 1 when entropyLogSumExp = 0, and vice-versa. Also, both can be 0.

    Returns:
      The second Renyi entropy
    """

    '''
    dhidden = config['dhidden']
    bias = config['bias'] # True or False
    rnn_cell = config['rnn_cell'] # GRU, Vanilla, Tensorized (Vanilla) Cell
    weight_sharing = config['weight_sharing']
    autoreg = config['autoreg'] # just telling us if the model is autoregressive or not
    numSpins_A = config['numSpins_A']
    '''
    # print("parameters =",parameters)
    print("Calculating the entropy")
    inputs_1A_2B, inputs_2A_1B = swap_A(inputs1, inputs2, numSpins_A)


    log_psi1 = model.apply({'params': parameters}, inputs1)
    log_psi2 = model.apply({'params': parameters}, inputs2)
    log_psi_1A_2B = model.apply({'params': parameters}, inputs_1A_2B)
    log_psi_2A_1B = model.apply({'params': parameters}, inputs_2A_1B)


    if exactOrMC == 'MC':
        log_swap = log_psi_1A_2B + log_psi_2A_1B - log_psi1 - log_psi2
        swap = jnp.exp(log_swap)
        purity = jnp.mean(swap)
        if purity_sampling_error:
            purity_error = jnp.std(swap)
        elif entropyLogSumExp:
            purity_error = jnp.std(swap)
            entropy_logSumExp = -1. * scipy.special.logsumexp(a=log_swap,b=1./inputs1.shape[0])
    elif exactOrMC == 'exact':
        purity = jnp.sum(jnp.exp(jnp.conj(log_psi1) + jnp.conj(log_psi2) + log_psi_1A_2B + log_psi_2A_1B))
    # this requires sending in the exact configurations and all possible pairs that can be formed from those configs.
    # It's more complicated, so we stick to using the partial trace method to calculate the purity and entropy in the
    # exact case for now.

    # print("purity =",purity)
    purity = purity.real
    entropy = -jnp.log(purity)

    if exactOrMC == 'MC' and purity_sampling_error:
        return purity, purity_error, entropy
    elif exactOrMC == 'MC' and entropyLogSumExp:
        return purity, purity_error, entropy, entropy_logSumExp
    else:
        return purity, entropy


'''
Build the machinery to calculate the entanglement entropy. Build a swap operator function outside of RNN2D and use jax.vmap to do it one
batch element at a time and parallelize those operations. 
'''


# 3 inputs, so in_axes will have 3 elements corresponding to those 3 inputs, and since the first two inputs are sets of samples, then
# both of those inputs should be vmapped along their batch dimension, which is dimension "0".
@partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
def swap_A(x1, x2, numSpins_A):
    """
    Swap the spins from regions A of x1 and x2

    Args:
      x1: first set of samples, configurations with dimensions (batch, Hilbert.size).
      x2: second set of samples, configurations with dimensions (batch, Hilbert.size).
      numSpins_A: numSpins in region A to be swapped. int

    Returns:
      x_1A_2B, x_2A_1B
    """
    x_1A_2B = jnp.append(x1[0:numSpins_A], x2[numSpins_A:])  # shape (L)
    x_2A_1B = jnp.append(x2[0:numSpins_A], x1[numSpins_A:])  # shape (L)
    return x_1A_2B, x_2A_1B


# This function is not needed
def full_density_matrix(model, configs, parameters):
    '''
    Builds the full density matrix of the system (before tracing out). This is a pure state density matrix

    Args:
      model: variational state that will allow us to compute the log_psi values we need to build the density matrix.
      (Type nk.vqs.MCState.)
      parameters: variational parameters, that were initialized when the variational state was defined.
                  type nk.vqs.MCState.parameters
      configs: set of all configurations (computational basis states) spanning the Hilbert space of the full system

    Returns:
      The exact pure state density matrix of the full state
    '''
    # Ok so now we have our configurations.
    log_psi = model.apply({'params': parameters}, configs)
    psi = jnp.exp(log_psi)
    rho = jnp.einsum('i,j->ij', psi, psi.conj())
    return rho


# Partial trace calculation
def partial_trace_np(psi: np.ndarray, keep: list, dims: list) -> np.ndarray:
    r"""
    Calculate the partial trace of an outer product

    .. math::

       \rho_a = \text{Tr}_b (| u \rangle \langle u |)

    Args:
        *psi (tensor)*:
            Quantum state of shape (None ,2,2,...,2), where None is a batch dimension.

        *keep (list)*:
            An array of indices of the spaces to keep after being traced. For instance, if the space is
            A x B x C x D and we want to trace out B and D, keep = [0,2]

        *dims (list)*:
            An array of the dimensions of each space. For instance, if the space is A x B x C x D,
             dims = [None, dim_A, dim_B, dim_C, dim_D]. None is used as a batch dimension.

    Returns (Tensor):
        Partially traced out matrix

    """
    letters = string.ascii_lowercase + string.ascii_uppercase
    keep = [k + 1 for k in keep]
    assert 2 * max(keep) < len(letters) - 1, "Not enough letters for einsum..."
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = letters[-1] + ''.join([letters[i] for i in range(1, Ndim)])
    idx2 = letters[-1] + ''.join([letters[Ndim + i] if i in keep else letters[i] for i in range(1, Ndim)])
    idx_out = letters[-1] + ''.join([i for i, j in zip(idx1, idx2) if i != j] + [j for i, j in zip(idx1, idx2) if i != j])
    psi = np.reshape(psi, dims)
    rho_a = np.einsum(idx1 + ',' + idx2 + '->' + idx_out, psi, np.conj(psi))
    return np.reshape(rho_a, (-1, Nkeep, Nkeep))


# Function that computes the Fourier transform of rho (Pauli strings) for k-body terms
def FT_k_terms(rho, k, N):
    '''
    rho = full density matrix of system, associated with a specific realization. size 2**N x 2**N
    
    k = k-body terms
    
    N = number of spins
    '''
    if k > N:
        print("k is greater than N: impossible")
        raise
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    identity = np.array([[1, 0], [0, 1]])

    # xyz_array_string_form = list(it.product('XYZ', repeat=k))
    # print("xyz_array_string_form =",xyz_array_string_form,len(xyz_array_string_form))
    if k == 1:
        xyz_array = [(pauli_x,), (pauli_y,), (pauli_z,)]
    elif k > 1:
        xyz_array = list(it.product((pauli_x, pauli_y, pauli_z), repeat=k))

    xyz_positions = list(it.combinations(range(N), k))

    sum_abs_Op = 0
    sum_abs_Op_squared = 0
    for i in range(len(xyz_array)):
        # print("xyz_array_string_form =",xyz_array_string_form[count])
        # print(xyz_array_string_form[i])
        for xyz_position in xyz_positions:
            # print("xyz_position =", xyz_position)
            # print()
            # print(xyz_position)
            list_of_paulis = [identity] * N
            # print()
            for j in range(len(xyz_position)):
                list_of_paulis[xyz_position[j]] = xyz_array[i][j]
            pauli_string_kronecker = ft.reduce(np.kron, list_of_paulis)

            O_p = np.trace(np.matmul(rho, pauli_string_kronecker)) / (2 ** N)
            sum_abs_Op += np.abs(O_p)
            sum_abs_Op_squared += np.abs(O_p) ** 2

    return sum_abs_Op, sum_abs_Op_squared


    
