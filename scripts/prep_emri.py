import numpy as np
from eryn.prior import PriorContainer, uniform_dist
#from lisatools.utils.utility import AET

from lisatools.sampling.likelihood import Likelihood
from lisatools.sensitivity import get_sensitivity

from few.waveform import GenerateEMRIWaveform
from eryn.utils import TransformContainer

from fastlisaresponse import ResponseWrapper

from few.utils.constants import *

import warnings

warnings.filterwarnings("ignore")

#some things which might be externally adjusted
lisa_gpu_path='/data/jgbaker/sw/lisa-on-gpu/lisa-on-gpu/'
subset=1

# function call
def prep(
        injection_params_dict,
        sampling_pars,
        prior_dict,
        Tobs,
        dt,
        use_gpu
):

    # sets the proper number of points and what not

    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    #t_arr = xp.arange(N_obs) * dt

    # frequencies
    #freqs = xp.fft.rfftfreq(N_obs, dt)

    few_gen = GenerateEMRIWaveform(
        "FastSchwarzschildEccentricFlux", 
        sum_kwargs=dict(pad_output=True),
        use_gpu=use_gpu
    )

    orbit_file_esa = lisa_gpu_path+"orbit_files/esa-trailing-orbits.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "2nd generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 9
    index_beta = 8

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird
    
    wave_gen = ResponseWrapper(
        few_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

    #There are relevant lists of parameters to relate injections,
    #sampling params and the ordered params needed for the signal calculation
    #There may be translations between these. Where possible, we do this here
    #automatically
    signames="M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0".split(',')
    nsigpars=len(signames)
    #Specify which are 2-pi periodic
    periodic_pars="phiS,phiK,Phi_phi0,Phi_theta0,Phi_r0".split(',')
    
    #Define transforms for how to get the signal params from other vars

    #This goes in a dict with sig-var:{other-var:,[other->sig func, inverse]}
    autotransforms={
        'M':{'lnM':[lambda x: np.exp(x),lambda x:np.log(x)]},
        'mu':{'lnmu':[lambda x: np.exp(x),lambda x:np.log(x)]},
        'qK':{'cosqK':[lambda x: np.arccos(x),lambda x:np.cos(x)]},
        'qS':{'cosqS':[lambda x: np.arccos(x),lambda x:np.cos(x)]}
    }

    #Now map injection pars to the signal pars and produce the injection array
    injection_params_array=np.zeros(nsigpars)
    for i in range(nsigpars):
        par=signames[i]
        
        if par in injection_params_dict:
            val=injection_params_dict[par]
        else:
            val=None
            if par in autotransforms:
                transopts=autotransforms[par]
                for otherpar in transopts:
                    if otherpar in injection_params_dict:
                        trans=transopts[otherpar][0]
                        val=trans(injection_params_dict[otherpar])
                        break
            if val is None:
                raise ValueError('Could not find or compute value for '+par+' from injection params:',injection_params_dict)
        if par in periodic_pars: val=val%(2*np.pi)
        injection_params_array[i]=val
    
    # We now set up how to fill signal parameters from sampling params
    # First determine which signal pars we can map from the sampling params
    # together with their transforms and order.  The way the codes work, the
    # order, after processing, has to match that of the signal, with the
    # fixed params excluded.
    ordered_sampling_pars=[]
    truths=[]
    sampling_param_transforms={}
    fixed_inds=[]
    fixed_vals=[]
    periodic_sampling_inds=[]
    for i in range(nsigpars):
        par=signames[i]
        sname=None
        trans=None
        if par in sampling_pars:
            sname=par
        else:
            if par in autotransforms:
                transopts=autotransforms[par]
                for otherpar in transopts:
                    if otherpar in sampling_pars:
                        sname=otherpar
                        trans=transopts[otherpar]
                        break
        if sname is None:
            fixed_inds.append(i)
            fixed_vals.append(injection_params_array[i])
        else:
            idx=len(ordered_sampling_pars)
            ordered_sampling_pars.append(sname)
            truth=injection_params_array[i]
            if trans is not None:
                sampling_param_transforms[idx]=trans[0]
                truth=trans[1](truth)
            if par in periodic_pars:
                periodic_sampling_inds.append(idx)
                truth=truth%(2*np.pi)
            #we may not need this, but have the info for it here
            truths.append(truth)

    print('ordered_sampling_pars',ordered_sampling_pars)
    print('periodic_sampling_inds',periodic_sampling_inds)
    print('fixed_inds',fixed_inds)
    print('injection_params_array',injection_params_array)
    print('truths',truths)
    truths=np.array(truths)
    ndim = len(ordered_sampling_pars)
    
    # generate starting points
    #Package the transformation info
    fill_dict = {
       "ndim_full": nsigpars,
       "fill_values": np.array(fixed_vals), 
       "fill_inds": np.array(fixed_inds),
    }
    transform_fn = TransformContainer(
        parameter_transforms=sampling_param_transforms,
        fill_dict=fill_dict,
    )

    #Handle periodic vars
    periodic = {
        "emri": {i:2*np.pi for i in periodic_sampling_inds}
    }

    # priors
    priors={}
    for i in range(ndim):
        par=ordered_sampling_pars[i]
        if par in prior_dict:
            priors[i]=prior_dict[par]
        else:
            raise ValueError('Could not find prior for sampling par ',par,'in',list(prior_dict.keys()))
    priors = {
        "emri": PriorContainer(priors)
    }

    ## get injected parameters after transformation
    #injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # this is a parent likelihood class that manages the parameter transforms
    like = Likelihood(
        wave_gen,
        2,  # channels (A,E)
        dt=dt,
        parameter_transforms={"emri": transform_fn},
        use_gpu=use_gpu,
        vectorized=False,
        transpose_params=False,
        subset=subset,  
    )

    #Package up the results
    res={}
    res['injection_params']=injection_params_array
    res['wave_gen']=wave_gen
    res['sampling_pars']=ordered_sampling_pars
    res['truths']=truths
    res['transforms']=transform_fn
    res['periodic']=periodic
    res['priors']=priors
    res['like']=like
    
    return res
