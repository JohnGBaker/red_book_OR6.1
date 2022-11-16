import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import PriorContainer, uniform_dist
import corner

from eryn.moves import StretchMove

from lisatools.sensitivity import get_sensitivity

import prep_emri

#from few.utils.constants import *
np.random.seed(1112)

try:
    import cupy as xp
    # set GPU device
    #xp.cuda.runtime.setDevice(2)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

import warnings

warnings.filterwarnings("ignore")

#set some variables
prep_emri.lisa_gpu_path='/data/jgbaker/sw/lisa-on-gpu/lisa-on-gpu/'
nchannels=2
prep_emri.nchannels=nchannels

# whether you are using 
use_gpu = True

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

# function call
def run_emri_pe(
    emri_prep, 
    fp,
    ntemps,
    nwalkers,
    emri_kwargs={}
):

    wave_gen=emri_prep['wave_gen']
    priors=emri_prep['priors']
    like = emri_prep['like']
    injection_params=emri_prep['injection_params']
    
    # get XYZ
    data_channels = wave_gen(*injection_params, **emri_kwargs)


    like.inject_signal(
        data_stream=[data_channels[i] for i in range(nchannels)],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "noisepsd_AE", "model": "SciRDv1", "includewd": None} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndim = len(emri_prep['sampling_pars'])

    # generate starting points
    emri_injection_params_in=emri_prep['truths'].copy()
    periodics=list(emri_prep['periodic']['emri'].keys())
    
    factor = 1e-5
    cov = np.ones(ndim) * 1e-3
    cov[0] = 1e-5

    start_like = np.zeros((nwalkers * ntemps))
    iter_check = 0
    max_iter = 1000
    while np.std(start_like) < 10.0:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndim))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
            tmp[fix] = (emri_injection_params_in[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndim)))[fix]
            tmp[fix][:,periodics]=np.mod(tmp[fix][:,periodics],2*np.pi)
            
            logp = priors["emri"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        start_like = like(tmp, **emri_kwargs)

        iter_check += 1
        factor *= 1.5

        print(np.std(start_like))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    start_params = tmp.copy()

    start_prior = priors["emri"].logpdf(start_params)

    # start state
    start_state = State(
        {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
        log_prob=start_like.reshape(ntemps, nwalkers), 
        log_prior=start_prior.reshape(ntemps, nwalkers)
    )

    # MCMC moves (move, percentage of draws)
    moves = [
        StretchMove()
    ]

    # prepare sampler
    sampler = EnsembleSampler(
        nwalkers,
        [ndim],  # assumes ndim_max
        like,
        priors,
        tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
        moves=moves,
        kwargs=emri_kwargs,
        backend=fp,
        vectorize=True,
        periodic=emri_prep['periodic'],  # TODO: add periodic to proposals
        #update_fn=None,
        #update_iterations=-1,
        branch_names=["emri"],
    )

    # TODO: check about using injection as reference when the glitch is added
    # may need to add the heterodyning updater

    nsteps = 100
    out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=20, burn=0)

    # get samples
    samples = sampler.get_chain(discard=0, thin=1)["emri"][:, 0].reshape(-1, ndim)

    # plot
    fig = corner.corner(samples, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2))
    fig.savefig(fp[:-3] + "_corner.png", dpi=150)
    return

if __name__ == "__main__":
    # set parameters
    M = 1e6
    a = 0.1  # will be ignored in Schwarzschild waveform
    mu = 100.0
    p0 = 12.0
    e0 = 0.2
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 1.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    #Tobs = 2.05
    Tobs = 0.1667
    dt = 15.0
    fp = "test_run_emri_pe.h5"

    emri_injection_params = np.array([
        M,  
        mu, 
        a,
        p0, 
        e0, 
        x0, 
        qK, 
        phiK, 
        qS, 
        phiS, 
        dist, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0
    ])

    # set parameters
    emri_injection_params_dict = {
        'M':  1e6,  
        'mu': 100, 
        'a':  0.001, # will be ignored in Schwarzschild waveform
        'p0': 12.0, 
        'e0': 0.2, 
        'x0': 1.0,   # will be ignored in Schwarzschild waveform
        'qK': 0.2,   # polar spin angle
        'phiK':0.2,  # azimuthal 
        'qS': 0.3,   # polar sky angle 
        'phiS': 0.3, # azimuthal 
        'dist': 1.0, # Gpc 
        'Phi_phi0':  1.0,  
        'Phi_theta0':2.0, 
        'Phi_r0':    3.0
    }
    
    sampling_pars="lnM,mu,p0,e0,dist,cosqS,phiS,cosqK,phiK,Phi_phi0,Phi_r0".split(',')
    
    #priors
    priors_dict={
                'lnM': uniform_dist(np.log(1e5), np.log(1e6)),  # M
                'mu': uniform_dist(1.0, 1000.0),  # mu
                'p0': uniform_dist(9.0, 16.0),  # p0 #raised LB from 8 bc crash at p/e=8.35/0.65
                'e0': uniform_dist(0.001, 0.7),  # e0
                'cosqK': uniform_dist(-0.99999, 0.99999),  # qK
                'phiK': uniform_dist(0.0, 2 * np.pi),  # phiK
                'cosqS': uniform_dist(-0.99999, 0.99999),  # qS
                'phiS': uniform_dist(0.0, 2 * np.pi),  # phiS
                'dist': uniform_dist(0.01, 100.0),  # dist in Gpc
                'Phi_phi0': uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                'Phi_r0': uniform_dist(0.0, 2 * np.pi),  # Phi_r0
    }

    ntemps = 4
    nwalkers = 30

    waveform_kwargs = {
        "eps": 1e-2
    }

    emri_prep=prep_emri.prep(
        emri_injection_params_dict,
        sampling_pars,
        priors_dict,
        Tobs,
        dt,
        use_gpu
    )        

    run_emri_pe(
        emri_prep,
        fp,
        ntemps,
        nwalkers,
        emri_kwargs=waveform_kwargs
    )
    # frequencies to interpolate to
    
    
