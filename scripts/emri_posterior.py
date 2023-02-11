import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner
from lisatools.utils.utility import AET

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

from lisatools.sensitivity import get_sensitivity

from few.waveform import GenerateEMRIWaveform
from eryn.utils import TransformContainer

from fastlisaresponse import ResponseWrapper

from few.utils.constants import *
np.random.seed(1112)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(6)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

import warnings

warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = False

if use_gpu is True:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

# function call
def run_emri_pe(
    emri_injection_params, 
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    emri_kwargs={}
):

    # sets the proper number of points and what not

    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)

    few_gen = GenerateEMRIWaveform(
        "FastSchwarzschildEccentricFlux", 
        sum_kwargs=dict(pad_output=True),
        use_gpu=use_gpu,
        return_list=False
    )

    orbit_file_esa = "../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

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
    # wave_gen = few_gen

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 14,
       "fill_values": np.array([0.0, 0.0, 0.0]), # spin and inclination and Phi_theta
       "fill_inds": np.array([2, 5, 12]),
    }

    (
        M,  
        mu,
        a, 
        p0, 
        e0, 
        x0,
        dist, 
        qS,
        phiS,
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0
    ) = emri_injection_params

    # get the right parameters
    # log of large mass
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)

    # phases
    emri_injection_params[-1] = emri_injection_params[-1] % (2 * np.pi)
    emri_injection_params[-2] = emri_injection_params[-2] % (2 * np.pi)
    emri_injection_params[-3] = emri_injection_params[-3] % (2 * np.pi)

    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(1e6)),  # M
                1: uniform_dist(1.0, 100.0),  # mu
                2: uniform_dist(12.0, 16.0),  # p0
                3: uniform_dist(0.001, 0.4),  # e0
                4: uniform_dist(0.01, 100.0),  # dist in Gpc
                5: uniform_dist(-0.99999, 0.99999),  # qS
                6: uniform_dist(0.0, 2 * np.pi),  # phiS
                7: uniform_dist(-0.99999, 0.99999),  # qK
                8: uniform_dist(0.0, 2 * np.pi),  # phiK
                9: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        0: np.exp,  # M 
        7: np.arccos, # qS
        9: np.arccos,  # qK
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,

    )

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {6: 2 * np.pi, 8: np.pi, 9: 2 * np.pi, 10: 2 * np.pi}
    }

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # get XYZ
    data_channels = wave_gen(*injection_in, **emri_kwargs)

    check_snr = snr([data_channels[0], data_channels[1]],
        dt=dt,

        PSD="noisepsd_AE",
        PSD_args=(),
        PSD_kwargs={},
        use_gpu=True,
        )

    # this is a parent likelihood class that manages the parameter transforms
    like = Likelihood(
        wave_gen,
        2,  # channels (A,E)
        dt=dt,
        parameter_transforms={"emri": transform_fn},
        use_gpu=use_gpu,
        vectorized=False,
        transpose_params=False,
        subset=1,  # may need this subset
    )

    nchannels = 2
    like.inject_signal(
        data_stream=[data_channels[0], data_channels[1]],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "noisepsd_AE"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndim = 11

    # generate starting points
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

            emri_injection_params_in[5] = emri_injection_params_in[5] % (2 * np.pi)
            emri_injection_params_in[7] = emri_injection_params_in[7] % (2 * np.pi)

            # phases
            emri_injection_params_in[-1] = emri_injection_params_in[-1] % (2 * np.pi)
            emri_injection_params_in[-2] = emri_injection_params_in[-2] % (2 * np.pi)
            
            logp = priors["emri"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        # like.injection_channels[:] = 0.0
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
        log_like=start_like.reshape(ntemps, nwalkers), 
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
        periodic=periodic,  # TODO: add periodic to proposals
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
    mu = 20.0
    p0 = 12.0
    e0 = 0.2
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 3.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    Tobs = 2.05
    dt = 15.0
    fp = "test_run_emri_pe_2.h5"

    emri_injection_params = np.array([
        M,  
        mu, 
        a,
        p0, 
        e0, 
        x0, 
        dist, 
        qS, 
        phiS, 
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0
    ])

    ntemps = 4
    nwalkers = 30

    waveform_kwargs = {
        "T": 1.0,
        "dt": 15.0,
        "eps": 1e-2
    }

    run_emri_pe(
        emri_injection_params, 
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        emri_kwargs=waveform_kwargs
    )
    # frequencies to interpolate to
    
    