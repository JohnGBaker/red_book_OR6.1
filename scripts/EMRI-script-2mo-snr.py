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
def run_emri_snr(
    emri_prep, 
    emri_kwargs={}
):

    wave_gen=emri_prep['wave_gen']
    priors=emri_prep['priors']
    like = emri_prep['like']
    injection_params=emri_prep['injection_params']
    
    # get XYZ
    data_channels = wave_gen(*injection_params, **emri_kwargs)

    #For SNR calc we set the injected signal to 0
    like.inject_signal(
        data_stream=[0*data_channels[0], 0*data_channels[1]],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "noisepsd_AE", "model": "SciRDv1", "includewd": None} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndraw = 10000

    draws = priors['emri'].rvs(ndraw)

    res=[]
    fails=[]
    for pars in draws:
        try:
            llike=like(pars.reshape(1,-1),**emri_kwargs)
            snr2=-2*llike[0]
            res.append([np.sqrt(snr2),*pars])
            print(*res[-1])
        except Exception as e:
            print(e)
            #snr2=-1
            fails.append([*pars])
            

    res=np.array(res)
    print(len(res),'/',ndraw,'good')
    sres=res[np.argsort(res[:,0])]
    header=' '.join(['','snr']+emri_prep['sampling_pars'])
    print('header',header)
    np.savetxt('sorted.dat',sres,header=header)
    np.savetxt('fails.dat',fails,header=header)

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

    run_emri_snr(
        emri_prep,
        emri_kwargs=waveform_kwargs
    )
    # frequencies to interpolate to
    
    
