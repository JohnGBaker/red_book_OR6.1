import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner

from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral

from eryn.moves import StretchMove

from lisatools.sensitivity import get_sensitivity
from astropy.cosmology import Planck15 as cosmo

from prep_emri import PnTrajectory, PnAAK
import prep_emri

import cupy as cp

print(cp.show_config())
print(cp.array([1]))

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
prep_emri.lisa_gpu_path='/discover/nobackup/znasipak/lisa-on-gpu/'
nchannels=2
prep_emri.nchannels=nchannels

# whether you are using 
use_gpu = True
OMP_NUM_THREADS = 2

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

# function call
def run_emri_snr(
    emri_prep,
    name,
    emri_kwargs={}
):

    wave_gen=emri_prep['wave_gen']
    priors=emri_prep['priors']
    like = emri_prep['like']
    injection_params=emri_prep['injection_params']
    waveform_model=emri_prep['waveform_model']
    
    # get XYZ
    data_channels = wave_gen(*injection_params, **emri_kwargs)

    #For SNR calc we set the injected signal to 0
    like.inject_signal(
        data_stream=[0*data_channels[i] for i in range(nchannels)],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "noisepsd_AE", "model": "SciRDv1", "includewd": None} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndraw = 200

    draws = priors['emri'].rvs(ndraw)
    
    #fom params
    z = 1.5
    M = 5e5*(1. + z)
    lnM = np.log(M)
    mu = 10*(1. + z)
    a = injection_params[2]
    e0 = injection_params[4]
    x0 = injection_params[5]

    inspiral_kwargs = {
	"DENSE_STEPPING": 0,
	"max_init_len": 160
    }

    if waveform_model == 'PnAAK':
        traj_module = PnTrajectory(**inspiral_kwargs)
    elif waveform_model == "Pn5AAKWaveform":
        traj_module = EMRIInspiral(func="pn5", **inspiral_kwargs)
    else:	
        traj_module = EMRIInspiral(func="SchwarzEccFlux", **inspiral_kwargs)
    traj_args = [M, mu, a, e0, x0] 
    p0 = get_p_at_t(
    	traj_module,
    	wave_gen.Tobs,
    	traj_args,
    	index_of_p=3,
    	index_of_a=2,
    	index_of_e=4,
    	index_of_x=5,
    	traj_kwargs={},
    	xtol=2e-12,
    	rtol=8.881784197001252e-16,
    	bounds=[8,30],
    )
    dist = cosmo.luminosity_distance(z).to('Gpc').value
    if waveform_model == "FastSchwarzschildEccentricFlux" and p0 > 16. + 2.*e0:
        p0 = 0.99*(16. + 2.*e0)

    res=[]
    fails=[]
    for pars in draws:
        try:
            pars[0] = lnM
            pars[1] = mu
            pars[2] = p0
            pars[3] = e0
            pars[4] = dist
            llike=like(pars.reshape(1,-1),**emri_kwargs)
            snr2=-2*llike[0]
            res.append([np.sqrt(snr2),*pars])
            #print(*res[-1])
        except Exception as e:
            print(e)
            #snr2=-1
            fails.append([*pars])
            

    res=np.array(res)
    print(len(res),'/',ndraw,'good')
    sres=res[np.argsort(res[:,0])]
    header=' '.join(['','snr']+emri_prep['sampling_pars'])
    #print('header',header)
    np.savetxt('/discover/nobackup/znasipak/sorted_redbook_{}.txt'.format(name),sres,header=header)
    #np.savetxt('/discover/nobackup/znasipak/fails_redbook_1000.txt',fails,header=header)

    return

if __name__ == "__main__":
    # set parameters
    M = 1e6
    a = 0.01  # will be ignored in Schwarzschild waveform
    mu = 100.0
    p0 = 12.0
    e0 = 0.01
    x0 = 0.99  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 1.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 0.0
    Phi_r0 = 3.0

    Tobs = 2.05
    #Tobs = 0.1667
    #Tobs=0.15
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
        'M':  M,  
        'mu': mu, 
        'a':  a, # will be ignored in Schwarzschild waveform
        'p0': p0, 
        'e0': e0, 
        'x0': x0,   # will be ignored in Schwarzschild waveform
        'qK': qK,   # polar spin angle
        'phiK':phiK,  # azimuthal 
        'qS': qS,   # polar sky angle 
        'phiS': phiS, # azimuthal 
        'dist': dist, # Gpc 
        'Phi_phi0':  Phi_phi0,  
        'Phi_theta0':Phi_theta0, 
        'Phi_r0':    Phi_r0
    }
    
    sampling_pars="lnM,mu,p0,e0,dist,cosqS,phiS,cosqK,phiK,Phi_phi0,Phi_r0".split(',')
    
    #priors
    priors_dict={
                'lnM': uniform_dist(np.log(1e5), np.log(5e6)),  # M
                'mu': uniform_dist(1.0, 100.0),  # mu
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

    waveform_kwargs_aak = {}

    emri_prep_few=prep_emri.prep(
        emri_injection_params_dict,
        sampling_pars,
        priors_dict,
        Tobs,
        dt,
        use_gpu
    )

    emri_prep_aak=prep_emri.prep(
        emri_injection_params_dict,
        sampling_pars,
        priors_dict,
        Tobs,
        dt,
        use_gpu,
 	waveform_model="PnAAK"
    )        

#    run_emri_snr(
#        emri_prep_few,
#	"FEW_low_ecc_TDI1_v2",
#        emri_kwargs=waveform_kwargs
#    )

    run_emri_snr(
	emri_prep_aak,
	"AAK_low_ecc_200_nogpu",
	emri_kwargs=waveform_kwargs_aak
    )
    
    
