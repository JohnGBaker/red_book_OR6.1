import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner

from eryn.moves import StretchMove

from lisatools.sensitivity import get_sensitivity

import prep_emri
from prep_emri import PnTrajectory, PnAAK
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t, get_fundamental_frequencies

traj = EMRIInspiral(func='pn5')
traj_2 = PnTrajectory()

print(traj(5e5, 10., 0.5, 11.21, 0.1, 0.7, Phi_r0=0., Phi_theta0=0., Phi_phi0=0., max_init_len=1000)[0][-1])
t, p, e, x, _, _, _ = traj_2(5e5, 10., 0.5, 11.21, 0.1, 0.7, Phi_r0=0., Phi_theta0=0., Phi_phi0=0., max_init_len=150)

print(t, p, e, x)
print(get_fundamental_frequencies(0.5, p, e, x))

wave_model=PnAAK(inspiral_kwargs={'max_init_len':160})
h=wave_model(5e5, 10., 0.5, 11.21, 0.1, 0.7, dist=10., qS=0., phiS=0., qK=0., phiK=0.)