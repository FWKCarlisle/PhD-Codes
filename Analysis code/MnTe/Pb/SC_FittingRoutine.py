"""Progressive normal-tip STS models. Requires numpy, scipy, matplotlib.

Energies: meV; biases: mV; temperature: K; modulation: mV RMS.
Set T and VMOD to your measured/effective experimental values before fitting.
"""
from pathlib import Path
from io import StringIO
import csv
from functools import lru_cache
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import least_squares, curve_fit

from datetime import datetime, timezone

# 0. Settings and data loading
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Pb_MnTe/Experiments/Control/Pb(111)_00104.dat')
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Hirsch/Analysis/Pb_SC/Pb(111)_00109.dat')
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Pb_MnTe/Experiments/MnTe2/Pb_MnTe_2_00041.dat')
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Pb_MnTe/Experiments/MnTe2/Pb_MnTe_2_00111.dat')
FILE = Path(r"E:\Nanoscience data\Pb@MnTe\Fitting example data\Pb(111)_00137.dat") ### Example file from Brian's ppt
FILE = Path(r"E:\Nanoscience data\Pb@MnTe\23-26.10.26\Pb_MnTe_2_temp_3_00001.dat")



# Pb(111) for Control Parameters 
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Hirsch/Manuscript_Data/Pb(111)_00137.dat')
# FILE = Path('/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/Pb_MnTe/Experiments/Control/Pb_SingleCrystalControl/Pb(111)_00073.dat')
BIAS = 'Bias calc [AVG] (V)'
SIGNAL = 'LI Demod 1 Y [AVG] (A)'  # Choose the correctly phased channel.

# __________________________________________________________________________________________
T = 0.33               # K: REQUIRED for broadened models
VMOD = 0.04               # mV RMS: REQUIRED for broadened models
# __________________________________________________________________________________________


FIT_LIMIT, NORMAL_MIN = 5.0, 3
DE, NTHETA = 0.002, 64       # Repeat with half DE / double NTHETA to check convergence.
MAKI_DX = 0.00025           # reduced-energy lookup spacing; halve to check convergence
GAP_HARMONIC = 2           # single anisotropic-s harmonic: 2 or 6, for example
NODAL_MIX = 0.5            # weak nodal shape: cos(2 theta) + a*cos(6 theta)
KB = 0.08617333262           # meV/K


def convert_to_epoch(date_time):
    
    dt = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    epoch_seconds = dt.timestamp()
    return epoch_seconds


def load(path, bias=BIAS, signal=SIGNAL):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    start = next(i for i, s in enumerate(lines) if s.strip() == '[DATA]')
    columns = lines[start + 1].split('\t')
    def find(name):
        if name in columns:
            return columns.index(name)
        clean = lambda s: re.sub(r'\s+', ' ', re.sub(r'\[AVG\]', '', s, flags=re.I)).strip()
        hits = [i for i, c in enumerate(columns) if clean(c) == clean(name)]
        if len(hits) != 1:
            raise ValueError(f'Choose an exact column for {name!r}: {columns}')
        return hits[0]
    table = np.genfromtxt(StringIO('\n'.join(lines[start + 2:])), delimiter='\t')
    v, y = table[:, find(bias)] * 1000, table[:, find(signal)]
    valid = np.isfinite(v) & np.isfinite(y)
    v, y = v[valid], y[valid]
    outer = np.abs(v) >= NORMAL_MIN
    if not outer.any():
        raise ValueError('Lower NORMAL_MIN to select a normal-state bias interval.')
    scale = np.mean(y[outer])
    if not np.isfinite(scale) or abs(scale) < 1e-30:
        raise ValueError('Normal-state signal is zero; check channel and phase.')
    order = np.argsort(v)
    return v[order], y[order] / scale

# 1. BCS (and the Dynes generalization): exact energy-bin averages
def dynes(E, gap, gamma=0):
    z = np.asarray(E, dtype=complex)[:, None] + 1j * gamma
    d = np.atleast_1d(gap)[None, :]
    # The product of square roots selects the retarded branch on both sides.
    primitive = lambda x: np.sqrt(x - d) * np.sqrt(x + d)
    return ((primitive(z + DE/2) - primitive(z - DE/2)).real / DE).mean(axis=1)

# 2. Thermal + first-harmonic lock-in broadening
def kernel(E, temperature, modulation):
    def normalize(k):
        if k.sum() == 0:
            raise ValueError('Kernel is unresolved: reduce DE.')
        return k / k.sum()       # discrete probability weights
    impulse = np.zeros_like(E)
    impulse[len(E)//2] = 1
    if temperature < 0 or modulation < 0:
        raise ValueError('T and VMOD must be nonnegative.')
    thermal = impulse.copy()
    if temperature > 0:
        x = np.exp(-np.abs(E) / (KB * temperature))
        thermal = normalize(x / (1 + x)**2)
    lockin = impulse.copy()
    if modulation > 0:
        peak = np.sqrt(2) * modulation
        lockin = normalize(np.sqrt(np.maximum(peak**2 - E**2, 0)))
    return normalize(fftconvolve(thermal, lockin, mode='same'))

# 3. Dynes: call dynes(E, gap, gamma); gamma is in meV.

# 4. Gap symmetry: uniform angular weight, cylindrical Fermi surface
def gaps(delta, symmetry, r=0, theta=None):
    if symmetry == 's':
        return np.array([delta]) if theta is None else np.full_like(theta, delta, dtype=float)
    if symmetry == 'd':
        # One symmetry-equivalent sector, with midpoint sampling (no exact nodes).
        if theta is None:
            theta = (np.arange(NTHETA) + 0.5) * np.pi / (4 * NTHETA)
        shape = np.cos(2*theta) + NODAL_MIX*np.cos(6*theta)
        dense = np.linspace(0, 2*np.pi, 10001)
        maximum = np.max(np.abs(np.cos(2*dense) + NODAL_MIX*np.cos(6*dense)))
        return delta * np.abs(shape) / maximum
    if symmetry == 'anisotropic_s':
        # A symmetry-equivalent sector: uniform angular weight, strictly positive gap.
        if GAP_HARMONIC <= 0 or int(GAP_HARMONIC) != GAP_HARMONIC:
            raise ValueError('GAP_HARMONIC must be a positive integer.')
        if theta is None:
            theta = (np.arange(NTHETA) + 0.5) * np.pi / (GAP_HARMONIC * NTHETA)
        return delta * (1 + r * np.cos(GAP_HARMONIC * theta))
    raise ValueError("symmetry must be 's', 'd' or 'anisotropic_s'")

def plot_gap(parameters, symmetry='anisotropic_s', ax=None, label=None):
    """Plot fitted gap magnitude in meV, using the SAME function as the DOS model."""
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.linspace(0, 2*np.pi, 1441)
    values = gaps(parameters['delta'], symmetry, parameters.get('r', 0), theta=theta)
    ax.plot(theta, values, label=label or symmetry)
    ax.set_ylim(bottom=0)
    ax.set_title(r'Gap magnitude $|\Delta(\theta)|$ (meV)', pad=25)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.25))
    return ax

# 5. Universal Maki DOS versus reduced energy E/gap.
def _maki_reduced(x, zeta):
    e = np.asarray(x) + 1j * 1e-6  # dimensionless retarded branch regularizer
    c = np.zeros(e.shape + (4, 4), dtype=complex)
    c[..., 0, :] = np.stack((2*e, 1-e**2-zeta**2, -2*e, e**2), axis=-1)
    c[..., 1, 0] = c[..., 2, 1] = c[..., 3, 2] = 1
    roots = np.linalg.eigvals(c)
    residual = roots - e[..., None] - zeta * roots / np.sqrt(1-roots**2)
    score = np.abs(residual)
    score = np.where((roots.imag >= 0) & (roots.real >= -1e-9), score, np.inf)
    index = np.argmin(score, axis=-1)[..., None]
    u = np.take_along_axis(roots, index, axis=-1)[..., 0]
    error = np.take_along_axis(score, index, axis=-1)[..., 0]
    if np.any(error > 1e-5 * (1 + np.abs(e))):
        raise RuntimeError('Maki root selection failed; inspect parameters.')
    return (u / (np.sqrt(u - 1) * np.sqrt(u + 1))).real

def maki(E, gap, zeta):
    if zeta == 0:
        return dynes(E, gap)
    # Every sample represents an equal angular interval: preserve multiplicities.
    # np.unique would silently reweight the DOS when float duplicates are uneven.
    gap = np.atleast_1d(gap)
    if np.all(gap == gap[0]):
        gap = gap[:1]
    if np.any(gap <= 0):
        raise ValueError('Maki gap samples must be positive.')
    positive, inverse = np.unique(np.abs(E), return_inverse=True)
    if len(gap) == 1:
        return _maki_reduced(positive / gap[0], zeta)[inverse]
    # Same zeta at every angle: solve ONE universal curve, then interpolate E/gap.
    xmax = max(4., positive[-1] / gap.min())
    dense = np.arange(0., 4. + MAKI_DX/2, MAKI_DX)
    tail = np.geomspace(4., max(4096., xmax * 1.01), 8193)
    edge = max(0., 1 - zeta**(2/3))**1.5
    near_edge = np.maximum(0., edge + np.arange(-128, 129) * MAKI_DX/8)
    x = np.unique(np.concatenate((dense, tail, near_edge)))
    universal = _maki_reduced(x, zeta)
    return np.interp(positive[:, None] / gap[None, :], x, universal).mean(axis=1)[inverse]

def lifetime_broaden(E, n, gamma):
    """E -> E+i*gamma via Lorentzian convolution of a retarded DOS.

    Fourier multiplier exp(-gamma*|k|); padding limits periodic-image errors.
    DOS outside the finite grid is approximated by 1. Check convergence in DE
    and grid extent when resolving very small gamma. gamma is HWHM in meV.
    """
    if gamma < 0:
        raise ValueError('gamma must be nonnegative.')
    if gamma == 0:
        return n
    step = float(E[1] - E[0])
    pad = len(E)
    length = 1 << (8*len(E) - 1).bit_length()
    excess = np.zeros(length)
    excess[pad:pad+len(E)] = n - 1
    multiplier = np.exp(-gamma * 2*np.pi * np.fft.rfftfreq(length, d=step))
    smooth = np.fft.irfft(np.fft.rfft(excess) * multiplier, n=length)
    return 1 + smooth[pad:pad+len(E)]

# 6. Shared forward model and fitting routine
def fit(v, y, kind='dynes', symmetry='s', broaden=True, *,
        temperature=None, modulation=None, plot=True, seed=None, second=None,
        second_gamma=False):
    """second=None, 's' (small isotropic BCS), or 'd' (clean d-wave DOS).

    Two-component options add a phenomenological DOS mixture, not a coupled
    two-band gap equation. second_gamma=True fits Dynes Gamma2 in band 2 only.
    No additional Maki pair-breaking parameter is introduced for band 2.
    """
    if kind == 'dynes_two_band':
        if symmetry != 's' or second is not None or second_gamma:
            raise ValueError('dynes_two_band uses two isotropic bands; omit second/second_gamma.')
        return fit_dynes_two_band(v, y, broaden=broaden, temperature=temperature,
                                 modulation=modulation, plot=plot, seed=seed)
    temperature = T if temperature is None else temperature
    modulation = VMOD if modulation is None else modulation
    if kind not in ('bcs', 'dynes', 'maki', 'maki_gamma'):
        raise ValueError("kind must be 'bcs', 'dynes', 'maki' or 'maki_gamma'")
    if second not in (None, 's', 'd'):
        raise ValueError("second must be None, 's', or 'd'")
    if second is not None and kind != 'maki':
        raise ValueError("Use kind='maki' for the two-component comparison (no extra Gamma).")
    if second_gamma and second is None:
        raise ValueError('second_gamma=True requires a second component.')
    if broaden and (temperature is None or modulation is None):
        raise ValueError('Set T (K) and VMOD (mV RMS) before fitting.')
    extent = max(12., np.max(np.abs(v)) + 0.5 + 20*KB*(temperature or 0) + np.sqrt(2)*(modulation or 0))
    E = np.arange(-int(np.ceil(extent/DE)), int(np.ceil(extent/DE)) + 1) * DE
    K = kernel(E, temperature, modulation) if broaden else None
    names = ['delta', 'scale', 'v0']
    initial, lower, upper = [1.3, 1., 0.], [0.2, 0.1, -0.5], [2.5, 3., 0.5]
    if kind != 'bcs':
        names += ['gamma' if kind == 'dynes' else 'zeta']
        initial += [0.03]
        lower += [0.0001]
        upper += [0.8]
    if symmetry == 'anisotropic_s':
        names += ['r']
        initial += [0.1]
        lower += [0.0]
        upper += [0.8]             # delta_min >= 0.2*delta; no nodes
    if kind == 'maki_gamma':
        names += ['gamma']         # additional lifetime/phenomenological broadening, meV
        initial += [0.005]
        lower += [0.0]
        upper += [0.100]           # exploratory small-Gamma bound; inspect active bounds
    if second is not None:
        names += ['gap2_ratio', 'w2']
        initial += [0.5, 0.03]
        lower += [0.02, 0.0]
        upper += [0.95, 0.20]      # weak second channel: <=20% normal-state tunneling weight
        # delta2 = gap2_ratio * delta * (1-r): smaller than the dominant pairing minimum.
    if second_gamma:
        names += ['gamma2']        # meV; applies ONLY to the weak second component
        initial += [0.003]
        lower += [0.0]
        upper += [0.050]           # 50 u eV exploratory bound; inspect active bounds
    names += ['slope', 'curvature']  # mV^-1 and mV^-2
    span = FIT_LIMIT + 0.5          # includes allowed bias offset
    # |slope*x| <= 0.6 and |curvature*x^2| <= 0.3, so background >= 0.1.
    background_limits = [0.6 / span, 0.3 / span**2]
    initial += [0.0, 0.0]
    lower += [-limit for limit in background_limits]
    upper += background_limits
    @lru_cache(maxsize=16)
    def intrinsic_spectrum(delta, broadening, r):
        d = gaps(delta, symmetry, r)
        return maki(E, d, broadening) if kind in ('maki', 'maki_gamma') else dynes(E, d, broadening)
    @lru_cache(maxsize=16)
    def spectrum(delta, broadening, r, gamma):
        n = lifetime_broaden(E, intrinsic_spectrum(delta, broadening, r), gamma)
        if broaden:
            n = 1 + fftconvolve(n - 1, K, mode='same')  # outside grid: DOS -> 1
        return n
    @lru_cache(maxsize=16)
    def second_spectrum(delta2, gamma2):
        n2 = dynes(E, gaps(delta2, second), gamma=gamma2)
        return 1 + fftconvolve(n2 - 1, K, mode='same') if broaden else n2
    def predict(x, p):
        delta, scale, v0 = p[:3]
        slope, curvature = p[-2:]
        n = spectrum(float(delta), float(p[3]) if kind != 'bcs' else 0.,
                     float(p[names.index('r')]) if 'r' in names else 0.,
                     float(p[names.index('gamma')]) if kind == 'maki_gamma' else 0.)
        if second is not None:
            r = p[names.index('r')] if 'r' in names else 0.
            delta2 = float(p[names.index('gap2_ratio')] * delta * (1-r))
            w2 = p[names.index('w2')]
            gamma2 = float(p[names.index('gamma2')]) if second_gamma else 0.
            # Linear convolution commutes with this constant-weight DOS mixture.
            n = (1-w2)*n + w2*second_spectrum(delta2, gamma2)
        background = 1 + slope * (x - v0) + curvature * (x - v0)**2
        return scale * background * np.interp(x - v0, E, n)
    use = np.abs(v) <= FIT_LIMIT
    if use.sum() <= len(names):
        raise ValueError('Too few points inside FIT_LIMIT.')
    results = []
    starts = [(d, None) for d in (1.0, 1.3, 1.6)]
    if symmetry == 'anisotropic_s':
        # r=0 has zero first derivative: use nonzero starts as well as the null model.
        starts = [(seed.get('delta', 1.3) if seed else 1.3, r) for r in (0., .05, .2, .4)]
    for start_number, (delta0, r0) in enumerate(starts, 1):
        print(f'  Start {start_number}/{len(starts)}: delta={delta0:g}, r={r0}', flush=True)
        p0 = initial.copy()
        if seed:
            p0 = [seed.get(name, value) for name, value in zip(names, p0)]
        p0[0] = delta0
        if r0 is not None:
            p0[names.index('r')] = r0
        if second is not None and not (seed and 'gap2_ratio' in seed):
            p0[names.index('gap2_ratio')] = (.2, .5, .8, .5)[(start_number-1) % 4]
        attempt = least_squares(lambda p: predict(v[use], p)-y[use], p0,
                                bounds=(lower, upper), diff_step=1e-3,
                                x_scale='jac', max_nfev=200)
        results.append(attempt)
        print(f'    converged={attempt.success}; cost={attempt.cost:.6g}; nfev={attempt.nfev}', flush=True)
    successful = [r for r in results if r.success]
    if not successful:
        raise RuntimeError('No fit converged: ' + str(results[-1].message))
    result = min(successful, key=lambda r: r.cost)
    parameters = dict(zip(names, result.x))
    if second is not None:
        parameters['delta2'] = (parameters['gap2_ratio'] * parameters['delta']
                                * (1-parameters.get('r', 0.)))
    result.model_label = f'{kind}, {symmetry}' + (f' + weak {second}' if second else '')
    if second_gamma:
        result.model_label += ' + Gamma2'
    result.predicted = predict(v, result.x)
    result.use = use
    result.parameter_names = names
    result.rmse = np.sqrt(np.mean(result.fun**2))
    n, k = use.sum(), len(names) + 1  # includes fitted residual variance for AICc
    result.aicc = (n*np.log(max(np.mean(result.fun**2), np.finfo(float).tiny))
                   + 2*k + 2*k*(k+1)/(n-k-1)) if n > k+1 else np.inf
    print(kind, symmetry, parameters, 'active bounds:', result.active_mask)
    if not plot:
        return parameters, result, None
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1])
    top = fig.add_subplot(layout[0, 0])
    ax = [top, fig.add_subplot(layout[1, 0], sharex=top)]
    polar = fig.add_subplot(layout[:, 1], projection='polar')
    plot_gap(parameters, symmetry, ax=polar)
    if second is not None:
        plot_gap({'delta': parameters['delta2']}, second, ax=polar,
                 label=f'Band 2: {second}, w={parameters["w2"]:.3f}')
    print(f"File: {FILE.name}, Temp = {temperature:.3g} K, VMOD = {modulation:.3g} mV RMS")
    ax[0].plot(v, y, '.', label='Data')
    ax[0].plot(v[use], predict(v[use], result.x), label=result.model_label)
    ax[0].set_ylabel('Normalized conductance')
    ax[0].set_title(f'Temp = {temperature:.3g} K, VMOD = {modulation:.3g} mV RMS')
    ax[0].legend()
    ax[1].plot(v[use], y[use]-predict(v[use], result.x), '.')
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set(xlabel='Bias (mV)', ylabel='Residual')
    return parameters, result, fig


def fit_dynes_two_band(v, y, broaden=True, *, temperature=None, modulation=None,
                       plot=True, seed=None, shared_gamma=False, max_nfev=500):
    """Two isotropic Dynes DOS: (1-w2)*N(delta1,gamma1)+w2*N(delta2,gamma2).

    Independent Gamma by default; shared_gamma=True fits one common Gamma.
    delta2=delta1*gap2_ratio <= delta1 fixes the band-label ambiguity. w2 is
    the smaller-gap band's normal-state tunnelling weight, allowed from 0 to 1.
    Temperature and modulation are fixed (defaults T, VMOD), not fitted.
    Phenomenological DOS mixture, not a self-consistent coupled gap equation.
    Returns (parameters, optimizer_result, figure), like fit(). Energies meV.
    Seed keys are the optimizer names printed below; delta1 aliases delta.
    """
    temperature = T if temperature is None else temperature
    modulation = VMOD if modulation is None else modulation
    v, y = np.asarray(v, float), np.asarray(y, float)
    if v.ndim != 1 or v.shape != y.shape or not np.isfinite([v,y]).all():
        raise ValueError('v and y must be matching finite one-dimensional arrays.')
    if broaden and (temperature is None or modulation is None):
        raise ValueError('Set temperature (K) and modulation (mV RMS).')
    if broaden and (temperature < 0 or modulation < 0):
        raise ValueError('Temperature and modulation must be nonnegative.')
    extent = max(12., np.max(np.abs(v))+.5+20*KB*(temperature or 0)+np.sqrt(2)*(modulation or 0))
    E = np.arange(-int(np.ceil(extent/DE)), int(np.ceil(extent/DE))+1)*DE
    K = kernel(E, temperature, modulation) if broaden else None
    names = ['delta', 'gap2_ratio', 'w2', 'gamma1']
    initial = [1.4, .90, .5, .005]
    lower = [.2, .05, 0., 0.]
    upper = [2.5, 1., 1., .8]
    if not shared_gamma:
        names += ['gamma2']; initial += [.005]; lower += [0.]; upper += [.8]
    names += ['scale', 'v0', 'slope', 'curvature']
    span = FIT_LIMIT+.5
    initial += [1., 0., 0., 0.]
    lower += [.1, -.5, -.6/span, -.3/span**2]
    upper += [3., .5, .6/span, .3/span**2]
    lower, upper = np.array(lower), np.array(upper)
    use = np.abs(v) <= FIT_LIMIT
    if use.sum() <= len(names):
        raise ValueError('Too few points inside FIT_LIMIT.')

    @lru_cache(maxsize=32)
    def band(delta, gamma):
        n = dynes(E, [delta], gamma)
        return 1+fftconvolve(n-1, K, mode='same') if broaden else n

    def components(x, p):
        q = dict(zip(names, p))
        n1 = band(float(q['delta']), float(q['gamma1']))
        n2 = band(float(q['delta']*q['gap2_ratio']), float(q.get('gamma2',q['gamma1'])))
        shifted = x-q['v0']
        prefactor = q['scale']*(1+q['slope']*shifted+q['curvature']*shifted**2)
        return (prefactor*(1-q['w2'])*np.interp(shifted,E,n1),
                prefactor*q['w2']*np.interp(shifted,E,n2))

    def predict(x, p):
        a,b = components(x,p)
        return a+b

    # Resolve both close Pb-like gaps and wider separations, with both weights.
    starts=[]
    if seed:
        seed=dict(seed)
        if 'delta1' in seed: seed['delta']=seed['delta1']
        starts.append([seed.get(n,b) for n,b in zip(names,initial)])
    for d, ratio, weight, gamma in ((1.3,.95,.3,.002),(1.4,.9,.7,.01),
                                    (1.5,.8,.5,.02),(1.4,.6,.3,.005),
                                    (1.6,.95,.7,.03)):
        p=dict(zip(names,initial)); p.update(delta=d,gap2_ratio=ratio,w2=weight,gamma1=gamma)
        if not shared_gamma: p['gamma2']=gamma
        starts.append([p[n] for n in names])
    attempts=[]
    for i,p0 in enumerate(starts,1):
        result=least_squares(lambda p: predict(v[use],p)-y[use],
            np.clip(p0,lower+1e-10,upper-1e-10), bounds=(lower,upper),
            x_scale='jac',max_nfev=max_nfev)
        attempts.append(result)
        print(f'Two-band start {i}/{len(starts)}: success={result.success}, SSE={2*result.cost:.6g}',flush=True)
    successful=[a for a in attempts if a.success]
    if not successful: raise RuntimeError('No two-band fit converged; increase max_nfev or inspect settings.')
    result=min(successful,key=lambda a:a.cost)
    parameters=dict(zip(names,result.x))
    parameters.update(delta1=parameters['delta'],delta2=parameters['delta']*parameters['gap2_ratio'])
    if shared_gamma: parameters['gamma2']=parameters['gamma1']
    result.parameter_names=names
    result.model_label='Two-band Dynes (shared Gamma)' if shared_gamma else 'Two-band Dynes'
    result.predicted=predict(v,result.x)
    result.components=np.array(components(v,result.x))
    result.use=use; result.rmse=float(np.sqrt(np.mean(result.fun**2)))
    n,k=use.sum(),len(names)+1
    result.aicc=(n*np.log(max(result.rmse**2,np.finfo(float).tiny))+2*k+2*k*(k+1)/(n-k-1)) if n>k+1 else np.inf
    print(result.model_label,parameters,'active bounds:',result.active_mask)
    if parameters['w2']<.01 or parameters['w2']>.99 or parameters['gap2_ratio']>.99:
        print('Bands may be unresolved: a weight is near zero or the gaps nearly coincide.')
    if not plot: return parameters,result,None
    fig=plt.figure(figsize=(11,6),constrained_layout=True)
    layout=fig.add_gridspec(2,2,height_ratios=[3,1],width_ratios=[2,1])
    ax=fig.add_subplot(layout[0,0]); ar=fig.add_subplot(layout[1,0],sharex=ax)
    polar=fig.add_subplot(layout[:,1],projection='polar')
    ax.plot(v,y,'.',ms=3,label='Data'); ax.plot(v[use],result.predicted[use],label=result.model_label)
    for i in range(2): ax.plot(v[use],result.components[i,use],'--',label=f'Weighted band {i+1}')
    ax.set_ylabel('Normalized conductance'); ax.legend()
    ar.plot(v[use],y[use]-result.predicted[use],'.',ms=3); ar.axhline(0,color='k',lw=.5)
    ar.set(xlabel='Bias (mV)',ylabel='Residual')
    for i in (1,2):
        plot_gap({'delta':parameters[f'delta{i}']},symmetry='s',ax=polar,
                 label=f'Band {i}: {parameters[f"delta{i}"]:.3f} meV')
    return parameters,result,fig

def natural_sort_key(value):
    """Sort embedded numbers numerically, e.g. run_2 before run_10."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    )

def temp_dependant_folder(data_dir, temps_file, effective_temps=None):
    files = sorted(
            Path(data_dir).glob("*.dat"),
            key=lambda path: natural_sort_key(path.stem),
        )
    
    dIdVArr = []
    dIdVarr_raw = []
    biasArr = []

    temps = []
    temps_plot = []
    gaps_list = []
    ### Grab temps from summary file
    if effective_temps == None:
        with open(temps_file, "r") as f:
            for i, line in enumerate(f):
                if line.startswith("point,"):
                    print("Found header line:", line)
                    header = line.strip().split(",")
                    continue
                else:
                    # print("Line does not start with 'point':", line)
                    time = convert_to_epoch(line.split(",")[1])
                    temp = float(line.split(",")[2])
                    temps.append(temp)
            temps = np.asarray(temps)
        
            
            if len(temps) != len(files):
                raise ValueError(f"Lengths of file_list ({len(files)}) and file_temps ({len(temps)}) are not equal")
    else:
        temps = effective_temps

        ### 
    for i,file in enumerate(files):
        # if i % 50 == 0 or i < 8:
        print(f"\n {file} \n")
        V, Y = load(Path(file))
        parameters,__,__ = fit(V, Y, 'dynes', temperature=temps[i], modulation=0.04, plot=True)
        delta = parameters['delta'] 
        gaps_list.append(delta)
        temps_plot.append(temps[i])

    fig2,ax2 = plt.subplots()
    temps_plot = np.asarray(temps_plot)
    gaps_list = np.asarray(gaps_list)
    # print("Length of temp")
    ax2.scatter(temps_plot, gaps_list)
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Fitted gap")

    # print(temps_plot)
    def BCS (X, Tc, Gap0):
        gap_t = 1.74 * Gap0 * np.sqrt(1-(X/Tc))

        return gap_t


    ### Fit BCS to data and plot
    para, cov = curve_fit(BCS,temps_plot, gaps_list, p0=(temps_plot.max(), 1.2),maxfev=10000,)
    print(f"Tc: {para[0]}K, Gap_0: {para[1]} meV")
    fit_gap = BCS(temps_plot, para[0],para[1])

    ax2.plot(temps_plot, fit_gap)

    return gaps_list, temps, files, [fig2, ax2]



if __name__ == '__main__':
    
    data_dir = Path(r"E:\Nanoscience data\Pb@MnTe\Temp_dependant_fig")

    temps_file = r"E:\Nanoscience data\Pb@MnTe\Summarys\temp_dependant_fig.csv"

    effective_temps = [
        # 1.78566667 ,1.8076     ,1.819    ,  1.8408,     1.8602,     1.87833333,1.999,      2.1464,     
        2.48566667 ,2.5076     ,2.519    ,  2.5408,     2.5602,     2.57833333,2.699,      2.8464,   # First set with offsets
        4.35766667, 4.38266667, 4.40816667, 4.433,
        4.4575     ,4.4826     ,4.5076     ,4.53216667, 4.5574   ,  4.5824,
        4.6044     ,4.45966667 ,4.5855     ,4.6866     ,4.7794  ,   4.864,
        4.94083333 ,5.0206     ,5.06833333 ,5.12933333 ,5.1686 ,    5.2235,
        5.248      ,5.2925     ,5.32416667 ,5.36233333 ,5.3906,     5.4182,
        5.4406     ,5.49533333 ,5.5544     ,5.6156     ,5.652,      5.708,
        5.7508     ,5.8112     ,5.8515     ,5.9108     ,5.9554,     5.994, 6.05116667]

    gaps, temps, files, plot = temp_dependant_folder(data_dir, temps_file) #  effective_temps=effective_temps )
                                                    
                                                     


    # plt.show()
    # V,Y = load(FILE)
    # Uncomment one stage at a time. Set T and VMOD above for stages 2-5.
    # fit(V, Y, 'bcs', broaden=False)       # 1: ideal BCS, numerical bin averaging only
    # fit(V, Y, 'bcs')               # 2: BCS + thermal + lock-in
    # fit(V, Y, 'dynes')  # 3: Dynes + thermal + lock-in
    # fit_dynes_two_band(V, Y)             # two isotropic gaps, independent Gamma1/Gamma2
    # fit_dynes_two_band(V, Y, shared_gamma=True)  # fewer free parameters
    # fit(V, Y, 'dynes_two_band')          # equivalent entry through fit()
    # fit(V, Y, 'bcs', symmetry='d')        # 4a: d-wave BCS
    # fit(V, Y, 'dynes', symmetry='d')      # 4b: d-wave Dynes
    # fit(V, Y, 'maki')                     # 5a: isotropic Maki
    # fit(V, Y, 'maki', symmetry='d')       # 5b: phenomenological angular Maki
    # fit(V, Y, 'maki', symmetry='anisotropic_s')  # nodeless, fitted r
    # fit(V, Y, 'maki_gamma', symmetry='anisotropic_s',
    #     temperature=.333, modulation=.040)      # nodeless Maki + fitted Gamma (meV)
    # fit(V, Y, 'maki', symmetry='anisotropic_s', second='s',
    #     temperature=.333, modulation=.040)      # weak smaller nodeless component
    # fit(V, Y, 'maki', symmetry='anisotropic_s', second='d',
    #     temperature=.333, modulation=.040)      # weak clean nodal component
    # fit(V, Y, 'maki', symmetry='anisotropic_s', second='d', second_gamma=True,
    #     temperature=.333, modulation=.040)      # Gamma2 broadens only the weak band
    plt.show()
