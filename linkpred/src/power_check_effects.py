import sys, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\claude data\enterprise-network\linkpred\src')
import power_sim as P
L = Path(r'D:\claude data\enterprise-network\linkpred'); P.L = L; P.D = L / 'data_v3x'
W = P.per_positive(L / 'out_dev_gpu10', ['M1+il', 'M1']); m = (W.mindeg <= 1).values
d0 = (W['M1+il'] - W['M1']).values[m]; cl = W.cluster.values[m]; mu = d0.mean()
for target in (mu, .026, .022, .013):
    for name, d in (('scaled', d0 * target / mu), ('shifted', d0 - (mu - target))):
        if target == mu and name == 'shifted': continue
        rng = np.random.default_rng(20261103)
        one, two, q = P.sim_one(d, cl, 140, rng)
        print(f'effect {target:.3f} {name:8s} one-sided {one:.3f} two-sided {two:.3f} est q05/50/95 {q}', flush=True)
