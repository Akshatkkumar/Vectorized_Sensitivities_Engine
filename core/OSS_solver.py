import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm

#Uses One-Step-Survival(OSS)and shifts to log-space to use ABM exact sol. 
def gen_gbm(S_0, r, b, v, ts, B, S_ref):
    Xs = [0]
    X = 0
    Ps = []
    for dt in np.diff([0]+ts):
        p = norm.cdf((np.log(B*S_ref/S_0) - X - (r-b)*dt) / (v*dt))
        X = X + ((r-b)-0.5*v**2)*dt + v*np.sqrt(dt)*norm.ppf(np.random.random()*p) 
        Xs.append(X)
        Ps.append(p)
    return S_0*np.exp(Xs), Ps

#Valuees a given structured product based on one price path with OSS for var reduc. 
def value(Ss, Ps, Qs, ts, q, r, B, S_ref):
    L = 1
    val = 0
    for j in range(len(ts)-1):
        val += np.exp(-r*ts[j]) * L * (1-Ps[j]) * Qs[j]
        L = L * Ps[j]
    val += np.exp(-r*ts[-1]) * L * q(Ss[-1]/S_ref) 
    return val

#Values a given structured product using procedural Monte-Carlo and OSS for varience reduction
#Note to self: try using antithetical sampling later, may not work due to mnotonicity requirement
def monte_carlo(S_0, r, b, v, Qs, ts, q, B, S_ref, N):
    vals = []
    for i in range(N):
        Ss, Ps = gen_gbm(S_0, r, b, v, ts, B, S_ref)
        vals.append(value(Ss, Ps, Qs, ts, q, r, B, S_ref))
    return vals

#Estimates delta using a central difference on the MC pricing function.
#note to self: try using RNG seed for future stabilization
def get_vanilla_delta(S_0, r, b, v, Qs, ts, q, B, S_ref, N, eps):
    p_plus = monte_carlo(S_0 + eps, r, b, v, Qs, ts, q, B, S_ref, N)
    p_minus = monte_carlo(S_0 - eps, r, b, v, Qs, ts, q, B, S_ref, N)
    vals = np.divide(np.subtract(p_plus, p_minus), 2 * eps)
    return vals

# Sim parameters from aforementioned ref paper 
S_0 = 3500
S_ref = 4000
B = 1.0
r = 0.04
b = 0
v_fixed = 0.3
ts = [1,2]
Qs = [110, 120]
q = lambda x: 100 * x
N = int(1e4)
eps = 10

estimated_values = monte_carlo(S_0, r, b, v_fixed, Qs, ts, q, B, S_ref, N)
estimated_deltas = get_vanilla_delta(S_0, r, b, v_fixed, Qs, ts, q, B, S_ref, N, eps)
val_mean, val_std = np.mean(estimated_values), np.std(estimated_values)
delta_mean, delta_std = np.mean(estimated_deltas), np.std(estimated_deltas)

# Print results
print("MS+OSS results")
print(f'value mean: {val_mean}, value std: {val_std}')
print(f'delta mean: {delta_mean}, delta std: {delta_std}')
print("-")

# Visualize results using a density-plot with sns KDE
# useful for seeing the continuity in referrence to other approach
fig, axes = plt.subplots(1, 2)
sns.histplot(estimated_values, kde=True, ax=axes[0], stat="density")
axes[0].set_title("Est. Val Dist")
sns.histplot(estimated_deltas, kde=True, ax=axes[1], stat="density")
axes[1].set_title("Est. Deltas Dist")
plt.suptitle("MC+OSS Simulation Results")
plt.show()