import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Generates a price path realization under GBM using the exact solution on discrete time steps
# note to self: product on line 11 could be unstable
def gen_gbm(S_0: float, r: float, b: float, v: float, ts: float):
    Ss = [S_0]
    S = S_0
    for dt in np.diff([0]+ts):
        S = S*np.exp((((r-b)-0.5*v**2)*dt + v*np.sqrt(dt)*np.random.normal()))
        Ss.append(S)
    return Ss

# Values a given structured product based on one price path
# note to self: this is not continuous since it is done without OSS
def value(Ss, Qs, ts, q, r, B, S_ref):
    for s_i in range(1,len(Ss)):
        if Ss[s_i] > B*S_ref:
            return np.exp(-r*(ts[s_i-1]))*Qs[s_i-1]
    return(np.exp(-r*(ts[-1]))*q(Ss[-1]/S_ref))

# Values a given structured product using procedural Monte-Carlo
def monte_carlo(S_0, r, b, v, Qs, ts, q, B, S_ref, N):
    vals = []
    for i in range(N):
        vals.append(value(gen_gbm(S_0, r, b, v, ts), Qs, ts, q, r, B, S_ref))
    return vals

# Estimates delta using a central difference on the MC pricing function.
# note to self: Try using RNG seed for stabilization
def get_vanilla_delta(S_0, r, b, v, Qs, ts, q, B, S_ref, N, eps):
    p_plus = monte_carlo(S_0 + eps, r, b, v, Qs, ts, q, B, S_ref, N)
    p_minus = monte_carlo(S_0 - eps, r, b, v, Qs, ts, q, B, S_ref, N)
    vals = np.divide(np.subtract(p_plus, p_minus), 2 * eps)
    return vals

# Sim Parameters pulled directly from the reference paper
S_0 = 3500
S_ref = 4000
B = 1.0
r = 0.04
b = 0
v_fixed = 0.2
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
print("Procedural Monte Carlo results: ")
print(f'value mean: {val_mean}, value std: {val_std}')
print(f'delta mean: {delta_mean}, delta std: {delta_std}')
print("_")

# Visualize results using a density-plot with sns KDE
# useful for seeing the continuity in referrence to other approach
fig, axes = plt.subplots(1, 2)
sns.histplot(estimated_values, kde=True, ax=axes[0], stat="density")
axes[0].set_title("Est. Val Dist")
sns.histplot(estimated_deltas, kde=True, ax=axes[1], stat="density")
axes[1].set_title("Est. Deltas Dist")
plt.suptitle("MC Simulation Results")
plt.show()