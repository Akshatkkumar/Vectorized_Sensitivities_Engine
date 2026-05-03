import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import seaborn as sns
from jax.scipy.stats import norm

#Generates a path using One-Step-Survival (OSS) logic in JAX.
#note to self: no vmap because X_{i+1} depends on previous layer. have to use .scan
@jax.jit
def gen_gbm_oss_jax(S_0, r, b, v, ts, B, S_ref, key):
    dts = jnp.diff(jnp.insert(ts, 0, 0.0))
    log_barrier = jnp.log(B*S_ref / S_0)

    def step_fn(carry, inputs):
        X_prev, current_key = carry
        dt = inputs
        # survival prob.
        p = norm.cdf((log_barrier - X_prev - (r - b) * dt) / (v * jnp.sqrt(dt)))
        # draw from truncated normal
        u = jax.random.uniform(current_key)
        z = norm.ppf(u * p)
        X_next = X_prev + ((r - b) - 0.5 * v**2) * dt + v * jnp.sqrt(dt) * z
        new_key, _ = jax.random.split(current_key)
        return (X_next, new_key), (X_next, p)

    keys = jax.random.split(key, len(dts))
    _, (Xs, Ps) = jax.lax.scan(step_fn, (0.0, key), dts)
    
    # Return terminal log-returns and the survival probability vector
    return jnp.insert(Xs, 0, 0.0), Ps

# Values path using OSS weightings.
# note to self: mention differentiable stability from cont. vectorized Obj. func. 
@jax.jit
def value_oss_jax(Xs, Ps, Qs, ts, S_0, S_ref, r):
    # L actually tracks survival till t_j
    L = 1.0
    val = 0.0
    
    # contribution from coupons and knockout
    for j in range(len(ts)-1):
        val += jnp.exp(-r * ts[j]) * L * (1.0 - Ps[j]) * Qs[j]
        L *= Ps[j]
        
    # addes contribution from surviving paths
    terminal_payout = 100.0 * (S_0 * jnp.exp(Xs[-1]) / S_ref)
    val += jnp.exp(-r * ts[-1]) * L * terminal_payout
    return val

# Run MC with JAX, no loop through paths needed thanks to vectorization
# Note to self: test-hardware speedup based on hardware+threading
# Note to self: Be careful with rand nums when testing threading
def monte_carlo_oss_jax(S_0, r, b, v, Qs, ts, B, S_ref, N, key):
    keys = jax.random.split(key, N)
    def single_path(k):
        Xs, Ps = gen_gbm_oss_jax(S_0, r, b, v, ts, B, S_ref, k)
        return value_oss_jax(Xs, Ps, Qs, ts, S_0, S_ref, r)
    return jax.vmap(single_path)(keys)

#find delta with a backward pass through the DCG. AAD implementation
#Note ot self: JAX delta uses shared random key, in ablation study look at effect of RNG seed normalization+CD vs AAD
#Note to self: No longer have N deltas to compare mean and var, run repeated delta calcs instead
def get_delta_aad_jax(S_0: float, r: float, b: float, v: float, Qs: jnp.ndarray, ts: jnp.ndarray, B: float, S_ref: float, N: int, key: jax.random.PRNGKey) -> float:
    # wrapper func for delta calc
    def expected_value_fn(s_start):
        paths = monte_carlo_oss_jax(s_start, r, b, v, Qs, ts, B, S_ref, N, key)
        return jnp.mean(paths)
    #implements AAD
    grad_fn = jax.grad(expected_value_fn)
    return grad_fn(S_0)

# Sim parameters from aforementioned ref paper 
S_0 = 3500.0
S_ref = 4000.0
B = 1.0
r = 0.04
b = 0.0
v_fixed = 0.5
ts = jnp.array([1.0, 2.0])
Qs = jnp.array([110.0, 120.0])
N = int(1e4)

m_key = jax.random.PRNGKey(42)
v_key, d_key = jax.random.split(m_key)

estimated_values = monte_carlo_oss_jax(S_0, r, b, v_fixed, Qs, ts, B, S_ref, N, v_key)
# For delta std we repeatedly calc delta since we no longer have multiple pathwise deltas after AAD
estimated_delta = get_delta_aad_jax(S_0, r, b, v_fixed, Qs, ts, B, S_ref, N, d_key)
val_mean, val_std = jnp.mean(estimated_values), jnp.std(estimated_values)

# Print results
print("MC+OSS+XLA/AAD results")
print("Note: No Delta distribution is available since AAD returns only one result, rather than one per path")
print(f'value mean: {val_mean}, value std: {val_std}')
print(f'delta mean: {estimated_delta}, delta std: N/A')
print("-")
# Visualize results using a density-plot with sns KDE
# useful for seeing the continuity in referrence to other approach
sns.histplot(estimated_values, kde=True, stat="density")
plt.title("Est. Val Dist")
plt.suptitle("MC+OSS Simulation Results")
plt.show()