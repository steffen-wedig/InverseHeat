import jax.numpy as jnp
from jax import vmap


def get_observation_fn(x,observation_ratio):

    Nx = x.shape[0]
    Nobs = int(observation_ratio * Nx)
    x_observed_idx = jnp.linspace(0,Nx-1,Nobs).astype(int)

    def observation_fn(T):
        return T[x_observed_idx]

    return x[x_observed_idx], vmap(observation_fn)