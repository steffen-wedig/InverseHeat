import heat_eq
import jax 
import jax.numpy as jnp
import numpy as np

from utils import plot_heatmap, calculate_cfl
from observations import get_observation_fn


Nx = 100
Lx = 2
Nt = 1000
Lt = 0.1
dt = Lt/(Nt-1)
dx = Lx/(Nx-1)


x = np.linspace(-Lx/2,Lx/2,Nx)
gt_T_0_fn = lambda x: jnp.sin(np.pi*x)
#gt_T_0_fn = lambda x: jnp.exp(-x**2)
gt_T_0 = jax.vmap(gt_T_0_fn)(x)

gt_diffusivity_fn = lambda x : jnp.exp(-x**2)
#gt_diffusivity_fn = lambda x : 1

gt_diffusivity = jax.vmap(gt_diffusivity_fn)(x)

CFL = calculate_cfl(dx,dt,jnp.max(gt_diffusivity))
print(f"CFL condition number {CFL}")
 
sim_parameters = {"Nt": Nt, "dx": dx, "dt": dt}

gt_rollout_fn = heat_eq.get_rollout_fn(gt_diffusivity,sim_parameters)
gt_temperature_profile = gt_rollout_fn(gt_T_0)



observation_ratio = 0.5
x_observered, observation_fn = get_observation_fn(x, observation_ratio)
observations = observation_fn(gt_temperature_profile)



ig_diffusivity = 0.5 * jnp.ones_like(gt_diffusivity)
ig_T_0 = jnp.ones_like(gt_T_0)

from basis import legendre_basis_1d
order = 5

legendre_poly, get_coeff, get_values = legendre_basis_1d(order,x)

diffusivity_coeff = get_coeff(ig_diffusivity)
T_0_coeff = get_coeff(ig_T_0)

def get_loss_fn(observations):

    @jax.jit
    def loss_fn(parameters):
        diffusivity_coeff = parameters[0,:]
        T_0_coeff = parameters[1,:]
        #convert parameters to values
        diffusivity = get_values(diffusivity_coeff)
        T_0 = get_values(T_0_coeff)

        rollout_fn = heat_eq.get_rollout_fn(diffusivity,sim_parameters)

        traj = rollout_fn(T_0)
        observed_traj = observation_fn(traj)

        loss = jnp.linalg.norm((observations-observed_traj))
        return loss
    return loss_fn

loss_fn = get_loss_fn(observations)


import optax
N_optim = 150
learning_rate = 0.03
optimizer = optax.adam(learning_rate= learning_rate)

params = jnp.vstack((diffusivity_coeff,T_0_coeff))
print(f"Params {params}")

opt_state = optimizer.init(params)

for i in range(N_optim):
    value, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params,updates)
    print(f"Step :{i}, Loss :  {value}")


diffusivity_coeff = params[0,:]
T_0_coeff = params[1,:]
#convert parameters to values
diffusivity = get_values(diffusivity_coeff)
T_0 = get_values(T_0_coeff)
rollout_fn = heat_eq.get_rollout_fn(diffusivity,sim_parameters)
traj = rollout_fn(T_0)

plot_heatmap(traj,Lt,Lx)
plot_heatmap(gt_temperature_profile,Lt,Lx)