import jax.numpy as jnp
import numpy as np
import jax
from utils import calculate_cfl
from functools import partial

from heat_eq import get_rollout_fn


Nx = 20
Lx = 2 *np.pi
Nt = 1000
Lt = 2
dt = Lt/(Nt-1)
dx = Lx/(Nx-1)


sim_parameters = {"Nt": Nt, "dx": dx, "dt": dt}

diffusivity = 1
CFL = calculate_cfl(dx,dt,diffusivity)
print(CFL)


x = np.linspace(-Lx/2,Lx/2,Nx)
T_0_function = lambda x: np.sin(x)
T_0_function = lambda x : np.exp(-x**2)

T_0 = T_0_function(x)


CFL = calculate_cfl(dx,dt,diffusivity)
print(CFL)



traj = [T_0]

T = T_0




observation_ratio = 0.5

def get_observation_fn(x,obeservation_ratio):

    Nx = x.shape[0]
    Nobs = int(observation_ratio * Nx)
    x_observed_idx = jnp.linspace(0,Nx-1,Nobs).astype(int)

    def observation_fn(T):
        return T[x_observed_idx]

    return x[x_observed_idx], jax.vmap(observation_fn)


x_observered, observation_fn = get_observation_fn(x, observation_ratio)

rollout_fn = get_rollout_fn(diffusivity,sim_parameters)

import time
start_time = time.time()
traj = rollout_fn(T_0)
end_time = time.time()

print(f"Took {end_time-start_time} s")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))

plt.imshow(traj,cmap="RdBu", aspect="equal",extent=(0,5,0,5))
#plt.show()

gt_traj_observed = observation_fn(traj)


def log_likelihood_fn(ground_truth_data, predicted_data, sigma):
    N = ground_truth_data.shape[0]*ground_truth_data.shape[1]
    ground_truth_data = jnp.reshape(ground_truth_data,(-1,))
    predicted_data = jnp.reshape(predicted_data,(-1,))

    logp = -N * jnp.log(sigma) - 0.5*(1/(sigma**2))*(ground_truth_data-predicted_data).T @ (ground_truth_data-predicted_data)  
    return logp


sigma = 0.05

parameters = {"sigma": sigma, "diffusivity": diffusivity}

MAP = 0


N_mcmc_steps = 10000


mcmc_stepsize = 0.001

key = jax.random.key(0)
key, key1 = jax.random.split(key,2)
log_prior = 0


initial_parameters = jnp.array([0.05, 2.])

def calculate_log_likeliehood(parameters):
    #create new rollout fn with parameters
    rollout_fn = get_rollout_fn(parameters[1],sim_parameters)
    #get trajectory
    traj = rollout_fn(T_0)
    #slice the observations
    traj_observed = observation_fn(traj)
    # calculate the likelihood
    log_likelihood = log_likelihood_fn(gt_traj_observed,traj_observed,parameters[0])
    return log_likelihood


def mcmc_rollout_fn(initial_key,initial_parameters,step_size,logpdf_fn):

    key1,key2 = jax.random.split(initial_key,2)
    initial_concat_parameters = jnp.hstack((initial_parameters,calculate_log_likeliehood(initial_parameters)))
    parameter_delta = jax.random.normal(key1,(N_mcmc_steps,1))


    accept_reject_probability = jnp.log(jax.random.uniform(key2,(N_mcmc_steps,1)))

    random_numbers = jnp.hstack((jnp.zeros_like(parameter_delta),parameter_delta,accept_reject_probability))



    @partial(jax.jit, static_argnums = (1,))
    def metropolis_transition_kernel(random_numbers, logpdf_fn, parameters, log_prob,step_size):

        proposal = parameters + random_numbers[0:2]*step_size

        log_likelihood = logpdf_fn(proposal)
        proposal_log_posterior = log_likelihood + log_prior
        do_accept = random_numbers[2] < proposal_log_posterior - log_prob
        parameters = jnp.where(do_accept,proposal,parameters)
        log_prob = jnp.where(do_accept, proposal_log_posterior, log_prob)

        return parameters, log_prob

    def step_fn(concat_parameters,random_number):
        parameters, log_prob= metropolis_transition_kernel(random_number,
                    logpdf_fn,concat_parameters[:2],
                    concat_parameters[2],step_size)

        concat_parameters = jnp.hstack((parameters,log_prob))
        return concat_parameters, concat_parameters

    def scan_fn():
        _, parameter_traj = jax.lax.scan(step_fn,initial_concat_parameters,random_numbers,N_mcmc_steps)
        return parameter_traj

    return scan_fn
        
mcmc_scan_fn = mcmc_rollout_fn(key1,initial_parameters,mcmc_stepsize,calculate_log_likeliehood)

parameter_traj = mcmc_scan_fn()
burnin = int(N_mcmc_steps/2)
burnin = 1
sigma = parameter_traj[burnin:,0]
diffusivity = parameter_traj[burnin:,1]
log_posterior = parameter_traj[burnin:,2]

plt.figure()
plt.plot(diffusivity)
plt.figure()
plt.scatter(diffusivity,log_posterior)
plt.show()

step_idx_MAP = jnp.argmax(log_posterior)
map_diffusivity = diffusivity[step_idx_MAP]

print(map_diffusivity)


