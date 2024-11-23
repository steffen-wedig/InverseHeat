import jax.numpy as jnp
import numpy as np
import jax
from utils import calculate_cfl
from functools import partial



def get_integrator(model,dt):
    def rk45_integrator(T):
        k1 = model(T)
        k2 = model(T+dt*k1/2)
        k3 = model(T+dt*k2/2)
        k4 = model(T+dt*k3)
        return T+dt/6 *( k1 + 2*k2 + 2*k3 +k4)
    return rk45_integrator

def get_heatmodel(diffusivity,sim_parameters):

    dx = sim_parameters["dx"]

    def gradient(T):
        #3 order approximation to gradient
        f = -jnp.roll(T,-2)
        f += 8*jnp.roll(T,-1)
        f += -8*jnp.roll(T,1)
        f += jnp.roll(T,2)
        return f/(12*dx) 

    def laplacian(T):
        f =  -jnp.roll(T,-2)
        f += +16*jnp.roll(T,-1)
        f += -30*T
        f += 16*jnp.roll(T,1)
        f += -jnp.roll(T,2)
        f += f/(12*dx**2)
        return f

    def heat_varying_diffusivity(x):
        return gradient(diffusivity*gradient(x))
        
    return heat_varying_diffusivity




def get_rollout_fn(diffusivity,sim_parameters):

    Nt = sim_parameters["Nt"]
    dt = sim_parameters["dt"]
    model = get_heatmodel(diffusivity,sim_parameters)
    integrator = get_integrator(model,dt)


    def scan_fn(x,_):
        x_next = integrator(x)
        return x_next, x_next
    
    def rollout_fn(T_0):
        _, trj = jax.lax.scan(scan_fn,T_0,None,Nt)
        trj = jnp.vstack((T_0,trj))
        return trj

    return rollout_fn