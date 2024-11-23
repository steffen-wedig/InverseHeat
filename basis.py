import jax.numpy as jnp
import numpy as np
#Transform from value to coefficient 


import jax.numpy as jnp

def legendre_basis_1d(order, x):
    """
    Generate the Legendre polynomial basis up to a given order for input x.
    """
    legendre_poly = jnp.zeros((order, x.shape[0]))
    
    # Initialize the first two Legendre polynomials
    legendre_poly = legendre_poly.at[0, :].set(1)  # P_0(x) = 1
    if order > 1:
        legendre_poly = legendre_poly.at[1, :].set(x)  # P_1(x) = x
    
    # Recurrence relation: P_{n+1}(x) = ((2n+1)xP_n(x) - nP_{n-1}(x)) / (n+1)
    for i in range(1, order - 1):
        legendre_poly = legendre_poly.at[i + 1, :].set(
            ((2 * i + 1) * x * legendre_poly[i, :] - i * legendre_poly[i - 1, :]) / (i + 1)
        )

    def get_coefficients(values):
        """
        Calculate the expansion coefficients for a given function using Legendre polynomials.
        
        Args:
            values (jax.numpy.ndarray): The function values at given x points.

        Returns:
            coeff (jax.numpy.ndarray): The Legendre expansion coefficients.
        """
        # Weights for uniform sampling
        N = x.shape[0]
        dx = 2 / N  # Uniform spacing assumption
        coeff = jnp.zeros(order)
        
        for n in range(order):
            coeff = coeff.at[n].set(
                (2 * n + 1) / 2 * jnp.sum(values * legendre_poly[n, :] * dx)
            )
        return coeff

    def get_values(coefficients):
        """
        Reconstruct the function values from Legendre expansion coefficients.
        """
        values = jnp.sum(coefficients[:, None] * legendre_poly, axis=0)
        return values

    return legendre_poly, get_coefficients, get_values


x = np.linspace(-1,1,1000)
order = 5

legendre_poly, get_coeff, get_values = legendre_basis_1d(order,x)

fn = lambda x : jnp.sin(jnp.pi*x)

y = fn(x)
print(y)
legendre_coeff = get_coeff(y)
print(legendre_coeff)
y_dash = get_values(legendre_coeff)

distance = jnp.linalg.norm((y - y_dash))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x,y)
plt.plot(x,y_dash)
plt.show()

