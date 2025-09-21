import jax
import jax.numpy as jnp
from jax import random

from zurcher_jax import nfxp

from tabulate import tabulate

jax.config.update("jax_enable_x64", True)

# Set dimension of models
M, D = 10, 2

# Set structural parameters
operating_costs= -0.05
replacement_costs = -1.0

# Set vector of structural parameters
parameter_values = jnp.asarray([operating_costs, replacement_costs]).copy()
parameter_names = ["Operating cost", "Replacement costs"]

# Set dimensions of arrays containing income and transition probabilities
mileage = jnp.ones((M, D)) * jnp.arange(M)[:,None]
replace = jnp.ones((M, D)) * jnp.arange(D)[None,:]

covariates = jnp.empty((M, D, 2))

covariates = covariates.at[..., 0].set(mileage)
covariates = covariates.at[..., 1].set(replace)

transition_prob = jnp.empty((M, M, D))

# Set transition probabilities when no replacement
transition_prob = transition_prob.at[..., 0].set(jnp.eye(M, k=1))
transition_prob = transition_prob.at[-1, -1, 0].set(1.0)

# Set transition probabilities when replacement
transition_prob = transition_prob.at[1:, :, 1].set(0.0)
transition_prob = transition_prob.at[0, :, 1].set(1.0)

print(f"transition prob. when no replacement:\n{transition_prob[..., 0]}")
print(f"transition prob. when replacement:\n{transition_prob[..., 1]}")

model = nfxp.zurcher(
    covariates=covariates,
    transition_prob=transition_prob,
)

utility = model.Utility(parameter_values)
solution = model.solve_and_store(utility)

choice_probabilities = jnp.exp(solution.log_q)

observations_per_state = 1_000
observed_choices = random.multinomial(
    random.PRNGKey(123), 
    observations_per_state, 
    p=choice_probabilities,
)

parameters_guess = jnp.zeros_like(parameter_values)
parameter_estimates = model.fit(parameters_guess, observed_choices)

# print tables of true and estimated parameters
print(
    tabulate(
        list(zip(parameter_names, parameter_values, parameter_estimates)), 
        headers=["True parameters", "Estimated parameters"],
        tablefmt="grid",
    )
)
print(f"Number of observations: {observed_choices.sum()}")


