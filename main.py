import jax
import jax.numpy as jnp
from jax import random

from zurcher_jax import nfxp

from tabulate import tabulate

jax.config.update("jax_enable_x64", True)

# Set dimension of models
L, H = 2, 4

# Set linear structural parameters
utility_money = 1.0
utility_work = -0.5

# Set vector of linear structural parameters
parameter_values = jnp.asarray([utility_money, utility_work]).copy()

# Set string
parameter_names = ["Utility of money", "Utility of work"]

# Set parameters describing the income process
benefit, wageBase, wageGrowth = 0.1, 1.0, 1.0 / (H - 1.0)

# Set dimensions of arrays containing income and transition probabilities
income = jnp.empty((H, L))
work = jnp.ones((H, L)) * jnp.arange(L)[None,:]

covariates = jnp.empty((H, L, 2))

# Set income process
income = income.at[:, 0].set(benefit)
income = income.at[:, 1].set(wageBase * ((1.0 + wageGrowth) ** jnp.arange(H)))

covariates = covariates.at[..., 0].set(income)
covariates = covariates.at[..., 1].set(work)

transition_prob = jnp.empty((H, H, L))

# Set transition probabilities when not working
transition_prob = transition_prob.at[..., 0].set(jnp.eye(H, k=-1))
transition_prob = transition_prob.at[0, 0, 0].set(1.0)

# Set transition probabilities when working
transition_prob = transition_prob.at[..., 1].set(jnp.eye(H, k=1))
transition_prob = transition_prob.at[-1, -1, 1].set(1.0)

model = nfxp.Zurcher(
    covariates=covariates,
    transition_prob=transition_prob,
)

utility = model.Utility(parameter_values)

solution = model.solve_and_store(utility)

parameters_guess = jnp.zeros_like(parameter_values)

choice_probabilities = jnp.exp(solution.log_q)

observations_per_state = 1_000

observed_choices = random.multinomial(random.PRNGKey(123), observations_per_state, p=choice_probabilities)
print(f"choice probabilities:\n{choice_probabilities}")
print(f"observed choices:\n{observed_choices}")
print(f"total number of observations: {observed_choices.sum()}")

parameter_estimates = model.fit(parameters_guess, observed_choices)

# print tables of true and estimated parameters
print(
    tabulate(
        list(zip(parameter_values, parameter_estimates)), 
        headers=["True parameters", "Estimated parameters"], 
        tablefmt="grid",
    )
)


