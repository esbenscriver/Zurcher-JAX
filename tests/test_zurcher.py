import jax
import jax.numpy as jnp

from zurcher_jax import nfxp

jax.config.update("jax_enable_x64", True)

def test_integration():
    # Set dimension of models
    M, D = 10, 2

    # Set structural parameters
    operating_costs= -0.05
    replacement_costs = -1.0

    # Set vector of structural parameters
    parameter_values = jnp.asarray([operating_costs, replacement_costs]).copy()

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

    model = nfxp.Zurcher(
        covariates=covariates,
        transition_prob=transition_prob,
    )

    utility = model.Utility(parameter_values)

    solution = model.solve_and_store(utility)

    parameter_guess = jnp.zeros_like(parameter_values)

    observed_choices = jnp.exp(solution.log_q)

    parameter_estimates = model.fit(parameter_guess, observed_choices)

    assert jnp.allclose(parameter_values, parameter_estimates), f"Error: {jnp.allclose(parameter_values, parameter_estimates) = }"


