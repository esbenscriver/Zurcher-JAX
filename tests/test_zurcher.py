import jax
import jax.numpy as jnp

from zurcher_jax import nfxp

jax.config.update("jax_enable_x64", True)

def test_integration():
    # Set dimension of models
    L, H = 2, 4

    # Set linear structural parameters
    utility_money = 1.0
    utility_work = -0.5

    # Set vector of linear structural parameters
    theta = jnp.asarray([utility_money, utility_work]).copy()

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

    utility = model.Utility(theta)

    solution = model.solve_and_store(utility)

    theta_guess = jnp.zeros_like(theta)

    observed_choices = jnp.exp(solution.log_q)

    theta_estimates = model.fit(theta_guess, observed_choices)

    assert jnp.allclose(theta, theta_estimates), f"Error: {jnp.allclose(theta, theta_estimates) = }"


