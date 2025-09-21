"""
Nested fixed-point (NFXP) algorithm for the empirical model of Optimal Replacement of GMC Bus Engines. For simplicity this version implemented with the following changes relatively to the original proposed version
    - the Newton-Kantorivich iteration method of the inner loop is replaced with the SQUAREM accelerator method
    - the BHHH method of the outer loop is replaced with LBFGS method

Reference:
Rust, John. “Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher.” Econometrica 55, no. 5 (1987): 999–1033. https://doi.org/10.2307/1911259.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import Array

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration, LBFGS
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


@dataclass
class EndogenousVariables(Pytree, mutable=False):
    """Class containing the endogenous variables of the model
    
    Attributes:
        EV (Array): Expected value function
        v (Array): value function
        log_q (Array): the logarithm of the choice probabilities
    """
    EV: Array
    v: Array
    log_q: Array


@dataclass
class zurcher(Pytree, mutable=False):
    """Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher

    Attributes:
        covariates (Array): covariates of utility function
        transition_prob (Array): transition probabilities for milage
        discount_factor (float): covariates of utility function of agents of type Y
        axis (int): axis that characterize the choice set of the agent
    """

    covariates: Array
    transition_prob: Array
    discount_factor: float = 0.95
    axis: int = -1

    def Utility(self, parameters: Array) -> Array:
        """Computes choice-specific utilities

        Args:
            parameters (Array): parameters of utility function

        Returns:
            utility (Array): choice-specific utilities
        """
        # explanation of the subscripts for jnp.einsum()
        #   a: alternatives of the replacement decision
        #   i: current milage level
        #   k: parameters of utility function
        return jnp.einsum("iak, k -> ia", self.covariates, parameters)

    def Expectations(self, EV: Array) -> Array:
        """Compute choice-specific expectations of value function

        Args:
            EV (Array): expected value of next period

        Returns:
            EV_next (Array): choice-specific expected value of next period
        """
        # explanation of the subscripts for jnp.einsum()
        #   a: alternatives of the replacement decision
        #   i: current milage level
        #   j: next period milage level
        return jnp.einsum("ija, j -> ia", self.transition_prob, EV)

    def ValueFunction(self, utility: Array, EV: Array) -> Array:
        return utility + self.discount_factor * EV

    def ExpectedValue(self, EV_old: Array, utility: Array) -> Array:
        """Solve the Bellman equation

        Args:
            EV_old (Array): old guess for the expected value function
            utility (Array): instantanous choice-specific utilities

        Returns:
            EV_new (Array): new guess for the expected value function
        """
        EV_next = self.Expectations(EV_old)
        v = self.ValueFunction(utility, EV_next)
        EV_new = logsumexp(v, axis=self.axis)
        return EV_new

    def log_choice_probabilities(self, v: Array, EV: Array) -> Array:
        """Compute the logarithm of the choice probabilities
        
        Args:
            v (Array): Value function
            EV (Array): Expected value function

        Returns:
            logarithm of the choice probabilities (Array)
        """
        return v - jnp.expand_dims(EV, axis=-1)

    def choice_probabilities(self, log_p: Array) -> Array:
        """Computes the choice probabilities from the logarithm of the choice probabilities
        
        Args:
            log_p (Array): logarithm of the choice probabilities

        Returns:
            choice probabilities (Array)
        """
        return jnp.exp(log_p)

    def solve(
        self,
        utility: Array,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False,
    ) -> Array:
        """Inner loop of the NFXP algorithm that solves EV

        Args:
            utility (Array): choice-specific utilities
            fixed_point_solver (SolverTypes): solver used for solving fixed point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            maxiter (int): maximum number of iterations
            verbose (bool): whether to print information on every iteration or not.

        Returns:
            EV (Array): expected (stationary) value function
        """
        # Initial guess for equilibrium transfers
        EV_init = jnp.zeros(utility.shape[:-1])

        # Find equilibrium transfers
        result = fixed_point_solver(
            self.ExpectedValue,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        ).run(EV_init, utility)
        return result.params

    def solve_and_store(self, utility: Array) -> EndogenousVariables:
        EV = self.solve(utility)
        EV_next = self.Expectations(EV)
        v = self.ValueFunction(utility, EV_next)
        log_q = self.log_choice_probabilities(v, EV)
        return EndogenousVariables(EV=EV, v=v, log_q=log_q)

    def neg_log_likelihood(self, params: Array, observed_choices: Array) -> Array:
        """Computes the negative log-likelihood function

        Args:
            params (Array): parameters of agents' utility functions
            observed_choices (Array): observed choices

        Returns:
            neg_log_lik (Array): negative log-likelihood value
        """
        utility = self.Utility(params)

        endo = self.solve_and_store(utility)

        number_of_observations = jnp.sum(observed_choices)
        neg_log_lik = (
            -jnp.nansum(observed_choices * endo.log_q) / number_of_observations
        )
        return neg_log_lik

    def fit(
        self,
        guess: Array,
        observed_choices: Array,
        tol: float = 1e-10,
        maxiter: int = 100,
        verbose: bool | int = True,
    ) -> Array:
        """Outer loop of the NFXP algorithm that estimate the parameters by maximizing the log-likelihood function

        Args:
            guess (Array): initial parameter guess
            observed_choices (Array): observed transfers and numbers of matched and unmatched agents
            tol (float): tolerance of the stopping criterion
            maxiter (int): maximum number of proximal gradient descent iterations
            verbose (bool): if set to True or 1 prints the information at each step of the solver, if set to 2, print also the information of the linesearch

        Returns:
            params (Array): parameter estimates
        """

        result = LBFGS(
            fun=self.neg_log_likelihood,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
        ).run(guess, observed_choices)
        return result.params
