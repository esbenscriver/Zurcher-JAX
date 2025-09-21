[![CI](https://github.com/esbenscriver/Zurcher-JAX/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/Zurcher-JAX/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/Zurcher-JAX/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/Zurcher-JAX/actions/workflows/cd.yml)

# Zurcher-JAX
This package implements, in JAX, a modified version of the nested fixed-point (NFXP) algorithm for the empirical model of Optimal Replacement of GMC bus engines for Harold Zurcher, originally introduced by [Rust (1987)](https://doi.org/10.2307/1911259). The algorithm has been modified in the following ways:  
- The Newton–Kantorovich iteration method in the inner loop is replaced with the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) acceleration method.  
- The BHHH method in the outer loop is replaced with the LBFGS method.  

The SQUAREM accelerator is employed to efficiently solve the nested fixed-point problem without relying on gradient information. In addition, the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation is used to automatically compute the gradient of the log-likelihood function.
