=========================
Bayesian Inference
=========================

**Bayesian inference** of ODE (LV) model to the ABM or experimental data.

Bayesian inference is a method of statistical inference using Bayes' theorem.
It used to update the probability for a hypothesis as more evidence or information becomes available.
The so-called posterior probability which represents the probability of the model given the available data
is calculated from an initial belief (the prior probability) and the new data using Bayes' theorem.

.. math::
    P(M|D) = \frac{P(D|M)P(M)}{P(D)}

where M is the model and D is the data.
:math:`P(M|D)` is the posterior probability, or probability of the model given
the data. :math:`P(D|M)` is the likelihood, or probability of the data given the model.
:math:`P(M)` is the prior probability, or probability of the model before the data is observed.
:math:`P(D)` is the marginal likelihood, or probability of the data under any model.

The posterior probability becomes the prior probability for the next iteration.
Bayesian inference can be performed iteratively, where each iteration incorporates new evidence to update
the posterior probability.By repeating the process, our beliefs become increasingly refined and better aligned
with the observed evidence.
Wikipedia: https://en.wikipedia.org/wiki/Bayesian_inference


To fit ODE model in this project pymc package is used.
https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html
