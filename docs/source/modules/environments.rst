============
Environments
============

You can use three different environments with this application:

* Lottka-Volterra
* PhysiCell
* Lattice-based model

Lottka-Volterra
---------------
This is a simple model of two species competing for resources.
The model is based on the following equations:

.. math::

    \frac{dx}{dt} = x(\alpha - \beta y)

    \frac{dy}{dt} = -y(\gamma - \delta x)

where :math:`x` and :math:`y` are the populations of the two species.

.. autoclass:: physilearning.envs.LvEnv

