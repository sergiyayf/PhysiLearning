================
Model Evaluation
================

You can evaluate your agent and compare its policy to the fixed adaptive therapy
protocols using the `evaluate` module.

The `evaluate` module contains the script for the default evaluation configured
by the `config.yaml` file, fixed_at utility function and the Evaluate class.

.. autoclass:: physilearning.evaluate.Evaluation
   :members:

.. autofunction:: physilearning.evaluate.fixed_at

.. autofunction:: physilearning.evaluate.evaluate

.. note::

   Fixed adaptive therapy may not work as expected with image observations.