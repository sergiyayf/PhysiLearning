import pandas as pd
import pytest
from physilearning.tools import ODEModel
from physilearning.tools import ODEBayesianFitter
import numpy as np


def test_init():
    ode = ODEModel()
    data = {'sensitive': [0, 2, 1, 2], 'resistant': [0, 1, 2,]}
    fitter = ODEBayesianFitter(ode=ode, data=data)
    assert fitter.model is not None
    assert fitter.data is not None
    assert fitter.ode is not None
    assert fitter.trace is None


def test_set_priors():
    ode = ODEModel()
    data = {'sensitive': [0, 2, 1, 2], 'resistant': [0, 1, 2,]}
    fitter = ODEBayesianFitter(ode=ode, data=data)
    priors = fitter.set_priors()
    assert len(priors) == 5


def test_likelihood():
    treatment_schedule = [0,0,0,0,0]
    ode = ODEModel(tmax=5, dt=1, treatment_schedule=treatment_schedule)
    time = ode.time
    data = pd.DataFrame(dict(x=[0, 1, 2, 3, 4], y=[0, 1, 2, 3, 4], time = time))
    fitter = ODEBayesianFitter(ode=ode, data=data)
    priors = fitter.set_priors()
    likelihood = fitter.likelihood(priors=priors)

    assert likelihood


def test_sample():
    treatment_schedule = [0,0,0,0,0]
    ode = ODEModel(tmax=5, dt=1, treatment_schedule=treatment_schedule)
    time = ode.time
    data = pd.DataFrame(dict(x=[0, 1, 2, 3, 4], y=[0, 1, 2, 3, 4], time = time))
    fitter = ODEBayesianFitter(ode=ode, data=data)
    trace = fitter.sample(draws=10, chains=8)

    assert trace is not None