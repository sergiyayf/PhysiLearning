from physilearning.tools.odemodel import ODEModel
import numpy as np


def test_get_treatment_intervals():
    # Arrange
    treatment_schedule = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    ode = ODEModel(treatment_schedule=treatment_schedule)
    # Act
    intervals = np.array(ode.get_treatment_intervals())
    # Assert
    diffs = intervals - np.array([[0, 3], [3, 6], [6, 9]])
    assert np.all(diffs == 0)

def test_LV_no_growth():
    # Arrange
    ode = ODEModel()
    ode.const = {'c_s': 1, 'c_r': 1, 'K': 1, 'Delta_s': 0, 'Delta_r': 0}
    t = 0
    X = [1, 1]
    ode.params = {'r_s': 0, 'r_r': 0, 'delta_s': 0, 'delta_r': 0}
    theta = ode.params.values()
    # Act
    result = ode.LV(t, X, theta)
    # Assert
    assert result == [0, 0]


def test_LV_zero_initial_conditions():
    # Arrange
    ode = ODEModel()
    ode.const = {'c_s': 1, 'c_r': 1, 'K': 1, 'Delta_s': 0, 'Delta_r': 0}
    t = 0
    X = [0, 0]
    ode.params = {'r_s': 1, 'r_r': 1, 'delta_s': 1, 'delta_r': 1}
    theta = ode.params.values()
    # Act
    result = ode.LV(t, X, theta)
    # Assert
    assert result == [0, 0]


def test_LV():
    # Arrange
    ode = ODEModel()
    ode.const = {'c_s': 1, 'c_r': 1, 'K': 9e999, 'Delta_s': 0, 'Delta_r': 0}
    t = 0
    X = [2, 2]
    ode.params = {'r_s': 1, 'r_r': 1, 'delta_s': 0, 'delta_r': 0}
    theta = ode.params.values()
    # Act
    result = np.array(ode.LV(t, X, theta))
    # Assert
    assert abs(result - np.array([2, 2]))[0] < 1e-10
    assert abs(result - np.array([2, 2]))[1] < 1e-10


def test_simulate():
    # arrange
    treatment_schedule = [0, 0]
    ode = ODEModel(treatment_schedule=treatment_schedule,
                   params={'r_s': 1, 'r_r': 1, 'delta_s': 0, 'delta_r': 0},
                   tmax=2, dt = 1, y0=[1, 1])
    ode.const = {'c_s': 1, 'c_r': 1, 'K': 9e999, 'Delta_s': 0, 'Delta_r': 0}
    # act
    result = np.array(ode.simulate())
    diffs = result - np.array([[1, 1], [2.71, 2.71]])
    # assert
    assert np.all(diffs < 1e-2)


def test_prep_treatment_schedule():
    treatment_schedule  = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    ode = ODEModel(treatment_schedule=treatment_schedule)

    # assert if all are int32
    assert all(isinstance(x, np.int32) for x in ode.treatment_schedule)


def test_shuffle_params_and_consts():
    # arrange
    treatment_schedule = [0, 0]
    ode = ODEModel(treatment_schedule=treatment_schedule,
                   params={'c_s': 1, 'c_r': 1, 'Delta_s': 0, 'Delta_r': 0},
                   tmax=2, dt = 1, y0=[1, 1])
    ode.const = {'r_s': 1, 'r_r': 1, 'delta_s': 0, 'delta_r': 0, 'K': 9e999}
    # act
    result = np.array(ode.simulate())
    diffs = result - np.array([[1, 1], [2.71, 2.71]])
    # assert
    assert np.all(diffs < 1e-2)