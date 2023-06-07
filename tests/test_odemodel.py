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
    ode.const = {'c_x': 1, 'c_y': 1, 'K': 1, 'Delta_y': 0, 'Delta_x': 0}
    t = 0
    X = [1, 1]
    theta = [0, 0, 0, 0]
    # Act
    result = ode.LV(t, X, theta)
    # Assert
    assert result == [0, 0]


def test_LV_zero_initial_conditions():
    # Arrange
    ode = ODEModel()
    ode.const = {'c_x': 1, 'c_y': 1, 'K': 1, 'Delta_y': 0, 'Delta_x': 0}
    t = 0
    X = [0, 0]
    theta = [1, 1, 1, 1]
    # Act
    result = ode.LV(t, X, theta)
    # Assert
    assert result == [0, 0]


def test_LV():
    # Arrange
    ode = ODEModel()
    ode.const = {'c_x': 1, 'c_y': 1, 'K': 9e999, 'Delta_y': 0, 'Delta_x': 0}
    t = 0
    X = [2, 2]
    theta = [1, 1, 0, 0]
    # Act
    result = np.array(ode.LV(t, X, theta))
    # Assert
    assert abs(result - np.array([2, 2]))[0] < 1e-10
    assert abs(result - np.array([2, 2]))[1] < 1e-10


def test_simulate():
    # arrange
    treatment_schedule = [0, 0]
    ode = ODEModel(treatment_schedule=treatment_schedule,
                   params=[1, 1, 0, 0],
                   tmax=2, dt = 1, y0=[1, 1])
    ode.const = {'c_x': 1, 'c_y': 1, 'K': 9e999, 'Delta_y': 0, 'Delta_x': 0}
    # act
    result = np.array(ode.simulate())
    diffs = result - np.array([[1, 1], [2.71, 2.71]])
    # assert
    assert np.all(diffs < 1e-2)
