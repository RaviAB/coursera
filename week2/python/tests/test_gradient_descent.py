import numpy as np
import pytest

from python.gradient_descent import (
    EX1_FILENAME,
    compute_cost,
    get_data,
    gradient_descent,
)


@pytest.mark.parametrize(
    ('X', 'y', 'theta', 'expected_cost'),
    (
        (*get_data(EX1_FILENAME), np.zeros((2, 1)), 32.1),
        (*get_data(EX1_FILENAME), np.array([[-1, 2]]).T, 54.2),
        (
            np.array([[2, 1, 3], [7, 1, 9], [1, 8, 1], [3, 7, 4]]),
            np.array([[2, 5, 5, 6]]).T,
            np.array([[0.4, 0.6, 0.8]]).T,
            5.29
        ),
    )
)
def test_compute_cost(X, y, theta, expected_cost):
    np.testing.assert_almost_equal(
        compute_cost(X, y, theta),
        expected_cost,
        0.01
    )


@pytest.mark.parametrize(
    ('iterations', 'expected_theta'),
    (
        (1, np.array([[0.0325, 0.1075]]).T),
        (2, np.array([[0.060375, 0.194887]]).T),
        (3, np.array([[0.084476, 0.265867]]).T),
    )
)
def test_gradient_descent(iterations, expected_theta):
    for _ in range(iterations):
        output_theta = gradient_descent(
            np.array([
                [1, 5],
                [1, 2],
                [1, 4],
                [1, 5],
            ]),
            np.array([[1, 6, 4, 2]]).T,
            np.zeros((2, 1)),
            0.01
        )

    np.testing.assert_array_almost_equal(
        output_theta,
        expected_theta,
        0.00001
    )


@pytest.mark.parametrize(
    ('iterations', 'initial_theta', 'expected_theta'),
    (
        (10, np.zeros((3, 1)), np.array([[0.25175, 0.53779, 0.32282]]).T),
        (10, np.array([[0.1, -0.2, 0.3]]).T, np.array([[0.18556, 0.50436, 0.40137]]).T),
    )
)
def test_gradient_descent_multi(iterations, initial_theta, expected_theta):
    for _ in range(iterations):
        output_theta = gradient_descent(
            np.array([
                [2, 1, 3],
                [7, 1, 9],
                [1, 8, 1],
                [3, 7, 4],
            ]),
            np.array([[2, 5, 5, 6]]).T,
            initial_theta,
            0.01
        )

    np.testing.assert_array_almost_equal(
        output_theta,
        expected_theta,
        0.00001
    )
