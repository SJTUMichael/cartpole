import math
import numpy as np


def pd_control(observation):
    x, x_dot, theta, theta_dot = observation
    theta += math.pi
    if theta > math.pi:
        theta -= 2*math.pi
    p = 1
    d = 0.1
    control = theta*p + theta_dot*d

    if control < -0.08:
        action = 0
    elif control < -0.01:
        action = 1
    elif control > 0.08:
        action = 4
    elif control > 0.01:
        action = 3
    else:
        action = 2

    return action

    # control = - 54.4218*x - 24.4898*x_dot + 93.2739*theta + 16.1633*theta_dot
    #
    # if control < -4:
    #     action = 0
    # elif control < -1:
    #     action = 1
    # elif control > 4:
    #     action = 4
    # elif control > 1:
    #     action = 3
    # else:
    #     action = 2
    #
    # return action
