#import numpy as np
import autograd.numpy as np

# TODO Use automatic differentiation for everything.

D_X_CONSTANT = 0
D_X_DRAG = 1
D_ANGLE = 2
D_Z_DRAG = 3
D_MASS = 4

class Quad2DModel:
  def __init__(self, m, g, I, disturbances=None):
    if disturbances is None:
      disturbances = [False] * 5

    self.m = m
    self.g = g
    self.I = I
    self.disturbances = disturbances

    if self.disturbances[D_MASS]:
      self.m += 2

  def get_disturbance(self, x, u):
    x = State(x)
    u = Control(u)

    accel_dist = np.zeros(2)

    if self.disturbances[D_X_CONSTANT]:
      accel_dist[0] -= 4.1
    if self.disturbances[D_X_DRAG]:
      accel_dist[0] -= 3.1 * x.x_vel
    if self.disturbances[D_Z_DRAG]:
      accel_dist[1] -= 3.1 * x.z_vel
    if self.disturbances[D_ANGLE]:
      accel_dist[0] += 1.4 * np.sin(x.theta)

    #xdd -= 4.1 * x.x
    #xdd -= 3.1 * x.x_vel ** 2
    #xdd -= 0.8 * x.x ** 2
    #tdd -= x.theta_vel
    #tdd -= x.x_vel

    return accel_dist

  def deriv(self, x, u):
    x = State(x)
    u = Control(u)

    accel = u.f / self.m
    xdd = -accel * np.sin(x.theta)
    zdd =  accel * np.cos(x.theta) - self.g
    tdd = u.tau / self.I

    accel_vec = np.array((xdd, zdd))

    accel_vec += self.get_disturbance(x, u)

    return np.array((x.x_vel, x.z_vel, x.theta_vel, accel_vec[0], accel_vec[1], tdd))

  def dderiv(self, x, xdot, u):
    x = State(x)
    xdot = State(xdot)
    u = Control(u)

    accel = u.f / self.m
    xddd = (-accel * np.cos(x.theta)) * x.theta_vel
    zddd = (-accel * np.sin(x.theta)) * x.theta_vel
    tddd = 0 # Really, this should be I inv * Tau_dot

    return np.array((xdot.x_vel, xdot.z_vel, xdot.theta_vel, xddd, zddd, tddd))

class State(np.ndarray):
  def __new__(cls, data):
    obj = super(State, cls).__new__(cls, data.shape, data.dtype)
    obj.data = data.data
    return obj

class Control(np.ndarray):
  def __new__(cls, data):
    obj = super(Control, cls).__new__(cls, data.shape, data.dtype)
    obj.data = data.data
    return obj

def set_props(cls_type, props):
  for i, prop in enumerate(props):
    setattr(cls_type, prop, property(lambda self, i=i: self[i]))

set_props(State, ['x', 'z', 'theta', 'x_vel', 'z_vel', 'theta_vel'])
set_props(Control, ['f', 'tau'])
