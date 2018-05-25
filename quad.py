import numpy as np

# TODO Use automatic differentiation for everything.

class Quad2DModel:
  def __init__(self, m, g, I, add_more=False):
    self.m = m
    self.g = g
    self.I = I
    self.add_more = add_more

    if self.add_more:
      #self.m *= 1.4
      pass

  def deriv(self, x, u):
    x = State(x)
    u = Control(u)

    accel = u.f / self.m
    xdd = -accel * np.sin(x.theta)
    zdd =  accel * np.cos(x.theta) - self.g
    tdd = u.tau / self.I

    if self.add_more:
      xdd -= 8.1 * x.x_vel
      #xdd += 3.1 * x.x_vel ** 2

      #xdd += 4 * x.theta
      #tdd -= x.theta_vel
      #tdd -= x.x_vel

    return np.array((x.x_vel, x.z_vel, x.theta_vel, xdd, zdd, tdd))

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
