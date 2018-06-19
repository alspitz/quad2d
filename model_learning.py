import numpy as np
import scipy.stats

from quad import Control, State

class LinearLearner:
  def __init__(self, n, model, dt):
    self.model = model
    self.dt = dt

    self.xs = []
    self.ys = []
    self.w = None

  def _get_x_in(self, x_t, u_t):
    #base = np.hstack((x_t, u_t))

    x_t = State(x_t)
    u_t = Control(u_t)
    #base = np.array((x_t.x, x_t.z, x_t.x_vel, x_t.z_vel))
    return np.array((u_t.f * np.sin(x_t.theta), u_t.f * np.cos(x_t.theta)))
    #return np.hstack((1, base, base**2))
    #return x_t
    #base = np.hstack((x_t))
    #return np.hstack((base))

  def get_deriv_u(self, x_t, u_t):
    x_t = State(x_t)
    u_t = Control(u_t) * self.model.m
    return self.w.T.dot(np.array((np.sin(x_t.theta), np.cos(x_t.theta))))

  def get_deriv_theta(self, x_t, u_t):
    x_t = State(x_t)
    u_t = Control(u_t) * self.model.m
    return self.w.T.dot(np.array((u_t.f * np.cos(x_t.theta), -u_t.f * np.sin(x_t.theta))))

  def get_dderiv_u2(self, x_t, u_t):
    return np.zeros(2)

  def get_dderiv_utheta(self, x_t, u_t):
    x_t = State(x_t)
    u_t = Control(u_t) * self.model.m
    return self.w.T.dot(np.array((np.cos(x_t.theta), -np.sin(x_t.theta))))

  def get_dderiv_theta2(self, x_t, u_t):
    x_t = State(x_t)
    u_t = Control(u_t) * self.model.m
    return self.w.T.dot(np.array((-u_t.f * np.sin(x_t.theta), -u_t.f * np.cos(x_t.theta))))

  def update(self, x_t, u_t, x_tp):
    xdot = self.model.deriv(x_t, u_t)
    model_prediction = self.dt * xdot# + 0.5 * self.dt**2 * self.model.dderiv(x_t, xdot, u_t)
    actual = x_tp - x_t

    diff = State(actual - model_prediction)

    regress_to = np.array((diff.x_vel, diff.z_vel))

    self.xs.append(self._get_x_in(x_t, u_t))
    self.ys.append(regress_to)

    #print("-" * 80)
    #print(np.linalg.norm(diff))
    #if self.w is not None:
    #  print(np.linalg.norm(diff - self.predict(x_t, u_t)))

  def compute(self):
    xs = np.array(self.xs)
    ys = np.array(self.ys)

    self.w = np.linalg.pinv(xs).dot(ys)

    #self.w, res, rank, s = np.linalg.lstsq(xs, ys)
    # TODO Figure out why the below hack is necessary.
    #self.w[np.abs(self.w) > 1] = 0
    #print(self.w)

    #self.w = np.zeros((17, 6))
    #self.w[4, 3] = -4.1 * self.dt
    #self.w[3, 3] = 4 * self.dt

    #self.w = np.zeros((9, 2))
    #self.w[1, 0] = -1.1 * self.dt
    #self.w[3, 0] = -8.1 * self.dt
    #self.w[4, 1] = -10.5 * self.dt
    #self.w[5, 0] = -0.8 * self.dt
    #self.w[7, 0] = -3.1 * self.dt

    print("Fit is", np.linalg.norm(xs.dot(self.w) - ys) / xs.shape[0])

    #import matplotlib.pyplot as pyplot
    #pyplot.figure()
    #pyplot.plot(xs[:, 1])
    #pyplot.figure()
    #pyplot.plot(ys[:, 1])
    #pyplot.show()

  def predict(self, x_t, u_t=None):
    return self._get_x_in(x_t, u_t * self.model.m).dot(self.w)

  def get_deriv_x(self, x_t, u_t=None):
    x_t = State(x_t)
    dzdx = np.array(((0, 0),
                     (1, 0),
                     (0, 1),
                     (0, 0),
                     (0, 0),
                     (2 * x_t.x, 0),
                     (0, 2 * x_t.z),
                     (0, 0),
                     (0, 0)))
    return self.w.T.dot(dzdx)

  def get_deriv_x_vel(self, x_t, u_t=None):
    x_t = State(x_t)
    dzdx_vel = np.array(((0, 0),
                         (0, 0),
                         (0, 0),
                         (1, 0),
                         (0, 1),
                         (0, 0),
                         (0, 0),
                         (2 * x_t.x_vel, 0),
                         (0, 2 * x_t.z_vel)))
    return self.w.T.dot(dzdx_vel)

  def get_dderiv_x_x(self, x_t):
    # This derivative is a tensor of rank 3.
    dx2 = np.zeros((9, 2, 2))
    dx2[5, 0, 0] = 2
    dx2[6, 1, 1] = 2
    return np.tensordot(self.w.T, dx2, axes=1)

  def get_dderiv_x_vel_x_vel(self, x_t):
    # This derivative is a tensor of rank 3.
    dx_vel2 = np.zeros((9, 2, 2))
    dx_vel2[7, 0, 0] = 2
    dx_vel2[8, 1, 1] = 2
    return np.tensordot(self.w.T, dx_vel2, axes=1)

  def clear(self):
    self.xs = []
    self.ys = []
