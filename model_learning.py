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
    self.xvel_input_ind = 4
    #base = np.hstack((x_t, u_t))

    x_t = State(x_t)
    base = np.array((x_t.x, x_t.z, x_t.x_vel, x_t.z_vel))
    return np.hstack((1, base, base**2))
    #return x_t
    #base = np.hstack((x_t))
    #return np.hstack((base))

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

    print("Fit is", np.linalg.norm(xs.dot(self.w) - ys) / xs.shape[0])

    #import matplotlib.pyplot as pyplot
    #pyplot.figure()
    #pyplot.plot(xs[:, 1])
    #pyplot.figure()
    #pyplot.plot(ys[:, 1])
    #pyplot.show()

  def predict(self, x_t, u_t=None):
    return self._get_x_in(x_t, u_t).dot(self.w)

  def clear(self):
    self.xs = []
    self.ys = []
