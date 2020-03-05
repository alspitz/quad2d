#import numpy as np
import autograd
import autograd.numpy as np
import scipy.stats

from quad import Control, State

class OrigFeats:
  def __init__(self, model):
    self.model = model
    self.n_features = 8

  def get_features_from_state_input(self, state_input):
    """ state_input is x, z, x vel, z vel, u, theta """
    return np.array((state_input[0], state_input[1], state_input[2], state_input[3],
                     np.sin(state_input[5]),
                     state_input[4] * np.sin(state_input[5]),
                     state_input[4] * np.cos(state_input[5]),
                     1))
    #ff = np.sqrt(2.0 / self.D) * np.cos(np.dot(self.omegas, state_input) + self.offsets)
    #return np.concatenate((state_input, ff))

  def get_x_in(self, x_t, u_t, t):
    state_input = np.array((x_t[0], x_t[1], x_t[3], x_t[4], u_t[0], x_t[2]))
    return self.get_features_from_state_input(state_input)
    #base = np.hstack((x_t, u_t))

    #x_t = State(x_t)
    #u_t = Control(u_t)
    ##base = np.array((x_t.x, x_t.z, x_t.x_vel, x_t.z_vel))
    #return np.array((x_t.x, x_t.z, x_t.x_vel, x_t.z_vel, np.sin(x_t.theta), u_t.f * np.sin(x_t.theta), u_t.f * np.cos(x_t.theta)))
    #return np.array((x_t[0], x_t[1], x_t[3], x_t[4], np.sin(x_t[2]), u_t[0] * np.sin(x_t[2]), u_t[0] * np.cos(x_t[2])))
    #return np.hstack((1, base, base**2))
    #return x_t
    #base = np.hstack((x_t))
    #return np.hstack((base))

  def get_derivs_state_input(self, state_input):
    state_input[4] *= self.model.m

    dzdsi = np.zeros((self.n_features, 6))
    dzdsi[:4, :4] = np.eye(4)

    # sin(theta)
    dzdsi[4, 5] = np.cos(state_input[5])

    # u sin(theta)
    dzdsi[5, 4] = np.sin(state_input[5])
    dzdsi[5, 5] = state_input[4] * np.cos(state_input[5])

    # u cos(theta)
    dzdsi[6, 4] = np.cos(state_input[5]) # u cos(theta)
    dzdsi[6, 5] = -state_input[4] * np.sin(state_input[5])

    #dzdstate_input_auto = self.get_dzdxu_f(state_input)
    #dzdstate_input_man = -np.sqrt(2.0 / self.D) * np.einsum("i,ij->ij", np.sin(np.dot(self.omegas, state_input) + self.offsets), self.omegas)
    #dzdstate_input_man = np.concatenate((np.eye(len(state_input)), dzdstate_input_man))

    return dzdsi

  def get_dderiv_state_input(self, state_input):
    state_input[4] *= self.model.m

    #d2zdstate_input2_auto = self.get_d2(state_input)

    #part1 = -np.sqrt(2.0 / self.D) * np.expand_dims(np.cos(np.dot(self.omegas, state_input) + self.offsets), axis=1) * self.omegas
    #d2zdstate_input2_man = np.einsum("ij,ik->ijk", part1, self.omegas)
    #d2zdstate_input2_man = np.concatenate((np.zeros((6, 6, 6)), d2zdstate_input2_man))

    #print(d2zdstate_input2_auto)
    #print(d2zdstate_input2_auto.shape)
    #print(d2zdstate_input2_man)
    #print(d2zdstate_input2_man.shape)
    #input()
    #assert np.allclose(d2zdstate_input2_auto, d2zdstate_input2_man)

    d2zdsi2 = np.zeros((self.n_features, 6, 6))
    d2zdsi2[4, 5, 5] = -np.sin(state_input[5])
    d2zdsi2[5, 4, 5] =  np.cos(state_input[5])
    d2zdsi2[5, 5, 4] =  np.cos(state_input[5])
    d2zdsi2[5, 5, 5] = -state_input[4] * np.sin(state_input[5])
    d2zdsi2[6, 4, 5] = -np.sin(state_input[5])
    d2zdsi2[6, 5, 4] = -np.sin(state_input[5])
    d2zdsi2[6, 5, 5] = -state_input[4] * np.cos(state_input[5])

    return d2zdsi2

class LinearLearner:
  def __init__(self, model, features, dt):
    self.model = model
    self.features = features(model)
    self.dt = dt

    self.xs = []
    self.ys = []
    self.w = None

    #self.get_dzdxu_f = autograd.elementwise_grad(self._get_x_in)
    self.get_dzdxu_f = autograd.jacobian(self.features.get_features_from_state_input)

    self.get_d2 = autograd.jacobian(self.get_dzdxu_f)

    self.D = 1080 # No. of random fourier features.
    self.omegas = np.random.normal(loc=0.0, scale=0.01, size=(self.D, 6))
    self.offsets = np.random.uniform(low=0.0, high=2 * np.pi, size=(self.D,))

  def get_deriv_time(self, t):
    return np.zeros(2)

  def get_dderiv_time(self, t):
    return np.zeros(2)

  def get_derivs_state_input(self, state_input):
    dzdstate_input = self.features.get_derivs_state_input(state_input)

    full_deriv = self.w.T.dot(dzdstate_input)
    return full_deriv[:, :4], full_deriv[:, 4:]

  def get_dderiv_state_input(self, state_input):
    d2zdstate_input2 = self.features.get_dderiv_state_input(state_input)

    full_dderiv = np.tensordot(self.w.T, d2zdstate_input2, axes=1)
    return full_dderiv

  #def get_deriv_state(self, x_t, u_t):
  #  # State is (x, z, x dot, z dot)
  #  dzdstate = np.array(((1, 0, 0, 0),
  #                       (0, 1, 0, 0),
  #                       (0, 0, 1, 0),
  #                       (0, 0, 0, 1),
  #                       (0, 0, 0, 0),
  #                       (0, 0, 0, 0),
  #                       (0, 0, 0, 0)))
  #  return self.w.T.dot(dzdstate)


  #def get_deriv_u(self, x_t, u_t):
  #  x_t = State(x_t)
  #  u_t = Control(u_t) * self.model.m
  #  return self.w.T.dot(np.array((0, 0, 0, 0, 0, np.sin(x_t.theta), np.cos(x_t.theta))))

  #def get_deriv_theta(self, x_t, u_t):
  #  x_t = State(x_t)
  #  u_t = Control(u_t) * self.model.m
  #  return self.w.T.dot(np.array((0, 0, 0, 0, np.cos(x_t.theta), u_t.f * np.cos(x_t.theta), -u_t.f * np.sin(x_t.theta))))

  #def get_dderiv_u2(self, x_t, u_t):
  #  return np.zeros(2)

  #def get_dderiv_utheta(self, x_t, u_t):
  #  x_t = State(x_t)
  #  u_t = Control(u_t) * self.model.m
  #  return self.w.T.dot(np.array((0, 0, 0, 0, 0, np.cos(x_t.theta), -np.sin(x_t.theta))))

  #def get_dderiv_theta2(self, x_t, u_t):
  #  x_t = State(x_t)
  #  u_t = Control(u_t) * self.model.m
  #  return self.w.T.dot(np.array((0, 0, 0, 0, -np.sin(x_t.theta), -u_t.f * np.sin(x_t.theta), -u_t.f * np.cos(x_t.theta))))

  def update(self, x_t, u_t, x_tp, t):
    xdot = self.model.deriv(x_t, u_t)
    #model_prediction = self.dt * xdot# + 0.5 * self.dt**2 * self.model.dderiv(x_t, xdot, u_t)
    model_prediction = xdot
    actual = (x_tp - x_t) / self.dt

    diff = State(actual - model_prediction)

    regress_to = np.array((diff.x_vel, diff.z_vel))
    #print("Prediction:", model_prediction)
    #print("Actual:", actual)
    #print(regress_to)
    #input()

    x_in = self.features.get_x_in(x_t, u_t, t)

    self.xs.append(x_in)
    self.ys.append(regress_to)

  def compute(self):
    xs = np.array(self.xs)
    ys = np.array(self.ys)

    #self.w = np.linalg.pinv(xs).dot(ys)

    self.w = np.linalg.pinv(xs.T.dot(xs) + 0.0000 * np.eye(xs.shape[1])).dot(xs.T.dot(ys))
    #self.w = np.zeros((8, 2))
    #self.w[0, 0] = -10.1
    #self.w[2, 0] = -3.1
    #self.w[4, 0] = 1.4

    #print(self.w)

    #self.w, res, rank, s = np.linalg.lstsq(xs, ys)
    # TODO Figure out why the below hack is necessary.
    #self.w[np.abs(self.w) > 1] = 0

    #self.w = np.zeros((17, 6))
    #self.w[4, 3] = -4.1 * self.dt
    #self.w[3, 3] = 4 * self.dt

    #self.w = np.zeros((9, 2))
    #self.w[1, 0] = -1.1 * self.dt
    #self.w[3, 0] = -8.1 * self.dt
    #self.w[4, 1] = -10.5 * self.dt
    #self.w[5, 0] = -0.8 * self.dt
    #self.w[7, 0] = -3.1 * self.dt

    #print(self.w)
    #print("Fit is", np.linalg.norm(xs.dot(self.w) - ys) / xs.shape[0])
    #self.w[0, :] = np.zeros(2)
    #self.w[1, 1] = 0
    #self.w[2, 0] = 0
    #print(self.w)
    #print("Fit is", np.linalg.norm(xs.dot(self.w) - ys) / xs.shape[0])

    #import matplotlib.pyplot as pyplot
    #pyplot.figure()
    #pyplot.plot(xs[:, 1])
    #pyplot.figure()
    #pyplot.plot(ys[:, 1])
    #pyplot.show()

  def predict(self, x_t, u_t=None, t=0):
    return self.features.get_x_in(x_t, u_t * self.model.m, t).dot(self.w)

  #def get_deriv_x(self, x_t, u_t=None):
  #  x_t = State(x_t)
  #  dzdx = np.array(((0, 0),
  #                   (1, 0),
  #                   (0, 1),
  #                   (0, 0),
  #                   (0, 0),
  #                   (2 * x_t.x, 0),
  #                   (0, 2 * x_t.z),
  #                   (0, 0),
  #                   (0, 0)))
  #  return self.w.T.dot(dzdx)

  #def get_deriv_x_vel(self, x_t, u_t=None):
  #  x_t = State(x_t)
  #  dzdx_vel = np.array(((0, 0),
  #                       (0, 0),
  #                       (0, 0),
  #                       (1, 0),
  #                       (0, 1),
  #                       (0, 0),
  #                       (0, 0),
  #                       (2 * x_t.x_vel, 0),
  #                       (0, 2 * x_t.z_vel)))
  #  return self.w.T.dot(dzdx_vel)

  #def get_dderiv_x_x(self, x_t):
  #  # This derivative is a tensor of rank 3.
  #  dx2 = np.zeros((9, 2, 2))
  #  dx2[5, 0, 0] = 2
  #  dx2[6, 1, 1] = 2
  #  return np.tensordot(self.w.T, dx2, axes=1)

  #def get_dderiv_x_vel_x_vel(self, x_t):
  #  # This derivative is a tensor of rank 3.
  #  dx_vel2 = np.zeros((9, 2, 2))
  #  dx_vel2[7, 0, 0] = 2
  #  dx_vel2[8, 1, 1] = 2
  #  return np.tensordot(self.w.T, dx_vel2, axes=1)

  def clear(self):
    self.xs = []
    self.ys = []

class TimeLearnerOld(LinearLearner):
  def _get_x_in(self, x_t, u_t, t):
    return np.array((1, t, t ** 2, t ** 3, t ** 4))

  def get_derivs_state_input(self, state_input):
    return np.zeros((2, 4)), np.zeros((2, 2))

  def get_dderiv_state_input(self, state_input):
    return np.zeros((2, 6, 6))

  def get_deriv_time(self, t):
    return self.w.T.dot(np.array((0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3)))

  def get_dderiv_time(self, t):
    return self.w.T.dot(np.array((0, 0, 2, 6 * t, 12 * t ** 2)))

class TimeLearner(LinearLearner):
  def __init__(self, model, dt):
    super().__init__(model, dt)

    self.D = 2080 # No. of random fourier features.
    self.omegas = np.random.normal(loc=0.0, scale=15.1, size=(self.D,))
    self.offsets = np.random.uniform(low=0.0, high=2 * np.pi, size=(self.D,))

  def _get_x_in(self, x_t, u_t, t):
    return np.sqrt(2 / self.D) * np.cos(self.omegas * t + self.offsets)

  def get_derivs_state_input(self, state_input):
    return np.zeros((2, 4)), np.zeros((2, 2))

  def get_dderiv_state_input(self, state_input):
    return np.zeros((2, 6, 6))

  def get_deriv_time(self, t):
    return self.w.T.dot(- np.sqrt(2 / self.D) * np.sin(self.omegas * t + self.offsets) * self.omegas)

  def get_dderiv_time(self, t):
    return self.w.T.dot(- np.sqrt(2 / self.D) * np.cos(self.omegas * t + self.offsets) * self.omegas * self.omegas)
