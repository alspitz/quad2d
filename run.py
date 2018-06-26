import numpy as np
import scipy.integrate

from controllers import FlatController
from model_learning import LinearLearner
from quad import Quad2DModel

def get_poly(x, v=0, a=0, j=0):
  poly_mat = np.array(((1, 0, 0, 0, 0, 0, 0, 0),
                       (0, 1, 0, 0, 0, 0, 0, 0),
                       (0, 0, 2, 0, 0, 0, 0, 0),
                       (0, 0, 0, 6, 0, 0, 0, 0),
                       (1, 1, 1, 1, 1, 1, 1, 1),
                       (0, 1, 2, 3, 4, 5, 6, 7),
                       (0, 0, 2, 6, 12, 20, 30, 42),
                       (0, 0, 0, 6, 24, 60, 120, 210)))
  poly_b = np.array((x, v, a, j, 1, 0, 0, 0))
  return np.flip(np.linalg.solve(poly_mat, poly_b), axis=0)

def plot(ts, data, ind, y_label=None, title=None, label="", ylim=None, symbol='-'):
  pyplot.figure(ind)
  pyplot.plot(ts, data, symbol, label=label)
  pyplot.xlabel('Time (s)')

  if y_label is not None:
    pyplot.ylabel(y_label)
  if title is not None:
    pyplot.title(title)
  if ylim is not None:
    pyplot.ylim(ylim)

  pyplot.legend()

def plot_des(ts, data, ind, label="Desired"):
  return plot(ts, data, ind, label=label, symbol='--')

def noisy(x):
  return x + np.random.normal(0, 1e-2, size=(6,))

def error(x1, x2):
  return np.mean(np.abs(x1 - x2))

if __name__ == "__main__":
  import matplotlib.pyplot as pyplot

  trials = 2

  m = 4.19
  g = 10.18
  I = 0.123

  x_start = 0
  # Be careful with this non zero
  # With disturbances may require a non zero theta or omega.
  x_vel_start = 0

  quad = Quad2DModel(m, g, I, add_more=True)
  control_model = Quad2DModel(m, g, I)

  dt = 0.001
  learner = LinearLearner(1, control_model, dt)

  x_poly = get_poly(x_start, x_vel_start)
  z_poly = x_poly
  #z_poly = [0, 1, 1, 0, 0]
  #z_poly = [1, 1, -12, 0, 0]
  #z_poly = [0, 0, 0, 0, 0]

  controller = FlatController(control_model, x_poly, z_poly, learner)
  closed_loop = lambda t, x: quad.deriv(x, controller.get_u(x, t))

  t_end = 1.0
  ts = np.linspace(0, t_end, 1001)
  x_des, z_des = np.array(controller.get_des(ts))

  z_start = z_des[0, 0]
  z_vel_start = z_des[1, 0]

  for trial in range(trials):
    ts = [0]
    xs = [np.array((x_start, z_start, 0, x_vel_start, z_vel_start, 0))]

    r = scipy.integrate.ode(closed_loop)
    r.set_initial_value(xs[0], ts[0])
    #r.set_integrator('dopri5', nsteps=1e9)

    theta_desires = []
    theta_vel_desires = []

    for i in range(x_des.shape[1]):
      a_norm, theta_des, theta_vel_des, theta_acc_des = controller.get_des_data(x_des[:, i], z_des[:, i])
      theta_desires.append(theta_des)
      theta_vel_desires.append(theta_vel_des)

    while r.successful() and r.t < t_end:
      t_now = r.t + dt
      x = r.integrate(t_now)

      learner.update(xs[-1], controller.get_u(xs[-1], r.t - dt), x)

      xs.append(x)
      ts.append(t_now)

    learner.compute()
    #learner.clear()

    ts = np.array(ts)
    xs = np.array(xs)

    x_vals = xs[:, 0]
    z_vals = xs[:, 1]
    v_vals = xs[:, 3]
    theta_vals = xs[:, 2]
    theta_vel_vals = xs[:, 5]

    run_s = "Run %d" % trial
    plot(ts, x_vals, 0, "$x$ (m)", "$x$ vs. time", run_s)
    plot(ts, v_vals, 1, "$v$ (m/s)", "$\dot x$ vs. time", run_s)
    plot(ts, z_vals, 2, "$z$ (m)", "$z$ vs. time", run_s)
    plot(ts, theta_vals, 3, "$\\theta$ (rad)", "$\\theta$ vs. time", run_s)
    plot(ts, theta_vel_vals, 4, "$\\omega$ (rad/s)", "$\\omega$ vs. time", run_s)
    plot_des(ts, theta_desires, 3, "Desired %d" % trial)
    plot_des(ts, theta_vel_desires, 4, "Desired %d" % trial)

    print("Run %d:" % trial)
    print("x error is %1.2e" % error(x_vals, x_des[0]))
    print("z error is %1.2e" % error(z_vals, z_des[0]))
    print("v error is %1.2e" % error(v_vals, x_des[1]))
    print("θ error is %1.2e" % error(theta_vals, theta_desires))
    print("θ vel error is %1.2e" % error(theta_vel_vals, theta_vel_desires))

  plot_des(ts, x_des[0], 0)
  plot_des(ts, x_des[1], 1)
  plot_des(ts, z_des[0], 2)
  #plot_des(ts, theta_desires, 3)
  #plot_des(ts, theta_vel_desires, 4)

  pyplot.show()
