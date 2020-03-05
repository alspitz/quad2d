# coding: utf-8
from __future__ import print_function

import time

#import numpy as np
import autograd.numpy as np
import scipy.integrate

import model_learning

from controllers import FlatController, Observer
from model_learning import LinearLearner, TimeLearner
from plotters import DistPlotter, TrajPlotter, XPlotter, XZPlotter
from quad import Quad2DModel, D_X_CONSTANT, D_X_DRAG, D_Z_DRAG, D_MASS, D_ANGLE

from polynomial_utils import deriv_fitting_matrix

import matplotlib
# These are very important when submitting a paper!
# Prevent Type 3 fonts!
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = "Times New Roman"
import matplotlib.pyplot as plt

try:
  input = raw_input
except NameError:
  pass

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

poly_fit_mat = np.linalg.inv(deriv_fitting_matrix(8))

def get_poly(x, v=0, a=0, j=0, end_pos=1.0, duration=1.0):
  poly = poly_fit_mat.dot(np.array((x, v, a, j, end_pos, 0, 0, 0)))

  divider = 1
  for i in range(8):
    poly[i] /= divider
    divider *= duration

  return poly[::-1]

def plot(ts, data, ind, y_label=None, title=None, label="", ylim=None, symbol='-'):
  plt.figure(ind)
  plt.plot(ts, data, symbol, label=label)
  plt.xlabel('Time (s)')

  if y_label is not None:
    plt.ylabel(y_label)
  if title is not None:
    plt.title(title)
  if ylim is not None:
    plt.ylim(ylim)

  plt.legend()

def plot_des(ts, data, ind, label="Desired"):
  return plot(ts, data, ind, label=label, symbol='--')

def noisy(x):
  noise_pos = np.random.normal(0, 1e-3, size=(2,))
  noise_vel = np.random.normal(0, 1e-2, size=(2,))
  noisy_x = x.copy()
  noisy_x[:2] += noise_pos
  noisy_x[4:6] += noise_vel

  return noisy_x

def error(x1, x2):
  return np.mean(np.abs(x1 - x2))

def simulate(disturbances, flags, run_params):
  n_trials = run_params.n_trials
  prints = run_params.prints
  show_control_desires = run_params.show_control_desires
  save_figs = run_params.save_figs
  extension = run_params.extension

  plotter_class = run_params.plotter_class
  plot_dists = run_params.plot_dists

  print_trial_no = run_params.print_trial_no

  m = run_params.m
  g = run_params.g
  I = run_params.I

  dt = run_params.dt
  end_pos = run_params.end_pos
  t_end = run_params.t_end

  latex_s = ""

  for disturbance_set, dist_name, folder_name in disturbances:
    print("-" * 30, dist_name, "-" * 30)
    plotter = plotter_class(dist_name, t_end)
    for i_strat, flag_set in enumerate(flags):
      feedforward, learn, deriv_correct, feedback, num_opt, use_observer, name, plot_sym = flag_set

      if type(deriv_correct) == type(()):
        correct_snap = deriv_correct[1]
        deriv_correct = deriv_correct[0]
      else:
        correct_snap = True

      assert not learn or not use_observer

      print("*** Starting %s ***" % name)

      x_start = 0
      # Be careful with this non zero
      # With disturbances may require a non zero theta or omega.
      x_vel_start = 0

      #x_poly = [0, 0, 0, 0, 0]
      x_poly = get_poly(x_start, x_vel_start, end_pos=end_pos, duration=t_end)
      #z_poly = [0, 1, 1, 0, 0]
      #z_poly = [1, 1, -12, 0, 0]
      z_poly = [0, 0, 0, 0, 0]
      #z_poly = get_poly(0, 0)

      disturbance_flags = [False] * 5
      for dist in disturbance_set:
        disturbance_flags[dist] = True

      starting_theta = 0.0
      if disturbance_flags[D_ANGLE] and disturbance_flags[D_X_CONSTANT]:
        starting_theta = -0.43134372711283847
      elif disturbance_flags[D_X_CONSTANT]:
        starting_theta = -0.38287523900965065

      quad = Quad2DModel(m, g, I, disturbances=disturbance_flags)
      control_model = Quad2DModel(m, g, I)

      if learn == "time":
        learner = TimeLearner(control_model, dt)
      else:
        learner = LinearLearner(control_model, model_learning.OrigFeats, dt) if learn else None

      observer = Observer(dt) if use_observer else None

      controller = FlatController(control_model, x_poly, z_poly, learner, feedforward, deriv_correct, feedback, num_opt, observer=observer, correct_snap=correct_snap)
      closed_loop = lambda t, x: quad.deriv(x, controller.get_u(x, t))

      N = int(t_end / dt) + 1
      ts = np.linspace(0, t_end, N)
      x_des, z_des = np.array(controller.get_des(ts))

      if rviz:
        rviz_helper.publish_trajectory(x_des[0], z_des[0])

      z_start = z_des[0, 0]
      z_vel_start = z_des[1, 0]

      trials = n_trials if learn or use_observer else 1

      dist_plotter = DistPlotter(dist_name, strategy=name)

      for trial in range(trials):
        ts = [0]
        xs = [np.array((x_start, z_start, starting_theta, x_vel_start, z_vel_start, 0))]

        r = scipy.integrate.ode(closed_loop)
        r.set_initial_value(xs[0], ts[0])
        #r.set_integrator('dopri5', nsteps=1e9)
        r.set_integrator('lsoda', nsteps=1e9)

        theta_desires = []
        theta_vel_desires = []

        # Disturbances
        true_dists = []
        est_dists = []

        print_step = 0.0
        print_freq = 0.01

        #print("Computing controls without integrating...")
        if show_control_desires:
          for i in range(x_des.shape[1]):
            a_norm, theta_des, theta_vel_des, theta_acc_des = controller.get_des_data(x_des[:, i], z_des[:, i], np.zeros(2))
            theta_desires.append(theta_des)
            theta_vel_desires.append(theta_vel_des)

        #print("Starting integration...")

        #while ts[-1] < t_end:
        #  new_x = xs[-1] + closed_loop(ts[-1], xs[-1]) * dt
        #  learner.update(xs[-1], controller.get_u(xs[-1], ts[-1]), new_x)

        #  if prints and ts[-1] >= print_step:
        #    print("%0.2f / %0.2f\r" % (print_step, t_end), end="")
        #    print_step += print_freq

        #  xs.append(new_x)
        #  ts.append(ts[-1] + dt)

        x_pos_des = np.polyval(x_poly, 0)
        z_pos_des = np.polyval(z_poly, 0)
        x_pos_act = xs[-1][0]
        z_pos_act = xs[-1][1]
        angle_act = xs[-1][2]
        if rviz:
          rviz_helper.publish_marker(x_pos_des, z_pos_des)
          rviz_helper.publish_robot(x_pos_act, z_pos_act, angle_act)
          time.sleep(0.2)
          rviz_helper.publish_marker(x_pos_des, z_pos_des)
          rviz_helper.publish_robot(x_pos_act, z_pos_act, angle_act)

        if print_trial_no:
          print("\ttrial %d of %d..." % (trial + 1, trials))

        if realtime:
          input("Press ENTER to run traj...")

        time_start = time.time()

        while r.successful() and r.t < t_end - 1e-5:
          if rviz:
            x_pos_des = np.polyval(x_poly, r.t)
            z_pos_des = np.polyval(z_poly, r.t)
            rviz_helper.publish_marker(x_pos_des, z_pos_des)
            x_pos_act = xs[-1][0]
            z_pos_act = xs[-1][1]
            angle_act = xs[-1][2]
            rviz_helper.publish_robot(x_pos_act, z_pos_act, angle_act)

          t_now = r.t + dt
          x = r.integrate(t_now)

          if learn or use_observer:
            last_u = controller.get_u(xs[-1], r.t - dt)

            true_dists.append(quad.get_disturbance(xs[-1], last_u))

          if learn:
            learner.update(xs[-1], last_u, x, r.t - dt)

            if learner.w is not None:
              pred = learner.predict(xs[-1], last_u, r.t - dt)
              est_dists.append(pred)

          if use_observer:
            current_deriv = control_model.deriv(xs[-1], last_u)
            pred_accel = np.array((current_deriv[3], current_deriv[4]))
            observer.update(pred_accel, np.array((x[3], x[4])))

            est_dists.append(observer.dist_est.copy())

          if prints and r.t >= print_step:
            curr_error = error(np.array(xs)[:, 0], x_des[0, :len(xs)])
            print("%0.2f / %0.2f (x error is %1.2e)\r" % (print_step, t_end, curr_error), end="")
            print_step += print_freq

          xs.append(x)
          ts.append(t_now)

          if rviz and realtime:
            time_elapsed = (time.time() - time_start) * 0.5
            if r.t > time_elapsed:
              time.sleep(r.t - time_elapsed)

        if prints:
          print()

        if learn:
          learner.compute()
          if learn == "time":
            learner.clear()

        ts = np.array(ts)
        xs = np.array(xs)

        if len(est_dists) > 0:
          true_dists = np.array(true_dists)
          est_dists = np.array(est_dists)
          dist_plotter.add_run(ts, true_dists, est_dists, trial)

        x_vals = xs[:, 0]
        z_vals = xs[:, 1]
        v_vals = xs[:, 3]
        theta_vals = xs[:, 2]
        theta_vel_vals = xs[:, 5]

        #run_s = "Run %d" % trial
        #plot(ts, x_vals, 0, "$x$ (m)", "$x$ vs. time", run_s)
        #plot(ts, v_vals, 1, "$v$ (m/s)", "$\dot x$ vs. time", run_s)
        #plot(ts, z_vals, 2, "$z$ (m)", "$z$ vs. time", run_s)
        #plot(ts, theta_vals, 3, "$\\theta$ (rad)", "$\\theta$ vs. time", run_s)
        #plot(ts, theta_vel_vals, 4, "$\\omega$ (rad/s)", "$\\omega$ vs. time", run_s)
        #if show_control_desires:
          #plot_des(ts, theta_desires, 3, "Desired %d" % trial)
          #plot_des(ts, theta_vel_desires, 4, "Desired %d" % trial)

        x_error = np.abs(x_vals - x_des[0])
        z_error = np.abs(z_vals - z_des[0])

      plotter.add_data(ts, x_error, z_error, plot_sym, name, x_vals, z_vals, x_des[0], z_des[0])

      max_x_error = np.max(x_error)
      max_z_error = np.max(z_error)
      mean_x_error = np.mean(x_error)
      mean_z_error = np.mean(z_error)

      pos_error = np.sqrt(x_error ** 2 + z_error ** 2)
      mean_pos_error = np.mean(pos_error)
      max_pos_error = np.max(pos_error)

      #print("Run %d:" % trial)
      #print("\tmean errors x, z:", mean_x_error, mean_z_error)
      #print("\tmax  errors x, z:", max_x_error, max_z_error)
      print("\tmean/max pos: %0.3f, %0.3f" % (mean_pos_error, max_pos_error))

      if i_strat == 0:
        latex_s += "\hline \\textbf{%s} & " % dist_name

      threshold = 0.01 if feedback else 0.03
      if max_pos_error < threshold:
        num_s = "$\\mathbf{%0.3f}$" % max_pos_error
      else:
        num_s = "$%0.3f$" % max_pos_error

      latex_s += num_s
      if i_strat < len(flags) - 1:
        latex_s += " & "
      else:
        latex_s += " \\\\\n"

      if plot_dists:
        dist_plotter.plot()

      #plt.show()

      #print("x error is %1.2e" % error(x_vals, x_des[0]))
      #print("z error is %1.2e" % error(z_vals, z_des[0]))
      #print("v error is %1.2e" % error(v_vals, x_des[1]))
      #if show_control_desires:
        #print("θ error is %1.2e" % error(theta_vals, theta_desires))
        #print("θ vel error is %1.2e" % error(theta_vel_vals, theta_vel_desires))

      #plot_des(ts, x_des[0], 0)
      #plot_des(ts, x_des[1], 1)
      #plot_des(ts, z_des[0], 2)
      #if show_control_desires:
        #plot_des(ts, theta_desires, 3)
        #plot_des(ts, theta_vel_desires, 4)

    fs = "ff" if not feedback else "fb"
    plotter.plot(save_fig=save_figs, filename='media/%s/%d.%s' % (folder_name, i_strat, extension))
    #plotter.plot(save_fig=save_figs, filename='media/%s_%s.%s' % (dist_name, fs, extension))

  print(latex_s)
  plt.show()

class Params:
  plot_dists = True
  dt = 0.005
  prints = False
  print_trial_no = True
  save_figs = False

  m = 4.19
  g = 10.18
  I = 0.123

  end_pos = 1.0
  t_end = 1.0

def paper_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "pdf"
  run_params.plotter_class = XZPlotter
  run_params.plot_dists = False
  run_params.print_trial_no = False
  run_params.dt = 0.001

  feedback = True

  disturbances = [
    ((D_X_CONSTANT,), "A", "A"),
    ((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "B", "B"),
    ((D_ANGLE, D_MASS), "C", "C"),
    ((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "D", "D")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observ. label   line style
    (True, False, False, feedback, False, False, "FF1", 'r-'),
    (True, True,  False, feedback, False, False, "FF2", 'r--'),
    (True, True,  False, feedback, True,  False, "FF3", 'k-.'),
    (True, True,  True,  feedback, False, False, "FF4", 'k--'),
    (True, True,  True,  feedback, True,  False, "FF5", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

def camera_ready_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = True
  run_params.extension = "pdf"
  run_params.plotter_class = XZPlotter
  run_params.plot_dists = False
  run_params.print_trial_no = False
  run_params.dt = 0.001

  feedback = True

  disturbances = [
    #((D_X_CONSTANT,), "A", "A"),
    #((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "B", "B"),
    #((D_ANGLE, D_MASS), "C", "C"),
    ((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "D", "D")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observ. label   line style
    (True, False, False, feedback, False, False, "FF1", 'k-'),
    (True, True,  False, feedback, False, False, "FF2", 'r--'),
    (True, True,  False, feedback, True,  False, "FF3", 'r-'),
    (True, True,  True,  feedback, False, False, "FF4", 'b--'),
    (True, True,  True,  feedback, True,  False, "FF5", 'b-'),
  ]

  simulate(disturbances, flags, run_params)

def poster_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = True
  run_params.extension = "pdf"
  run_params.plotter_class = XZPlotter
  run_params.plot_dists = False
  run_params.print_trial_no = False
  run_params.dt = 0.001

  feedback = False

  disturbances = [
    ((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "Drag", "Drag"),
    ((D_ANGLE, D_MASS), "Mass & Angle", "Mass & Angle"),
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observ. label   line style
    (True, False, False, feedback, False, False, "No learning", 'k-'),
    (True, True,  False, feedback, False, False, "Basic Comp.", 'r--'),
    (True, True,  False, feedback, True,  False, "Basic Comp. + Accel Balance", 'r-'),
    (True, True,  True,  feedback, False, False, "Disturbance Dynamics Compensation", 'b--'),
    (True, True,  True,  feedback, True,  False, "Disturbance Dynamics Compensatio + Accel Balance", 'b-'),
  ]

  simulate(disturbances, flags, run_params)

def talk_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "png"
  run_params.plot_dists = False
  run_params.plotter_class = XPlotter

  run_params.end_pos = 4.0
  run_params.t_end = 2.0

  feedback = True

  disturbances = [
    #((D_X_CONSTANT,), "Constant Disturbance", "constant"),
    ((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "Drag", "drag"),
    #((D_ANGLE, D_MASS), "Angle & Mass Disturbance", "angle-mass"),
    #((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "Drag, Angle, & Mass Disturbance", "all")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observer label   line style
    #(True, False, False, feedback, False, False,   "No compensation", 'r-'),
    (True, True,  False, feedback, False, False,   "Acceleration compensation", 'r--'),
    #(True, True,  False, feedback, True,  False,   "Acceleration optimization", 'k-.'),
    (True, True,  True,  feedback, False, False,   "Disturbance dynamics compensation", 'k--'),
    #(True, True,  True,  feedback, True,  False,   "Acceleration optimization w/ disturbance dynamics compensation", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

def extra_baselines():
  run_params = Params()
  run_params.n_trials = 4
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "png"
  run_params.plotter_class = XZPlotter

  feedback = True

  disturbances = [
    ((D_X_CONSTANT,), "Constant Disturbance", "constant"),
    ((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "Drag", "drag"),
    ((D_ANGLE, D_MASS), "Angle & Mass Disturbance", "angle-mass"),
    #((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "Drag, Angle, & Mass Disturbance", "all")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observer  label   line style
    (True, False, False, feedback, False, False,    "No compensation", 'r-'),
    (True, True,  False, feedback, False, False,    "Basic compensation", 'r--'),
    (True, False, False, feedback, False, True,     "Observer", 'b-'),
    (True, True,  True,  feedback, False, False,    "Dynamics compensation", 'k--'),
    (True, True,  True,  feedback, True,  False,    "Optimization w/ dynamics compensation", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

def ilc_testing():
  run_params = Params()
  run_params.n_trials = 5
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "png"
  run_params.plotter_class = XZPlotter

  feedback = True

  disturbances = [
    ((D_X_CONSTANT,), "Constant Disturbance", "constant"),
    ((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "Drag", "drag"),
    ((D_ANGLE, D_MASS), "Angle & Mass Disturbance", "angle-mass"),
    ((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "Drag, Angle, & Mass Disturbance", "all")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observer  label   line style
    (True, False, False, feedback, False, False,    "No compensation", 'r-'),
    (True, True,  False, feedback, False, False,    "Basic compensation", 'r--'),
    (True, "time",  True, feedback, False, False,     "ILC compensation", 'b-'),
    (True, True,  True,  feedback, False, False,    "Dynamics compensation", 'k--'),
    (True, True,  True,  feedback, True,  False,    "Optimization w/ dynamics compensation", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

def may_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "pdf"
  run_params.plotter_class = XZPlotter
  run_params.plot_dists = True
  run_params.dt = 0.001

  feedback = False

  disturbances = [
    ((D_X_CONSTANT,), "A", "A"),
    #((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "B", "B"),
    #((D_ANGLE, D_MASS), "C", "C"),
    #((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "D", "D")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observ. label   line style
    (True, False, False, feedback, False, False, "FF1", 'r-'),
    (True, True,  False, feedback, False, False, "FF2", 'r--'),
    #(True, True,  False, feedback, True,  False, "FF3", 'k-.'),
    (True, True,  True,  feedback, False, False, "FF4", 'k--'),
    (True, True,  True,  feedback, True,  False, "FF5", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

def rss_video_results():
  run_params = Params()
  run_params.n_trials = 3
  run_params.show_control_desires = False
  run_params.prints = False
  run_params.save_figs = False
  run_params.extension = "pdf"
  run_params.plotter_class = TrajPlotter
  run_params.plot_dists = False
  run_params.dt = 0.001

  feedback = False

  disturbances = [
    #((D_X_CONSTANT,), "A", "A"),
    #((D_X_CONSTANT, D_X_DRAG, D_Z_DRAG), "Drag", "Drag"),
    ((D_ANGLE, D_MASS), "C", "C"),
    #((D_X_CONSTANT, D_X_DRAG, D_ANGLE, D_Z_DRAG, D_MASS), "D", "D")
  ]

  flags = [
     # ff  learn  deriv     fb      opt   observ. label   line style
    (True, False, False, feedback, False, False, "No Learning", "#ff0000"),
    (True, True,  False, feedback, True, False, "Order 0", "#0000ff"),
    #(True, True,  False, feedback, True,  False, "FF3", 'k-.'),
    (True, True,  (True, False),  feedback, True, False, "Order 1", "#af00ff"),
    (True, True,  (True, True),  feedback, True, False, "Order 2", "#00bb00"),
    #(True, True,  True,  feedback, True,  False, "FF5", 'k-'),
  ]

  simulate(disturbances, flags, run_params)

if __name__ == "__main__":
  rviz = False
  realtime = False
  if rviz:
    from rviz_helper import RVIZHelper
    rviz_helper = RVIZHelper()

  #may_results()
  #rss_video_results()
  #talk_results()
  poster_results()
