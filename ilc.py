import argparse

import numpy as np

from collections import deque

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import math_utils

from run import get_poly

g = 9.81

def simulate_3d(t_end, fun, dt):
  pos = np.zeros(3)
  vel = np.zeros(3)
  rpy = np.zeros(3)
  rot = Rotation.from_euler('ZYX', rpy[::-1])
  quat = rot.as_quat()
  ang_body = np.zeros(3)
  ang_world = rot.apply(ang_body)

  x = np.zeros(12)
  xs = [x.copy()]

  for i in range(int(round(t_end / dt)) - 1):
    u_out = fun(x)

    u = u_out[0]
    ang_accel_body = np.array((u_out[1], u_out[2], u_out[3]))
    ang_accel_world = rot.apply(ang_accel_body)

    acc = u * rot.apply(np.array((0, 0, 1))) - np.array((0, 0, g))

    ang_world += ang_accel_world * dt

    quat_deriv = math_utils.quat_mult(math_utils.vector_quat(ang_world), quat) / 2.0
    quat += quat_deriv * dt
    quat /= np.linalg.norm(quat)

    vel += acc * dt
    pos += vel * dt

    rot = Rotation.from_quat(np.array((quat[1], quat[2], quat[3], quat[0])))
    rpy = rot.as_euler('ZYX')[::-1]
    ang_body = rot.inv().apply(ang_world)

    x = np.hstack((pos.copy(), vel.copy(), rpy.copy(), ang_body.copy()))

    xs.append(x)

  return np.array(xs)

class ILCBase:
  def __init__(self, feedback):
    self.use_feedback = feedback

class Trivial(ILCBase):
  """
    state is vel
    control is acc

    x_{t+1} = x_t + dt*(u_t - k_vel (x_t - x_td))
            = x_t + dt*(u_t - k_vel*x_t + k_vel*x_td)
            = x_t + dt*u_t - dt*k_vel*x_t + dt*k_vel*x_td
            = (1 - dt*k_vel)*x_t + dt*u_t + dt*k_vel*x_td

    y_t = u_t - k_vel(x_t - x_td)
        = u_t - k_vel*x_t + k_vel*x_td
        = -k_vel*x_t + u_t + k_vel*x_td

    dx_{t+1}/dx = A = 1 - dt*k_vel
    dx_{t+1}/du = B = dt

    dy/dx = C = -k_vel
    dy/du = D = 1

    x_{t+1} = Ax + Bu + dt*k_vel*x_td
    y = Cx + Du + k_vel*x_td

    y_{t+1} = C(Ax + Bu_t + dt*k_vel*x_td) + Du_{t+1} + k_vel*x_td
            = C((1 - dt*k_vel)*x + dt*u_t + dt*k_vel*x_td) + u_{t+1} + k_vel*x_td
            = C(x - dt*k_vel*x + dt*u_t + dt*k_vel*x_td) + u_{t+1} + k_vel*x_td
           ?= C(x + dt*u_t) + u_{t+1} + k_vel*x_td
            = -k_vel*x - k_vel*dt*u_t + u_{t+1} + k_vel*x_td
           ?= -k_vel*dt*u_t + u_{t+1}

    # Not considering affine terms...
    y_{t+1} = C(Ax + Bu_t) + Du_{t+1}
            = C((1 - dt*k_vel)*x + dt*u_t) + u_{t+1}
            = C(x - dt*k_vel*x + dt*u_t) + u_{t+1}
            = -k_vel*x + dt*k_vel*k_vel*x - dt*k_vel*u_t + u_{t+1}
            = -k_vel*dt*u_t + u_{t+1}

    y_{t+1} = CAx + CBu_t + Du_{t+1}
  """
  n_state = 1
  n_control = 1
  n_out = 1

  k_vel = 5

  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.k_vel = 0

    A = np.array(( (1 - dt * self.k_vel,), ))
    B = np.array(( (dt,), ))
    C = np.array(( (-self.k_vel,), ))
    D = np.array(( (1,), ))
    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    vel = x = np.zeros(1)
    xs = [x.copy()]
    for i in range(int(round(t_end / dt))):
      acc = fun(x)
      vel += acc * dt
      x = vel.copy()
      xs.append(x)
    return np.array(xs)

  def feedback(self, x, vel_des, acc_des):
    return np.array( -self.k_vel * (x - vel_des) + acc_des,)

class One(ILCBase):
  """
    state is (pos, vel)
    control is (acc)
  """
  n_state = 2
  n_control = 1
  n_out = 1

  k_pos = 5
  k_vel = 5

  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.k_pos = 0
      self.k_vel = 0

    A = np.array(( (1, dt), (-self.k_pos * dt, 1 - self.k_vel * dt,), ))
    B = np.array(( (0,), (dt,), ))
    C = np.array(( (-self.k_pos, -self.k_vel,), ))
    D = np.array(( (1,), ))

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel = x = np.zeros(2)
    xs = [x.copy()]
    for i in range(int(round(t_end / dt))):
      acc = fun(x)
      pos += vel * dt
      vel += acc * dt
      x = np.hstack((pos, vel))
      xs.append(x)
    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, acc_des):
    K_pos = np.array((self.k_pos, self.k_vel))
    pos_vel_des = np.hstack((pos_des, vel_des))
    return -K_pos.dot(x - pos_vel_des) + acc_des

class QuadLin(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (angular acceleration)

    om_{t+1} = om_{t} + dt*{-K_att0 * (theta - theta_des) - K_att1 * (om - om_des)}
  """
  n_state = 4
  n_control = 1
  n_out = 1

  K_pos = np.array((60, 35))
  K_att = np.array((200, 60))

  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.K_pos *= 0
      self.K_att *= 0

    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    A[self.n_state - 1, 0] =    -dt * self.K_att[0] * self.K_pos[0]
    A[self.n_state - 1, 1] =    -dt * self.K_att[0] * self.K_pos[1]
    A[self.n_state - 1, 2] =    -dt * self.K_att[0]
    A[self.n_state - 1, 3] = 1 - dt * self.K_att[1]

    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (0, 0, 1, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel, acc, jerk = x = np.zeros(4)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      snap = u_out[0]

      pos += vel * dt
      vel += acc * dt
      acc += jerk * dt
      jerk += snap * dt

      x = np.hstack((pos.copy(), vel.copy(), acc.copy(), jerk.copy()))
      xs.append(x)

    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, angaccel_des):
    pos_vel = x[:2]
    theta = x[2]
    angvel = x[3]

    accel_des = -self.K_pos.dot(pos_vel - np.hstack((pos_des, vel_des))) + acc_des
    theta_des = accel_des

    theta_err = theta - theta_des
    angvel_error = angvel - angvel_des
    u_ang_accel = -self.K_att.dot(np.hstack((theta_err, angvel_error))) + angaccel_des

    return np.array((u_ang_accel,))

class QuadLinPos(QuadLin):
  """ The output we want to track here is position instead of acceleration. """
  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.K_pos *= 0
      self.K_att *= 0

    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    A[self.n_state - 1, 0] =    -dt * self.K_att[0] * self.K_pos[0]
    A[self.n_state - 1, 1] =    -dt * self.K_att[0] * self.K_pos[1]
    A[self.n_state - 1, 2] =    -dt * self.K_att[0]
    A[self.n_state - 1, 3] = 1 - dt * self.K_att[1]

    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (1, 0, 0, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D

class NL1D(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (angular acceleration)
  """
  n_state = 4
  n_control = 1
  n_out = 1

  c = 100

  def get_ABCD(self, state, control, dt):
    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    A[1, 2] = np.cos(state[2] / self.c) * dt
    A[self.n_state - 1, 3] = 1

    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (1, 0, 0, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel, theta, jerk = x = np.zeros(4)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      snap = u_out[0]

      acc = self.c * np.sin(theta / self.c)

      pos += vel * dt
      vel += acc * dt
      theta += jerk * dt
      jerk += snap * dt

      x = np.hstack((pos.copy(), vel.copy(), theta.copy(), jerk.copy()))
      xs.append(x)

    return np.array(xs)

class Quad2DLin(ILCBase):
  """
    Same as Quad2D but no sines and cosines
  """
  n_state = 6
  n_control = 2
  n_out = 2

  def get_ABCD(self, state, control, dt):
    X = slice(0, 2)
    V = slice(2, 4)
    TH = slice(4, 5)
    OM = slice(5, 6)
    U = slice(0, 1)
    AA = slice(1, 2)

    theta = state[TH][0]
    u = control[U][0]

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(2)
    A[V, V] = np.eye(2)
    A[X, V] = dt * np.eye(2)
    A[V, TH] = u * dt * np.array(((-1, 0),)).T
    A[TH, TH] = A[OM, OM] = 1
    A[TH, OM] = dt

    B[V, U] = dt * np.array(((-theta, 1),)).T
    B[OM, AA] = dt

    C[X, X] = np.eye(2)

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos = np.zeros(2)
    vel = np.zeros(2)
    theta = 0
    angvel = 0

    x = np.zeros(6)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      u, angaccel = u_out

      acc = u * np.array((-theta, 1)) - np.array((0, g))

      pos += vel * dt
      vel += acc * dt
      theta += angvel * dt
      angvel += angaccel * dt

      x = np.hstack((pos.copy(), vel.copy(), theta, angvel))
      xs.append(x)

    return np.array(xs)

class Quad2D(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (u, angular acceleration)
  """
  n_state = 6
  n_control = 2
  n_out = 2

  K_pos = np.array((
    (8, 0, 16, 0),
    (0, 8, 0, 4)
  )) / 4
  K_att = np.array((200, 60))

  def get_ABCD(self, state, control, dt):
    X = slice(0, 2)
    V = slice(2, 4)
    TH = slice(4, 5)
    OM = slice(5, 6)
    U = slice(0, 1)
    AA = slice(1, 2)

    theta = state[TH][0]
    u = control[U][0]

    ct = np.cos(theta)
    st = np.sin(theta)

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(2)
    A[V, V] = np.eye(2)
    A[X, V] = dt * np.eye(2)
    A[V, TH] = u * dt * np.array(((-ct, -st),)).T
    A[TH, TH] = A[OM, OM] = 1
    A[TH, OM] = dt

    B[V, U] = dt * np.array(((-st, ct),)).T
    B[OM, AA] = dt

    C[X, X] = np.eye(2)

    if self.use_feedback:
      K_x = np.zeros((self.n_control, self.n_state))
      K_u = np.zeros((self.n_control, self.n_control))

      pos_vel = state[:4]
      a = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + np.array((0, g))
      adota = a.T.dot(a)
      adir = a / np.sqrt(adota)

      K_x[U, X] = adir.dot(-self.K_pos[:, X])
      K_x[U, V] = adir.dot(-self.K_pos[:, V])

      K_x[AA, X] = self.K_att[0] * (a[1] * self.K_pos[0, X] - a[0] * self.K_pos[1, X]) / adota
      K_x[AA, V] = self.K_att[0] * (a[1] * self.K_pos[0, V] - a[0] * self.K_pos[1, V]) / adota
      K_x[AA, TH] = -self.K_att[0]
      K_x[AA, OM] = -self.K_att[1]

      K_u[U, U] = 1
      K_u[AA, AA] = 1

      A = A + B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, u_ff, angaccel_des):
    pos_vel = x[:4]
    theta = x[4]
    angvel = x[5]

    accel_des = -self.K_pos.dot(pos_vel - np.hstack((pos_des, vel_des))) + acc_des + np.array((0, g))
    a_norm = np.linalg.norm(accel_des)
    z_axis_des = accel_des / a_norm
    theta_des = np.arctan2(z_axis_des[1], z_axis_des[0]) - np.pi / 2

    theta_err = theta - theta_des
    angvel_error = angvel - angvel_des
    u_ang_accel = -self.K_att.dot(np.hstack((theta_err, angvel_error))) + angaccel_des

    return np.hstack((a_norm + u_ff - g, u_ang_accel,))

  def simulate(self, t_end, fun, dt):
    pos = np.zeros(2)
    vel = np.zeros(2)
    theta = 0
    angvel = 0

    x = np.zeros(6)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      u, angaccel = u_out

      st = np.sin(theta)
      ct = np.cos(theta)

      acc = u * np.array((-st, ct)) - np.array((0, g))

      pos += vel * dt
      vel += acc * dt
      theta += angvel * dt
      angvel += angaccel * dt

      x = np.hstack((pos.copy(), vel.copy(), theta, angvel))
      xs.append(x)

    return np.array(xs)

class Quad3D:
  """
    state is (z axis, angular velocity)
    control is (u, angular acceleration)

    z_next = z + dt * [angvel]_x[:, 3]
    angvel_next = omega + dt * angular acceleration

    obs = accel = u * z axis + g
  """
  n_state = 6
  n_control = 4
  n_out = 3

  def get_ABCD(self, state, control, dt):
    z_axis, angvel = state[:3], state[3:]
    u, angaccel = control[0], control[1:]

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    # d z_next / d z
    A[0:3, 0:3] = np.eye(3)
    # d z_next / d angvel
    A[0:3, 3:6] = dt * np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 0)))
    # d angvel_next / d angvel
    A[3:6, 3:6] = np.eye(3)

    # d z_next / d angular acceleration
    #B[0:3, 1:4] = 0.5 * dt * dt * np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 0)))
    # TODO How to do this properly? (z's dependence on ang accel?)

    # d angvel_next / d angular acceleration
    B[3:6, 1:4] = dt * np.eye(3)

    # d accel / d z
    C[:3, :3] = u * np.eye(3)
    # d accel / d u
    D[:3, 0] = z_axis

    return A, B, C, D

class QuadILC:
  """
    state is (acceleration, z axis, angular velocity)
    control is (u, torque)
  """
  n_state = 9
  n_control = 4

  def __init__(self):
    self.g_vec = np.array((0, 0, -g))

  def con_state(self, acceleration, z_axis, angvel):
    assert np.isclose(np.linalg.norm(z_axis), 1.0)
    return np.hstack((acceleration, z_axis, angvel))
  def decon_state(self, state):
    return state[:3], state[3:6], state[6:]

  def con_control(self, u, torque):
    return np.hstack((u, torque))
  def decon_control(self, control):
    return control[0], control[1:]

  def get_next(self, state, control, dt):
    accel, z_axis, angvel = self.decon_state(state)
    u, torque = self.decon_control(control)

    accel_next = u * z_axis + self.g_vec
    z_axis_next = z_axis + dt * math_utils.skew_matrix(angvel)[:, 2]
    z_axis_next /= np.linalg.norm(z_axis_next)
    angvel_next = angvel + dt * torque

    return self.con_state(accel_next, z_axis_next, angvel_next)

  def get_AB(self, state, control, dt):
    accel, z_axis, angvel = self.decon_state(state)
    u, torque = self.decon_control(control)

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))

    A[:3, 3:6] = u * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[3:6, 6:9] = dt * np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 0)))
    A[6:9, 6:9] = np.eye(3)

    B[:3, 0] = z_axis
    B[6:9, 1:4] = dt * np.eye(3)

    return A, B

def cascaded_response(x, pos_des, vel_des, acc_des, angvel_des, angaccel_des):
  K_pos = 0 * np.array((
    (8.1, 0, 0, 4.05, 0, 0),
    (0, 8.1, 0, 0, 4.05, 0),
    (0, 0, 4.3, 0, 0, 2.0)
  ))

  K_att = 0 * np.array((
    (100, 0, 0, 15, 0, 0),
    (0, 100, 0, 0, 15, 0),
    (0, 0, 60, 0, 0, 12)
  ))

  pos_vel = x[:6]
  rpy = x[6:9]
  ang_vel = x[9:]

  pos_vel_error = pos_vel - np.hstack((pos_des, vel_des))

  # Position Control
  accel_des = -K_pos.dot(pos_vel_error) + acc_des

  # Reference Conversion
  euler_des = accel_to_euler_rpy(accel_des)
  rot = Rotation.from_euler('ZYX', rpy[::-1])
  #u_accel = rot.inv().apply(accel_des)[2]
  u_accel = np.linalg.norm(accel_des)

  # Attitude Control
  euler_error = rpy - euler_des
  angvel_error = ang_vel - angvel_des
  euler_angvel = np.hstack((euler_error, angvel_error))
  u_ang_accel = -K_att.dot(euler_angvel) + angaccel_des

  return np.hstack((u_accel, u_ang_accel))

if __name__  == "__main__":
  from lqr_gain_match.match_full_state import accel_to_euler_rpy
  import matplotlib.pyplot as plt

  system_map = {
                # ilc, DIMS
    'trivial': (Trivial, 1),
    'simple': (One, 1),
    'linear': (QuadLin, 1),
    'linearpos': (QuadLinPos, 1),
    'nl1d' : (NL1D, 1),
    '2dposlin': (Quad2DLin, 2),
    '2dpos':     (Quad2D, 2),
    '3d':     (Quad3D, simulate_3d, cascaded_response, 3),
  }

  parser = argparse.ArgumentParser()
  parser.add_argument("--system", type=str, default=1, choices=system_map.keys(), help="Type of system to simulate.")
  parser.add_argument("--trials", type=int, default=4, help="Number of ILC trials to run.")
  parser.add_argument("--alpha", type=float, default=1.0, help="Percentage of update (0 - 1) to use at each iteration. Lower values increase stability.")
  parser.add_argument("--dt", type=float, default=0.02, help="Size of timestep along trajectory.")
  parser.add_argument("--feedback", default=False, action='store_true', help="Apply feedback along the trajectory.")
  parser.add_argument("--noise", default=False, action='store_true', help="Add noise to the position errors fed into ILC.")
  parser.add_argument("--filter", default=False, action='store_true', help="Filter the position errors fed into ILC.")
  args = parser.parse_args()

  dt = args.dt
  t_end = 1.0
  end_pos = 0.5

  ilc_c, DIMS = system_map[args.system]

  AXIS = 1 if DIMS == 3 else 0

  ilc = ilc_c(feedback=args.feedback)

  N = int(round(t_end / dt))
  ts = np.linspace(0, t_end, N + 1)

  print("No. of steps is", N)

  pos_poly = get_poly(0, end_pos=end_pos, duration=t_end)
  poss_des = np.polyval(pos_poly, ts)

  if args.system in ['2dpos', 'linearpos', '2dposlin', 'nl1d']:
    for i in range(min(4, N)):
      poss_des[i] = 0.0

  poss_des_vec = np.zeros((N + 1, DIMS))
  poss_des_vec[:, AXIS] = poss_des

  vels_des = np.polyval(np.polyder(pos_poly), ts)
  vels_des_vec = np.zeros((N + 1, DIMS))
  vels_des_vec[:, AXIS] = vels_des

  acc_poly = np.polyder(np.polyder(pos_poly))
  accels_des = np.polyval(acc_poly, ts)

  if args.system in ['linear', '2d', 'linearpos']:
    accels_des[1] = 0.0

  accels_des_vec = np.zeros((N + 1, DIMS))
  accels_des_vec[:, AXIS] = accels_des

  lifted_control = np.zeros(N * ilc.n_control)
  if DIMS == 2 or DIMS == 3:
    lifted_control[::ilc.n_control] = g

  lifted_state = np.zeros(N * ilc.n_state)
  if DIMS == 3:
    lifted_state[2::ilc.n_state] = 1.0

  class Controller:
    def __init__(self, lifted_control, lifted_state, poss_des, vels_des):
      self.poss_des = poss_des
      self.vels_des = vels_des

      self.controls = []
      self.accs_des = []
      self.angvels_des = []

      if args.system in ['linear', 'linearpos']:
        control_j = dt * np.cumsum(lifted_control)
        control_a = dt * np.cumsum(control_j)

      for i in range(N):
        self.controls.append(lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)])

        state_now = lifted_state[ilc.n_state * i : ilc.n_state * (i + 1)]

        if args.system == 'simple' or args.system == 'trivial':
          pass

        elif args.system in ['linear', 'linearpos']:
          _, _, theta, angvel = state_now
          #theta = control_a[i]
          #angvel = control_j[i]
          self.accs_des.append(theta)
          self.angvels_des.append(angvel)

        elif '2d' in args.system:
          px, pz, vx, vz, theta, angvel = state_now
          #z_axis = np.array((-np.sin(theta), np.cos(theta)))
          #accel = self.controls[-1][0] * z_axis - np.array((0, g))
          accel = np.zeros(2)

          self.accs_des.append(accel)
          self.angvels_des.append(angvel)

        elif args.system == '3d':
          z_axis, angvel = state_now[:3], state_now[3:]
          accel = self.controls[-1][0] * z_axis - np.array((0, 0, g))

          self.accs_des.append(accel)
          self.angvels_des.append(angvel)

      self.index = 0

    def get(self, x):
      ilc_controls = self.controls[self.index]
      if args.feedback:
        if args.system == 'trivial':
          feedback = ilc.feedback(x, self.vels_des[self.index], ilc_controls[0])
        elif args.system == 'simple':
          feedback = ilc.feedback(x, self.poss_des[self.index], self.vels_des[self.index], ilc_controls[0])
        elif args.system in ['linear', 'linearpos']:
          feedback = ilc.feedback(x,
            self.poss_des[self.index],
            self.vels_des[self.index],
            self.accs_des[self.index],
            self.angvels_des[self.index],
            ilc_controls[0]
          )
        else:
          feedback = ilc.feedback(x,
            self.poss_des[self.index],
            self.vels_des[self.index],
            self.accs_des[self.index],
            self.angvels_des[self.index],
            ilc_controls[0],
            ilc_controls[1:]
          )

        self.index += 1
        return feedback

      self.index += 1
      return ilc_controls

  trial_poss = []
  trial_vels = []
  trial_accels = []
  trial_omegas = []

  for iter_no in range(args.trials):
    controller = Controller(lifted_control, lifted_state, poss_des_vec, vels_des_vec)
    data = ilc.simulate(t_end, controller.get, dt=dt)

    if args.system == '3d':
      poss_vec = data[:, :3]
      accels_vec = np.diff(data[:, 3:6], axis=0) / dt
      rpys = data[:, 6:9]
      ang_body = data[:, 9:]

    elif '2d' in args.system:
      poss_vec = data[:, :2]
      vels = data[:, 2:4]
      accels_vec = np.diff(data[:, 2:4], axis=0) / dt
      accels_vec = np.vstack((accels_vec, np.zeros(2)))
      thetas = data[:, 4]
      angs = data[:, 5]

    elif args.system in ['linear', 'linearpos', 'nl1d']:
      poss_vec = data[:, 0:1]
      vels = data[:, 1:2]
      accels_vec = np.diff(data[:, 1:2], axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))
      #accels_vec = data[:, 2:3]
      thetas = data[:, 2]
      angs = data[:, 3]

    elif args.system == 'simple':
      poss_vec = data[:, 0:1]
      accels_vec = np.diff(data[:, 1:2], axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))

    elif args.system == 'trivial':
      poss_vec = data[:, 0:1]
      accels_vec = np.diff(data[:, 0:1], axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))

    pos_errors = poss_vec - poss_des_vec
    abs_pos_errors = np.abs(pos_errors)
    accel_errors = accels_vec - accels_des_vec
    abs_accel_errors = np.abs(accel_errors)

    title_s = "Iteration %d" % (iter_no + 1)
    print("============")
    print(title_s)
    print("============")
    print("Avg. pos error:", np.mean(abs_pos_errors))
    print("Max. pos error:", np.max(abs_pos_errors))
    print("Avg. acc error:", np.mean(abs_accel_errors))
    print("Max. acc error:", np.max(abs_accel_errors))

    trial_poss.append(poss_vec)
    if args.system in ['linear', 'linearpos', 'nl1d', '2dpos', '2dposlin']:
      trial_vels.append(vels)

    trial_accels.append(accels_vec)
    if args.system  in ['linear', '2d', '2dpos']:
      trial_omegas.append(angs[:, np.newaxis])

    #calB = np.zeros((N * ilc.n_state, N * ilc.n_control))
    #calC = np.zeros((N * ilc.n_out, N * ilc.n_state))
    #calD = np.zeros((N * ilc.n_out, N * ilc.n_control))
    calCBpD = np.zeros((N * ilc.n_out, N * ilc.n_control))
    #vecc_gen = np.zeros((A.shape[0] * N, A.shape[0]))

    lin_states = np.zeros((ilc.n_state * (N + 1)))

    As = []
    Bs = []
    Cs = []
    Ds = []

    # First we linearize the dynamics around the controls and resulting states.
    for i in range(N + 1):
      if args.system == '3d':
        rot = Rotation.from_euler('ZYX', rpys[i, ::-1]).as_dcm()
        z_axis = rot[:, 2]
        state = np.hstack((z_axis, ang_body[i]))

      elif args.system in ['linear', 'linearpos', 'nl1d']:
        state = np.array((poss_vec[i, 0], vels[i], thetas[i], angs[i]))

      elif args.system in ['2dpos', '2dposlin']:
        assert poss_vec.shape[0] == N + 1
        assert vels.shape[0] == N + 1
        assert len(thetas) == N + 1

        state = np.array((poss_vec[i, 0], poss_vec[i, 1], vels[i, 0], vels[i, 1], thetas[i], angs[i]))

      elif args.system == 'simple' or args.system == 'trivial':
        state = poss_vec[i]

      lin_states[ilc.n_state * i : ilc.n_state * (i + 1)] = state
      if i < N:
        control = lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)]
      else:
        control = lifted_control[ilc.n_control * (i - 1) : ilc.n_control * i]

      ilc.pos_des = poss_des_vec[i, :]
      ilc.vel_des = vels_des_vec[i, :]
      ilc.acc_des = accels_des_vec[i, :]

      A, B, C, D = ilc.get_ABCD(state, control, dt)
      As.append(A)
      Bs.append(B)
      Cs.append(C)
      Ds.append(D)

      # TODO: Use D
      assert np.all(D == 0)

    Apowers = [np.eye(ilc.n_state) for _ in range(N)]
    for i in range(N):
      for j in range(N - i):
        row_ind = i + j
        col_ind = j

        calCBpD[ilc.n_out *     row_ind : ilc.n_out * (row_ind + 1),
                ilc.n_control * col_ind : ilc.n_control * (col_ind + 1)] = Cs[row_ind + 1].dot(Apowers[j].dot(Bs[col_ind]))

        Apowers[j] = As[row_ind + 1].dot(Apowers[j])
      #calCBpD[ilc.n_out * i : ilc.n_out * (i + 1), ilc.n_control * i : ilc.n_control * (i + 1)] += D

    #output_map = calC.dot(calB) + calD

    if args.noise:
      for i in range(len(pos_errors)):
        pos_errors[i] += np.random.normal(0, 0.001)

    lifted_output_error = np.zeros((ilc.n_out * N))
    for i in range(N):
      if 'pos' in args.system or args.system == 'nl1d':
        if args.filter:
          pos_errors[:, 0] = savgol_filter(pos_errors[:, 0], 11, 3)
          pos_errors[:, 1] = savgol_filter(pos_errors[:, 1], 11, 3)

        lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = pos_errors[i + 1]
      else:
        lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = accel_errors[i + 1]

    # ILC update
    update, _, _, _ = np.linalg.lstsq(calCBpD, lifted_output_error, rcond=None)
    new_lifted_control = lifted_control - args.alpha * update

    #print(lifted_control)
    #print(calCBpD)
    #print(lifted_output_error)
    #print(update)
    #input()
    #print(new_lifted_control)

    #lifted_state = lin_states + calB.dot(new_lifted_control - lifted_control)
    lifted_control = new_lifted_control

  def plot_trials(datas, desired, title, ylabel):
    start_color = np.array((1, 0, 0, 0.5))
    end_color = np.array((0, 1, 0, 0.5))
    axes = "XYZ" if DIMS == 3 else "XZ" if DIMS == 2 else "X"
    #for axis in [AXIS]:
    for axis in range(DIMS):
      title_s = "Actual vs. Desired %s %s" % (title, axes[axis])
      plt.figure(title_s)
      plt.plot(ts, desired[:, axis], "k:", linewidth=2, label="Desired")
      for i, trial_data in enumerate(datas):
        alpha = float(i) / len(datas)
        line_color = (1 - alpha) * start_color + alpha * end_color

        plot_args = {}
        if i == 0 or i == len(datas) - 1:
          plot_args['label'] = "Trial %d" % (i + 1)

        plt.plot(ts[:len(trial_data)], trial_data[:, axis], color=line_color, linewidth=2, **plot_args)

      plt.xlabel("Time (s)")
      plt.ylabel(ylabel % axes[axis])
      plt.legend()
      plt.title(title_s)

    plt.savefig("ilc_%s.png" % title.lower())

  plot_trials(trial_poss, poss_des_vec, "Position", "Pos. %s (m)")
  plot_trials(trial_vels, vels_des_vec, "Velocity", "Vel. %s (m/s)")
  plot_trials(trial_accels, accels_des_vec, "Acceleration", "Accel. %s (m/s^2)")
  #plot_trials(trial_omegas, np.zeros((N, DIMS)), "Angular Velocity", "$\omega$ %s (rad/s)")

  plt.show()
