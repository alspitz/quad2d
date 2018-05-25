import numpy as np

from quad import Control, State

class PD:
  def __init__(self, P, D):
    self.P = P
    self.D = D
    self.K = np.array((P, D))

  def output(self, value, derivative, desired_value, desired_derivative):
    return -self.P * (value - desired_value) - self.D * (derivative - desired_derivative)

class FlatController:
  def __init__(self, model, x_poly_coeffs, z_poly_coeffs, learner=None):
    self.model = model
    self.learner = learner

    self.x_polys = [x_poly_coeffs]
    for i in range(4):
      self.x_polys.append(np.polyder(self.x_polys[-1]))

    self.z_polys = [z_poly_coeffs]
    for i in range(4):
      self.z_polys.append(np.polyder(self.z_polys[-1]))

    self.pd = PD(10, 10)

  def compute_theta(self, z_body):
    return np.arcsin(-z_body[0])

  def solve_for_scalar(self, v1, A, v2):
    """ A^{-1} = A^T """
    return v1.T.dot(A).dot(v2)

  def get_des_data(self, x_des, z_des=np.zeros(4)):
    if self.learner is not None and self.learner.w is not None:
      return self.get_des_data_corrected(x_des, z_des)

    a_des = np.array((x_des[2], z_des[2])) # Acceleration
    j_des = np.array((x_des[3], z_des[3])) # Jerk
    s_des = np.array((x_des[4], z_des[4])) # Snap

    acc_vec = a_des + np.array((0, self.model.g))
    z_body = acc_vec / np.linalg.norm(acc_vec)

    theta = self.compute_theta(z_body)

    a_norm = np.linalg.norm(acc_vec)
    a_norm_dot = j_des.dot(z_body)

    z_body_dot = (j_des - a_norm_dot * z_body) / a_norm
    # z_body_dot = theta_dot * cross_mat * z_body
    cross_mat = np.array(((0, -1), (1, 0)))
    theta_vel = self.solve_for_scalar(z_body_dot, cross_mat, z_body)

    a_norm_ddot = s_des.dot(z_body) + j_des.dot(z_body_dot)
    z_body_ddot = (s_des - a_norm_ddot * z_body - 2 * a_norm_dot * z_body_dot) / a_norm

    theta_acc = z_body_ddot.T.dot(cross_mat).dot(z_body) + z_body_dot.T.dot(cross_mat).dot(z_body_dot)

    return a_norm, theta, theta_vel, theta_acc

  def get_des_data_corrected(self, x_des, z_des=np.zeros(4)):
    v_des = np.array((x_des[1], z_des[1])) # Velocity
    a_des = np.array((x_des[2], z_des[2])) # Acceleration
    j_des = np.array((x_des[3], z_des[3])) # Jerk
    s_des = np.array((x_des[4], z_des[4])) # Snap

    # TODO We omit angle and angle vel here because the learner doesn't use them... yet.
    nom_state = np.array((x_des[0], z_des[0], 0, x_des[1], z_des[1], 0))

    err = self.learner.predict(nom_state) / self.learner.dt

    dfdx = self.learner.get_deriv_x(nom_state) / self.learner.dt
    dfdx_vel = self.learner.get_deriv_x_vel(nom_state) / self.learner.dt

    dfdx2 = self.learner.get_dderiv_x_x(nom_state) / self.learner.dt
    dfdx_vel2 = self.learner.get_dderiv_x_vel_x_vel(nom_state) / self.learner.dt

    dfdt = dfdx.dot(v_des) + dfdx_vel.dot(a_des)
    d2fdt2 = dfdx2.dot(v_des).dot(v_des) + dfdx.dot(a_des) + dfdx_vel2.dot(a_des).dot(a_des) + dfdx_vel.dot(j_des)

    acc_vec = a_des + np.array((0, self.model.g)) - err
    z_body = acc_vec / np.linalg.norm(acc_vec)

    theta = self.compute_theta(z_body)

    a_norm = np.linalg.norm(acc_vec)
    a_norm_dot = (j_des - dfdt).dot(z_body)

    z_body_dot = (j_des - a_norm_dot * z_body - dfdt) / a_norm
    # z_body_dot = theta_dot * cross_mat * z_body
    cross_mat = np.array(((0, -1), (1, 0)))
    theta_vel = self.solve_for_scalar(z_body_dot, cross_mat, z_body)

    a_norm_ddot = (s_des - d2fdt2).dot(z_body) + (j_des - dfdt).dot(z_body_dot)
    z_body_ddot = (s_des - d2fdt2 - a_norm_ddot * z_body - 2 * a_norm_dot * z_body_dot) / a_norm

    theta_acc = z_body_ddot.T.dot(cross_mat).dot(z_body) + z_body_dot.T.dot(cross_mat).dot(z_body_dot)

    return a_norm, theta, theta_vel, theta_acc

  def _compute(self, x_des):
    m = self.model.m
    g = self.model.g
    I = self.model.I

    F = m * np.sqrt(self.model.g ** 2 + x_des[2] ** 2)
    tau = I * (-x_des[2]**2 * x_des[4] * g - x_des[4] * g**3 + 2 * x_des[2] * x_des[3]**2 * g) / \
              ( x_des[2]**2 + g**2) ** 2

    return F, tau

  def get_des(self, t):
    return [np.polyval(poly, t) for poly in self.x_polys], [np.polyval(poly, t) for poly in self.z_polys]

  def get_u(self, x, t):
    x_des, z_des = self.get_des(t)
    a_norm, theta_des, theta_vel_des, theta_acc_des = self.get_des_data(x_des, z_des)

    #diff = self.learner.predict(np.array((x_des[0], 0, 0, x_des[1], 0, 0)), np.array((0, 0)))

    ##xdot = np.array((x[3], x[4], x[5], x_des[2], 0, 0))
    ##xddot = np.array((x_des[2], 0, 0, x_des[3], 0, 0))

    #acc_corr = diff[0] / self.learner.dt
    #jerk_corr = x_des[2] * self.learner.w[3, 0] / self.learner.dt# + \
    #            #theta_vel_des * self.learner.w[3, 3] / self.learner.dt
    #snap_corr = x_des[3] * self.learner.w[3, 0] / self.learner.dt# + \
    #            #(tau / I) * self.learner.w[3, 3] / self.learner.dt

    #x_des[2] -= acc_corr
    #x_des[3] -= jerk_corr
    #x_des[4] -= snap_corr

    #return np.array(self._compute(x_des))

    F = self.model.m * a_norm
    tau = self.model.I * theta_acc_des

    return np.array((F, tau))

    #print("First u is", F, tau)

    #for j in range(1):
    #  F1, tau1 = F, tau

    #  if self.learner is not None and self.learner.w is not None:
    #    theta_des = self.get_des_theta(x_des)
    #    theta_vel_des = self.get_des_theta_vel(x_des)

    #    #theta_vel_des_2 = x_des[3] - x_des[3] * sin(x.theta) * sin

    #    #diff = self.learner.predict(x, np.array((F, tau)))
    #    diff = self.learner.predict(np.array((x_des[0], 0, theta_des, x_des[1], 0, theta_vel_des)), np.array((F, tau)))

    #    #xdot = np.array((x[3], x[4], x[5], x_des[2], 0, 0))
    #    #xddot = np.array((x_des[2], 0, 0, x_des[3], 0, 0))

    #    acc_corr = diff[3] / self.learner.dt
    #    jerk_corr = x_des[2] * self.learner.w[self.learner.xvel_input_ind, 3] / self.learner.dt# + \
    #                #theta_vel_des * self.learner.w[3, 3] / self.learner.dt
    #    snap_corr = x_des[3] * self.learner.w[self.learner.xvel_input_ind, 3] / self.learner.dt# + \
    #                #(tau / I) * self.learner.w[3, 3] / self.learner.dt

    #    #print("Correction is", acc_corr, jerk_corr, snap_corr)

    #    x_des[2] -= acc_corr
    #    x_des[3] -= jerk_corr
    #    x_des[4] -= snap_corr

    #    F, tau = self._compute(x_des)

    #    #tau -= diff[5] / self.learner.dt
    #    #tau -= x.theta_vel * self.learner.w[6, 5] / self.learner.dt
    #    #tau -= x.x_vel * self.learner.w[4, 5] / self.learner.dt

    #    #if abs(F - F1) < 1e-1 and abs(tau - tau1) < 1e-1:
    #    #  break

    #    #print("*" * 80)
    #    #print(j, F - F1)
    #    #print(j, tau - tau1)

    ##print("Second u is", F, tau)

    ##x_out = self.pd.output(x.x, x.x_vel, x_des[0], x_des[1])
    ##z_out = self.pd.output(x.z, x.z_vel, 0, 0)

    ##angle = np.tan(x_out / F)

    ##theta_des = np.arctan(-x_des[2] / g) - angle
    ##theta_vel_des = (-x_des[3] * g) / (g**2 + x_des[2]**2)

    ##torque_out = self.pd.output(x.theta, x.theta_vel, theta_des, theta_vel_des)

    ##F += m * z_out
    ##tau += I * torque_out

    #return np.array((F, tau))
