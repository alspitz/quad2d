import matplotlib.pyplot as plt

class XZPlotter:
  def __init__(self, dist_name, t_end):
    self.dist_name = dist_name

    self.data = []

  def add_data(self, ts, x_error, z_error, plot_sym, name, *args):
    self.data.append((ts, x_error, z_error, plot_sym, name))

  def plot(self, save_fig, filename=None):
    fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True)

    self.ax1.set_title("%s Disturbance" % self.dist_name, fontsize=18)
    self.ax1.set_ylabel('|X| Error (m)', fontsize=14)
    self.ax2.set_ylabel('|Z| Error (m)', fontsize=14)
    self.ax2.set_xlabel('Time (s)', fontsize=14)
    self.ax2.set_xlim((0, 1.0))
    self.ax1.grid()
    self.ax2.grid()

    for ts, x_error, z_error, plot_sym, name in self.data:
      self.ax1.plot(ts, x_error, plot_sym, label=name, linewidth=3)
      self.ax2.plot(ts, z_error, plot_sym, linewidth=3)

    #self.ax1.legend(loc='upper left', fontsize=14)
    plt.tight_layout()

    if save_fig:
      plt.savefig(filename)

class XPlotter:
  def __init__(self, dist_name, t_end):
    self.dist_name = dist_name
    self.t_end = t_end

    self.data = []

  def add_data(self, ts, x_error, z_error, plot_sym, name, *args):
    self.data.append((ts, x_error, plot_sym, name))

  def plot(self, save_fig, filename=None):
    fig, ax1 = plt.subplots(1, 1)

    x_lim = self.t_end
    y_lim = 0.8
    y_lim_plot = 1.0

    ax1.set_title("%s" % self.dist_name, fontsize=28)
    ax1.set_ylabel('X error (m)', fontsize=20)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_xticks([0, x_lim])
    ax1.set_xticklabels(["0", str(x_lim)], fontsize=20)

    ax1.set_ylim([0, y_lim_plot])
    ax1.set_yticks([0, y_lim])
    ax1.set_yticklabels(["0", str(y_lim)], fontsize=20)

    for ts, x_error, plot_sym, name in self.data:
      ax1.plot(ts, x_error, plot_sym, label=name, linewidth=6)

    ax1.legend(loc='upper left', fontsize=16)
    plt.tight_layout()

    if save_fig:
      plt.savefig(filename)

class TrajPlotter:
  def __init__(self, dist_name, t_end):
    self.dist_name = dist_name
    self.t_end = t_end

    self.data = []

  def add_data(self, ts, x_error, z_error, plot_sym, name, x_vals, z_vals, x_des, z_des):
    self.des_x = x_des
    self.des_z = z_des
    self.data.append((ts, x_vals, z_vals, plot_sym, name))

  def plot(self, save_fig, filename=None):
    fig, ax1 = plt.subplots(1, 1)

    x_lim = self.t_end
    y_lim = 1.0
    y_lim_plot = 1.65

    for ts, xs, zs, plot_sym, name in self.data[::-1]:
      ax1.plot(ts, xs, color=plot_sym, label=name, linewidth=5)

    ax1.plot(ts, self.des_x, "k--", label="Desired", linewidth=2, alpha=1.0)

    ax1.legend(loc='upper left', fontsize=16)

    #ax1.set_title("Position vs. Time", fontsize=28)
    ax1.set_ylabel('Position', fontsize=20)
    ax1.set_xlabel('Time', fontsize=20)
    ax1.set_xlim([0, x_lim])
    #ax1.set_xticks([0, x_lim])
    #ax1.set_xticklabels(["0", str(x_lim)], fontsize=20)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_ylim([0, y_lim_plot])
    #ax1.set_yticks([0, y_lim])
    #ax1.set_yticklabels(["0", str(y_lim)], fontsize=20)

    plt.tight_layout()

    if save_fig:
      plt.savefig(filename)

class DistPlotter:
  def __init__(self, dist_name, strategy):
    self.dist_name = dist_name
    self.strategy = strategy
    self.data = []

    self.colors = (
      'red',
      'blue',
      'green',
      'orange',
      'black',
      'purple'
    )

  def add_run(self, ts, true, est, trial):
    self.data.append((ts, true, est, trial))

  def plot(self):
    if not len(self.data):
      return

    self.fig, (self.ax_x, self.ax_z) = plt.subplots(2, 1, sharex=True)

    self.ax_x.set_ylabel("X Dist.", fontsize=20)
    self.ax_z.set_ylabel("Z Dist.", fontsize=20)
    self.ax_x.set_title("%s: %s" % (self.dist_name, self.strategy), fontsize=28)
    self.ax_z.set_xlabel("Time (s)", fontsize=20)

    for i, (ts, true, est, trial) in enumerate(self.data):
      for ax, ind in [(self.ax_x, 0), (self.ax_z, 1)]:
        ax.plot(ts[:-1], true[:, ind], color=self.colors[i], linestyle='--')
        ax.plot(ts[:-1], est[:, ind],  color=self.colors[i], label="Trial %d" % (trial + 1))

    self.ax_x.legend()
    plt.tight_layout()
