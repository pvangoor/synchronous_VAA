# Velocity Aided Attitude Simulation Code
# by Pieter van Goor (p.c.h.vangoor@utwente.nl), May 2025

# This could requires numpy, matplotlib, and pylie to run.
# Pylie can be installed from https://github.com/pvangoor/pylie
# progressbar2 is an optional extra import used to display simulation progress.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors
from pylie import SE3, SO3, R3
try:
    from progressbar import progressbar
except ImportError:
    def progressbar(s): return s

# Set matplotlib's line and font styles and some figure size variables.
plt.rc('lines', linewidth=1.0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
figheight = 2.0
figsize_factor = 1.5

# Simulation Settings
dt = 0.01
max_t = 10.0
gnss_rel_freq = 100
mag_rel_freq = 20
integration_steps = 50

max_steps = int(max_t / dt)

mRef = np.array([1, 0, 0])

# This class implements the VAA observer, with optional gains set at the start.
class VAAObserver:
    def __init__(self, gain_k, gain_c, gain_m, poor_init=True):
        self.XHat = SE3.identity()
        self.ZHat = SE3.identity()

        # Use a large offset to show almost-global convergence.
        # Set poor_init to False to turn this off.
        if poor_init:
            self.XHat._R = SO3.exp(np.pi*0.99*np.array([1.0, 0.0, 0.0]))
            self.XHat._x = R3(np.array([3, -2, 2]))

        self.gain_k = gain_k
        self.gain_c = gain_c
        self.gain_m = gain_m

    def compute_corrections(self, vTru):
        # Compute the correction terms Gamma and Delta based on the measured velocity
        vEst = self.XHat.x().as_vector()
        z = self.ZHat.x().as_vector()

        Gamma = np.zeros(6)
        Delta = np.zeros(6)

        Gamma[3:6] = self.gain_k * (vTru - z)
        Delta[0:3] = self.gain_c * SO3.skew(vEst-z) @ (vTru - z)
        Delta[3:6] = self.gain_k * (vTru - vEst) - \
            SO3.skew(Delta[0:3]) @ z

        return Delta, Gamma

    def integrate_dynamics(self, G, U):
        # Integrate the observer dynamics using Lie-group Euler integration
        self.XHat = SE3.exp(dt * G) * self.XHat * SE3.exp(dt * U)
        self.ZHat = SE3.exp(dt * G) * self.ZHat

    def receive_gnss(self, y):
        Delta, Gamma = self.compute_corrections(y)
        self.XHat = SE3.exp(dt * (Delta)) * self.XHat
        self.ZHat = self.ZHat * SE3.exp(dt*Gamma)

    def receive_mag(self, m):
        Delta = np.zeros(6)
        Delta[0:3] = self.gain_m * SO3.skew(self.XHat.R() * m) @ (mRef)
        Delta[3:6] = - SO3.skew(Delta[0:3]) @ self.ZHat.x().as_vector()
        self.XHat = SE3.exp(dt * (Delta)) * self.XHat


class SynchronousVAAObserver(VAAObserver):
    def receive_gnss(self, y):
        adt = dt*gnss_rel_freq/integration_steps
        for _ in range(integration_steps):
            Delta, Gamma = self.compute_corrections(y)
            self.XHat = SE3.exp(adt * Delta) * self.XHat
            self.ZHat = self.ZHat * SE3.exp(adt * Gamma)

    def receive_mag(self, m):
        adt = dt*mag_rel_freq/integration_steps
        Delta = np.zeros(6)
        for _ in range(integration_steps):
            Delta[0:3] = self.gain_m * SO3.skew(self.XHat.R() * m) @ (mRef)
            Delta[3:6] = - SO3.skew(Delta[0:3]) @ self.ZHat.x().as_vector()
            self.XHat = SE3.exp(adt * Delta) * self.XHat


def generate_velocity(t):
    # Generate an input velocity.
    omega = 1.0 * np.array([0, 0, 1])
    g = 9.81 * np.array([0, 0, 1])
    a = np.array([0.0, 2.0, 0]) - g

    G = np.concatenate((np.zeros(3), g))
    U = np.concatenate((omega, a))

    return U, G


def run_once(gain_k, gain_c, gain_m):
    # Run one simulation.

    # Initialise the true state to identity.
    X = SE3(SO3.identity(), np.array([2., 0, 0]))

    # Initialise the observer with poor initial conditions.
    # Use either a continuous-time or discretised observer.
    # obs = VAAObserver(gain_k, gain_c, gain_m, poor_init=True)
    obs = SynchronousVAAObserver(gain_k, gain_c, poor_init=True)

    # Create some lists to store data.
    statesTru = []
    statesEst = []
    statesAux = []

    for step in progressbar(range(max_steps)):
        # Store data
        statesTru.append(X)
        statesEst.append(obs.XHat)
        statesAux.append(obs.ZHat)

        # Measure the input signals and GNSS velocity
        U, G = generate_velocity(step*dt)
        vTru = X.x().as_vector()
        mTru = X.R().inv() * mRef

        # Integrate the system and observer
        X = SE3.exp(dt * G) * X * SE3.exp(dt * U)
        if (step+1) % gnss_rel_freq == 0:
            obs.receive_gnss(vTru)
        if (step+1) % mag_rel_freq == 0:
            obs.receive_mag(mTru)
        obs.integrate_dynamics(G, U)

    return statesTru, statesEst, statesAux


# Run the simulation to gather data.
statesTru, statesEst, statesAux = run_once(5.0, 1.0, 5.0)


def plot_errors(fig, ax, statesTru, statesEst, statesAux, ls, label):
    # This function plots the attitude and velocity errors, and the Lyapunov function value.

    def segs(seq): return [((times[k], seq[k]), (times[k+1], seq[k+1]))
                           for k in range(len(times)-1)]
    colors = np.tile(matplotlib.colors.hex2color('#0033CC'), (len(times)-1, 1))
    for k in range(colors.shape[0] // mag_rel_freq):
        colors[mag_rel_freq *
               (k+1)-1, :] = matplotlib.colors.hex2color('#FFAA00')
    for k in range(colors.shape[0] // gnss_rel_freq):
        colors[gnss_rel_freq *
               (k+1)-1, :] = matplotlib.colors.hex2color('#FF0000')

    att_error = np.array([np.linalg.norm(SO3.log(XTru.R() * XEst.R().inv()))
                          * 180.0 / np.pi for XTru, XEst in zip(statesTru, statesEst)])
    ax[0].add_collection(LineCollection(
        segs(att_error), colors=colors, label=label))
    ax[0].plot(times, att_error, ls, alpha=0)

    ax[0].set_ylabel("Attitude\nError (deg)")
    ax[0].set_ylim([0, None])
    ax[0].set_xlim([times[0], times[-1]])
    ax[0].set_xticklabels([])
    ax[0].grid(True)

    vel_error = np.array([np.linalg.norm(XTru.x().as_vector(
    ) - XEst.x().as_vector()) for XTru, XEst in zip(statesTru, statesEst)])
    ax[1].add_collection(LineCollection(
        segs(vel_error), colors=colors, label=label))
    ax[1].plot(times, vel_error, ls, alpha=0)

    ax[1].set_ylabel("Velocity\nError (m/s)")
    ax[1].set_ylim([0, None])
    ax[1].set_xlim([times[0], times[-1]])
    ax[1].set_xticklabels([])
    ax[1].grid(True)

    lyapunov = np.array([np.linalg.norm((ZHat.inv()*XTru*XEst.inv()*ZHat).as_matrix() -
                        np.eye(4))**2 for XTru, XEst, ZHat in zip(statesTru, statesEst, statesAux)])
    ax[2].add_collection(LineCollection(
        segs(lyapunov), colors=colors, label=label))
    ax[2].plot(np.array(times), lyapunov, ls, alpha=0)

    ax[2].set_ylabel("Lyapunov\nValue")
    ax[2].set_yscale('log')
    ax[2].set_xlim([times[0], times[-1]])
    ax[2].grid(True)
    ax[2].set_xlabel("Time (s)")

    fig.suptitle("Observer Error Metrics")
    fig.set_figwidth(3.5*figsize_factor)
    fig.set_figheight(figheight*figsize_factor)


# Plot the errors.
error_fig, error_ax = plt.subplots(3, 1, layout='constrained')
times = [dt * step for step in range(max_steps)]
plot_errors(error_fig, error_ax, statesTru, statesEst, statesAux, 'r-', "Est.")


def plot_estimates(ax, statesEst, ls, label, highlight_corrections=False):
    eul_est = np.vstack([XEst.R().as_euler() for XEst in statesEst])
    vel_est = np.vstack([XEst.x().as_vector() for XEst in statesEst])

    if highlight_corrections:
        eul_seg = [[((times[k], eul_est[k, i]), (times[k+1], eul_est[k+1, i]))
                    for k in range(len(times)-1)] for i in range(3)]
        vel_seg = [[((times[k], vel_est[k, i]), (times[k+1], vel_est[k+1, i]))
                    for k in range(len(times)-1)] for i in range(3)]

        colors = np.tile(matplotlib.colors.hex2color(
            '#0033CC'), (len(times)-1, 1))
        for k in range(colors.shape[0] // mag_rel_freq):
            colors[mag_rel_freq *
                   (k+1)-1, :] = matplotlib.colors.hex2color('#FFAA00')
        for k in range(colors.shape[0] // gnss_rel_freq):
            colors[gnss_rel_freq *
                   (k+1)-1, :] = matplotlib.colors.hex2color('#FF0000')

        for i in range(3):
            eul_lc = LineCollection(eul_seg[i], colors=colors, label=label)
            vel_lc = LineCollection(vel_seg[i], colors=colors, label=label)
            ax[i, 0].add_collection(vel_lc)
            ax[i, 1].add_collection(eul_lc)

        # plot invisible lines to help matplotlib sort out axis limits
        for i in range(3):
            ax[i, 0].plot(times, vel_est[:, i], alpha=0)
            ax[i, 1].plot(times, eul_est[:, i], alpha=0)

    else:
        for i in range(3):
            ax[i, 0].plot(times, vel_est[:, i], ls, label=label)
            ax[i, 1].plot(times, eul_est[:, i], ls, label=label)


# Plot the estimates compared to the true values.
estimate_fig, estimate_ax = plt.subplots(3, 2, layout='constrained')
plot_estimates(estimate_ax, statesTru, 'k--', "True")
plot_estimates(estimate_ax, statesEst, 'r-',
               "Est.", highlight_corrections=True)

# Adjust the axes and labels of the estimate plot.
for i in range(3):
    estimate_ax[i, 0].set_xlim([times[0], times[-1]])
    estimate_ax[i, 1].set_xlim([times[0], times[-1]])

    if i == 1:
        estimate_ax[i, 1].set_ylim([-180.0/2, 180.0/2])
    else:
        estimate_ax[i, 1].set_ylim([-180.0, 180.0])

    estimate_ax[i, 0].grid()
    estimate_ax[i, 1].grid()

    if i < 2:
        estimate_ax[i, 0].set_xticklabels([])
        estimate_ax[i, 1].set_xticklabels([])
    else:
        estimate_ax[i, 0].set_xlabel("Time (s)")
        estimate_ax[i, 0].legend(loc='upper right')
        estimate_ax[i, 1].set_xlabel("Time (s)")


estimate_ax[0, 0].set_title("Velocity Estimation")
estimate_ax[0, 1].set_title("Attitude Estimation")
estimate_ax[0, 0].set_ylabel("x (m/s)")
estimate_ax[1, 0].set_ylabel("y (m/s)")
estimate_ax[2, 0].set_ylabel("z (m/s)")
estimate_ax[0, 1].set_ylabel("roll (deg)")
estimate_ax[1, 1].set_ylabel("pitch (deg)")
estimate_ax[2, 1].set_ylabel("yaw (deg)")

estimate_fig.set_figwidth(7.16*figsize_factor)
estimate_fig.set_figheight(figheight*figsize_factor)

# Uncomment these lines to save the figures.
# error_fig.savefig("figures/VAA_observer_error.pdf",
#                   bbox_inches='tight', pad_inches=0.02)
# estimate_fig.savefig("figures/VAA_estimation.pdf",
#                      bbox_inches='tight', pad_inches=0.02)

plt.show()
