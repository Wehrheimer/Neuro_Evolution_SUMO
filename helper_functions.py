from matplotlib import cbook
import numpy as np
import matplotlib.pyplot as plt
from mpldatacursor import datacursor


def strcmp(a, b):
    if a == b:
        result = 1
    else:
        result = 0
    return result


class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists, tolerance=5, offsets=(-20, 20),
                 template='x: %0.2f\ny: %0.2f', display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be
            selected.
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%)
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            annotation.set_text(self.template % (x, y))
            annotation.set_visible(True)
            event.canvas.draw()


def plot_results(cum_reward, reward_mean100, SUMO, vehicle_ego, dynamics_ego, controller, training):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(4, 1, 1)  # (5, 1, 1)
    plt.ylabel('v in km/h', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.ylim([0, 80])

    #plt.xlabel('t in s', fontsize=46)

    ax2 = fig1.add_subplot(4, 1, 2, sharex=ax1)
    plt.ylabel('d in m', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.ylim([0, 800])
    ax3 = fig1.add_subplot(4, 1, 3, sharex=ax1)
    plt.ylabel('a in m/s^2', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.ylim([-8, 5])
    # ax4 = fig1.add_subplot(4, 1, 4, sharex=ax1)
    # plt.ylabel('mode', fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.grid(True)
    ax5 = fig1.add_subplot(4, 1, 4, sharex=ax1)
    plt.ylabel('M_trac in Nm', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.xlabel('t in s', fontsize=12)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    #fig1.canvas.mpl_connect('pick_event', onpick)
    ax1.plot(SUMO.time, SUMO.v_prec * 3.6, 'r', linewidth=2)
    ax1.plot(SUMO.time, SUMO.v_ego * 3.6, c='#1f77b4', linewidth=2)

    if controller == 'DDPG_v':
        ax1.plot(SUMO.time, SUMO.v_set*3.6)
        #ax1.plot(SUMO.time, SUMO.v_set_nonoise)
    ax2.plot(SUMO.time, SUMO.distance, c='#1f77b4', linewidth=2)
    ax3.plot(SUMO.time, SUMO.a_car, 'r', linewidth=2)
    ax3.plot(SUMO.time, SUMO.a_ego, c='#1f77b4', linewidth=2)

    #ax3.plot(SUMO.time, SUMO.a_set)
    if controller == 'hybrid':
        ax3.plot(SUMO.time, SUMO.a_hybrid)
    #ax4.plot(SUMO.time, SUMO.mode_ego)
    ax5.plot(SUMO.time, dynamics_ego.M_trac, c='#1f77b4', linewidth=2)

    plt.xlim([0, (SUMO.step+1)*SUMO.sim['timestep']])

    # Plot Rewards
    if training:
        fig2 = plt.figure()
        ax6 = fig2.add_subplot(2, 1, 1)
        plt.ylabel('Cumulative Reward')
        ax7 = fig2.add_subplot(2, 1, 2, sharex=ax6)
        plt.ylabel('Cum. Reward Mean 100')
        plt.xlabel('Episode')
        ax6.plot(cum_reward)
        ax7.plot(reward_mean100)

    # Plot operating strategy behaviour
    fig3 = plt.figure()
    ax8 = fig3.add_subplot(7, 1, 1)
    plt.ylabel('v in m/s')
    ax9 = fig3.add_subplot(7, 1, 2, sharex=ax8)
    plt.ylabel('M_ICE in Nm')
    ax10 = fig3.add_subplot(7, 1, 3, sharex=ax8)
    plt.ylabel('M_EM in Nm')
    ax11 = fig3.add_subplot(7, 1, 4, sharex=ax8)
    plt.ylabel('TM gear')
    ax12 = fig3.add_subplot(7, 1, 5, sharex=ax8)
    plt.ylabel('eTM gear')
    ax13 = fig3.add_subplot(7, 1, 6, sharex=ax8)
    plt.ylabel('kombTM gear')
    ax14 = fig3.add_subplot(7, 1, 7, sharex=ax8)
    plt.ylabel('Fuel consumption in l/100km')

    ax8.plot(SUMO.time, SUMO.v_ego)
    ax9.plot(SUMO.time, dynamics_ego.M_ICE_opt)
    ax10.plot(SUMO.time, dynamics_ego.M_EM_opt)
    ax11.plot(SUMO.time, dynamics_ego.TM_gear_opt)
    ax12.plot(SUMO.time, dynamics_ego.eTM_gear_opt)
    ax13.plot(SUMO.time, dynamics_ego.kombTM_gear_opt)
    #ax14.plot(SUMO.time, vehicle_ego.fuel_cons, 'b')
    ax14.plot(SUMO.time, vehicle_ego.fuel_cons_ECMS, 'r')
    plt.xlim([0, (SUMO.step + 1) * SUMO.sim['timestep']])
    plt.ylim([0, 50])
    datacursor(draggable=True)
    #DataCursor([ax8, ax9, ax10, ax11, ax12, ax13, ax14])
    plt.show(block=True)


class Plot():
    """Not used at the moment"""
    def __init__(self, training):
        plt.ion()
        if training:
            self.fig_running, (self.ax_running_1, self.ax_running_2, self.ax_running_3, self.ax_running_4) = figure.subplots(4, 1)
            self.ax_running_1.set_ylabel('Cum. Reward Mean 100')
            self.ax_running_2.set_xlabel('Episode')
            #self.reward_mean100 = []
            #self.ax_running_1.plot(self.reward_mean100)
            self.ax_running_1.plot([])
            self.fig_running.show()

    def plot_running(self, r_mean100, observed_weights, episode, cum_reward_evaluation, critic_loss, weight_grad_sum, warmup_time, step_counter):
        r_mean100 = r_mean100[:episode+1]
        observed_weights = observed_weights[:episode+1, :]
        critic_loss = critic_loss[warmup_time:step_counter+1]
        plt.close(self.fig_running)
        self.ax_running_1.set_ylabel('Cum. reward (mean100)')
        self.ax_running_1.plot(r_mean100)
        self.ax_running_2.set_ylabel('Weights and biases')
        self.ax_running_2.set_xlabel('Episode')
        for ii in range(np.size(observed_weights, 1)):
            self.ax_running_2.plot(observed_weights[:, ii])  # self.ax_running_2.plot(weight_grad_sum)
        self.ax_running_3.set_ylabel('Cum. reward (evaluation)')
        self.ax_running_3.set_xlabel('Evaluation episode')
        self.ax_running_3.plot(cum_reward_evaluation)
        self.ax_running_4.set_ylabel('Critic Loss')
        self.ax_running_4.plot(critic_loss)



        del r_mean100, observed_weights, critic_loss


def plot_Q_map(Q_map):
    """TODO: Adapt for current controllers"""
    if nn_controller.feature_number == 2:  # currently: distance, v_diff
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        distance = range(500)
        actions = np.linspace(-2., 2., nn_controller.action_space)
        X, Y = np.meshgrid(distance, actions)
        ax.plot_surface(X, Y, Q_map[:, 19, :].T)  # Q_map[:, 19, :].T
        ax.set_xlabel('distance')
        ax.set_ylabel('action')
        ax.set_zlabel('Q-value')
        plt.xlim([0, 500])
    elif nn_controller.feature_number == 1:  # currently: v_ego
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v_ego = np.linspace(0., 30., 50)
        actions = np.linspace(-3., 3., 50)
        X, Y = np.meshgrid(v_ego, actions)
        ax.plot_surface(X, Y, Q_map.T)
        ax.set_xlabel('v_ego')
        ax.set_ylabel('action')
        ax.set_zlabel('Q-value')
        #plt.xlim([0, 50])
    plt.show(block=True)


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

