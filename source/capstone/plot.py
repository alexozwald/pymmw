class Plot3Dv:
    """  *** WIP -> MOVING TO MATPLOTLIB.ANIMATION.FUNCANIMATION() VIA main() AS GENERATOR FUNC ***
    Refs for improvement:
    [1] [Blitting Class sans-animation (Article)](https://coderslegacy.com/matplotlib-blitting-tutorial/)
    [2] [Matplotlibs official adv. example blitting class](https://matplotlib.org/stable/tutorials/advanced/blitting.html)
    """

    def __init__(self, frame_lag: int = 20):
        self.bounds = dict(x=[-2,25], y=[-2,25], z=[-5,15])
        init_v_bounds = dict(vmin=-40, vmax=40)
        view_angle_3d = dict(elev=14, azim=-135, roll=0)

        self.df = pd.DataFrame()
        self.frame_lag = frame_lag

        set_theme(style='darkgrid', context='notebook')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(**view_angle_3d)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(*self.bounds['x'])
        self.ax.set_ylim(*self.bounds['y'])
        self.ax.set_zlim(*self.bounds['z'])

        # pre-generate colorbar
        #self.cmap = cubehelix_palette(as_cmap=True)
        self.cmap = diverging_palette(h_neg=150, h_pos=276, s=73, l=30, sep=1, center='light', as_cmap=True)
        self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(**init_v_bounds))
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=self.ax, pad=0.1)
        self.cbar.ax.set_title("Velocity (m/s)")

        # Launch Interactive Mode & GUI (non-blocking ig)
        plt.ion()
        plt.show(block=False)

    def check_max_bounds(self, df_new: pd.DataFrame):
        """ Update axis bounds from defaults if needed """
        bounds_updated = False

        for axis in ['x', 'y', 'z']:
            min_val, max_val = df_new[axis].agg(['min', 'max'])
            old_min, old_max = self.bounds[axis]

            if (new_bounds := (min(min_val, old_min), max(max_val, old_max))) != (old_min, old_max):
                self.bounds[axis] = new_bounds
                bounds_updated = True

        return bounds_updated

    def add_pts(self, df_new: pd.DataFrame):
        """ Add points from current df-reading to 3D Scatter """
        # clear axis of data
        self.ax.cla()
        #self.ax.lines.clear()

        # update & prune points
        fr_cutoff = df_new['frame'].max() - self.frame_lag
        self.df = pd.concat([self.df, df_new], ignore_index=True)
        self.df = self.df.loc[self.df['frame'] >= fr_cutoff]

        # re-normalize opacity
        min_opacity, max_opacity = 0.3, 1.0
        min_frame, max_frame = self.df['frame'].min(), self.df['frame'].max()
        normalized_opacity = self.df['frame'].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity).fillna(0.3)

        # compute contents of new new scatter plot
        x = self.df['x'].values
        y = self.df['y'].values
        z = self.df['z'].values
        v = self.df['v'].values
        self.ax.scatter(x, y, z, c=v, cmap='Purples', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)

        # Set limits, axes, and title+subtitle
        self.ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')
        time_str = pd.to_datetime(df_new['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        self.ax.set_title(f"Velocity Plot\nTime: {time_str} (fr: {df_new['frame'].max()})")

        # Write changes onto gui
        plt.draw()
        plt.pause(0.001)
        #self.fig.canvas.flush_events()
