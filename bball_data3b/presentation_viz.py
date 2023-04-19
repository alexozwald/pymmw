import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import numpy as np
from helper import REPO_DIR

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from helper import REPO_DIR, orjson_to_df2, do_hdbscan, mk_stacks, stack_over_v

PATH__JC_LOG_DATA = [ f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log2.json' ]
PATH__EXTRAS = [
    f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log1.json',
    f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log2.json',
    f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log3.json',
    f'{REPO_DIR}/bball_data3b/data/2023.03.21_jc/log4.json',
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-56-36 -- OLD.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-38-00.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-40-23.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-46-36.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_09-50-42.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.30_lab_lobs/2023-03-30_10-33-06.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-38-09.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-39-26.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_12-50-52.json.log', 
    f'{REPO_DIR}/bball_data3b/data/2023.03.31_pitch/2023-03-31_13-04-42.json.log', 
]
PATH__3D_EXTRAS = PATH__EXTRAS[4:]
PATH_TAGGED_DATA = f"{REPO_DIR}/bball_data3b/data/all_data_tagged.parquet"

## Pre-Generated Data Files
#JC_LOG_DATA :: orjson_to_df2(PATH__JC_LOG_DATA).to_csv(f"{REPO_DIR}/bball_data3b/data/data_jc_12hits.csv", index=False)
#DATA_3D     :: orjson_to_df2(PATH__3D_EXTRAS).to_csv(f"{REPO_DIR}/bball_data3b/data/data_3d.csv", index=False)
#DATA_ALL    :: orjson_to_df2(PATH__EXTRAS).to_csv(f"{REPO_DIR}/bball_data3b/data/data_all.csv", index=False)
DATA_JC_TST = pd.read_csv(f'{REPO_DIR}/bball_data3b/data/data_jc_12hits.csv', engine='pyarrow')
DATA_ALL = pd.read_csv(f'{REPO_DIR}/bball_data3b/data/data_all.csv', engine='pyarrow')
DATA_3D = pd.read_csv(f'{REPO_DIR}/bball_data3b/data/data_3d.csv', engine='pyarrow')
DATA_ALL_TAGGED = pd.read_parquet(PATH_TAGGED_DATA, engine='pyarrow')

'''
def ball_detect(df: pd.DataFrame, OUT_SCORE=0.75, V_MIN=17) -> pd.DataFrame:
    # calc distance for all potential bballs
    df['dist'] = (df['x'].pow(2) + df['y'].pow(2) + df['z'].pow(2)).pow(0.5)
    # filter out velocities + outliers
    bballs = df.loc[df['v'].abs() >= V_MIN]
    bballs = bballs.loc[bballs['outlier'] >= OUT_SCORE]
    # note all potential bballs
    bball_v = df.loc[df['dist'] == df['dist'].min()]
    bball_v = bball_v.loc[bball_v['v'] == bball_v['v'].max()].iloc[0] # iloc[0] => convert to series

    # pick ball@max_v using closest point, fallback to max velocity (fallback unlikely)
    bball_sent = bball.loc[bball['dist'] == bball['dist'].min()]
    bball_sent = bball_sent.loc[bball_sent['v'] == bball_sent['v'].max()]

    pass


    df.loc[bballs.index, ''] = 

    # Create a boolean mask to select the rows
    mask = df.index.isin(IDXS)
    # Use .loc to set the values
    df.loc[mask, 'COL'] = VALS
    df.at[IDX, 'COL'] = VAL

    pass

# USAGE EXAMPLE FOR SIMPLICITY:
df_done = 
for g, df_grpby in df.groupby('stack'):
    ball_detect(df_grpby)
'''


class Viz:
    def __init__(self) -> None:
        self.OPACITY_COL = 'frame'
        self.OPACITY_PCT = (0.3, 1.0)
        self.ANIM_COL = 'stack'
        self.bounds = dict(x=[-2,15], y=[-2,15], z=[-3,10])
        self.INIT_V_BOUNDS = dict(vmin=-40, vmax=40)
        self.VIEW_ANGLE_3D = dict(elev=25, azim=-135, roll=0)
        sns.set_theme(style='darkgrid', context='notebook')

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

    def update(self, val, ax, color_col, color_title):
        ax.cla()
        frame_data = df.loc[df[self.ANIM_COL] == self.anim_frames[val]]

        x = frame_data['x'].values
        y = frame_data['y'].values
        z = frame_data['z'].values
        c = frame_data[color_col].values

        # Normalize opacity
        min_opacity, max_opacity = 0.3, 1.0
        min_frame, max_frame = df[self.OPACITY_COL].min(), df[self.OPACITY_COL].max()
        normalized_opacity = frame_data[self.OPACITY_COL].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity)

        ax.scatter(x, y, z, c=c, cmap='Purples', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)

        # Set limits, axes, and title+subtitle
        ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')
        time_str = pd.to_datetime(frame_data['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        ax.set_title(f"Velocity Plot\nTime: {time_str} (fr: {frame_data['frame'].max()})")

        # Set title and subtitle
        time_str = pd.to_datetime(frame_data['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        ax.set_title(f"{color_title} Plot\nTime: {time_str} (fr: {frame_data['frame'].min()})")

    def viz_slider(self, df: pd.DataFrame, COLOR1_COL: str = 'v', COLOR1_TITLE: str = None, COLOR2_COL: str = 'clstr', COLOR2_TITLE: str = None):
        """ This function generates widget-slider enabled, 3D, colored scatter plots 
        -- side-by-side or single plot based on input columns (COLOR1_COL, 
        COLOR2_COL).  The slider widget is used to control the animation frame, 
        affecting both subplots simultaneously.
        """
        # Check if both columns (COLOR1_COL and COLOR2_COL) are provided and are in the dataframe
        color1_valid = COLOR1_COL in df.columns
        color2_valid = COLOR2_COL in df.columns
        if color2_valid and not color1_valid: # switch color1/color2 if only color2 exists
            COLOR1_COL, COLOR1_TITLE = COLOR2_COL, COLOR2_TITLE
            color2_valid, color1_valid = color1_valid, color2_valid
        COLOR1_TITLE = COLOR1_TITLE or COLOR1_COL
        COLOR2_TITLE = COLOR2_TITLE or COLOR2_COL

        if not color1_valid and not color2_valid:
            raise ValueError("Neither COLOR1_COL nor COLOR2_COL were found in the DataFrame")

        # Print a log of all of the input string variables supplied
        print(f"len(df)={len(df)}    ANIM_COL={self.ANIM_COL}")
        print(f"COLOR1_COL: {COLOR1_COL if color1_valid else 'None':>10}    COLOR1_TITLE: {COLOR1_TITLE if color1_valid else 'None':>16}")
        print(f"COLOR2_COL: {COLOR2_COL if color2_valid else 'None':>10}    COLOR2_TITLE: {COLOR2_TITLE if color2_valid else 'None':>16}")

        VIEW_ANGLE_3D = dict(elev=12, azim=-135, roll=0)
        self.anim_frames = df[self.ANIM_COL].drop_duplicates().sort_values(ascending=True).unique()

        fig = plt.figure()

        if color1_valid and color2_valid:
            print(f"Displaying columns '{COLOR1_COL}' and '{COLOR2_COL}' on two subplots")
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            # set mplot3d view angle once
            ax1.view_init(**self.VIEW_ANGLE_3D)
            ax2.view_init(**self.VIEW_ANGLE_3D)

            # Create colorbars
            c1_min, c1_max = df[COLOR1_COL].min(), df[COLOR1_COL].max()
            sm1 = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(**self.INIT_V_BOUNDS))
            sm1.set_array([])
            cbar1 = fig.colorbar(sm1, ax=ax1, pad=0.1, location='left')
            cbar1.ax.set_title(COLOR1_TITLE)

            c2_min, c2_max = df[COLOR2_COL].min(), df[COLOR2_COL].max()
            sm2 = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(**self.INIT_V_BOUNDS))
            sm2.set_array([])
            cbar2 = fig.colorbar(sm2, ax=ax2, pad=0.1, location='right')
            cbar2.ax.set_title(COLOR2_TITLE)

            def update_fn(val):
                return self.update(val, ax1, COLOR1_COL, COLOR1_TITLE), self.update(val, ax2, COLOR2_COL, COLOR2_TITLE)

        elif color1_valid:
            print(f"Displaying column {COLOR1_COL} on a single subplot")
            ax1 = fig.add_subplot(111, projection='3d')

            # set mplot3d view angle once
            ax1.view_init(VIEW_ANGLE_3D)

            # Create colorbar
            c1_min, c1_max = df[COLOR1_COL].min(), df[COLOR1_COL].max()
            sm1 = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=c1_min, vmax=c1_max))
            sm1.set_array([])
            cbar1 = fig.colorbar(sm1, ax=ax1, pad=0.1, location='right')
            cbar1.ax.set_title(COLOR1_TITLE or COLOR1_COL)

            def update_fn(val):
                return self.update(val, ax1, COLOR1_COL, COLOR1_TITLE or COLOR1_COL)

        else:
            print("None of the provided columns were found in the dataframe")
            return

        # Create the slider widget and set its properties
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(slider_ax, 'Frame', 0, len(self.anim_frames) - 1, valinit=0, valstep=1, valfmt='%d')

        # Connect the slider to the update function
        slider.on_changed(update_fn)

        # Initialize the plot
        update_fn(0)

        # Display the plot
        plt.show()

    '''
    def plot(self, df: pd.DataFrame, COLS: list[str],  COL_TITLES: list[str]):
        if len(COLS) > 4:
            raise ValueError(f"Too many columns in COLS. Reduce it to <= 4. COLS={COLS}")
            pass
    '''
    def plot_subplot(self, ax, val, col, title):
        self.update(val, ax, col, title)

    def plot(self, df: pd.DataFrame, COLS: list[str], COL_TITLES: list[str] = None):
        COL_TITLES = COL_TITLES or [col for col in COLS]

        # Determine subplot layout
        ncols = len(COLS)
        nrows = 1 if ncols <= 2 else 2
        ncols = ncols if ncols <= 2 else 2

        # Create a figure
        fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows), subplot_kw={'projection': '3d'})
        self.anim_frames = df[self.ANIM_COL].drop_duplicates().sort_values(ascending=True).unique()

        # Make sure axs is always a 2D array
        if ncols == 1 and nrows == 1:
            axs = np.array([[axs]])
        elif ncols == 1:
            axs = axs[:, np.newaxis]
        elif nrows == 1:
            axs = axs[np.newaxis, :]

        # Initialize subplots
        for i, (col, title) in enumerate(zip(COLS, COL_TITLES)):
            ax = axs[i//ncols, i%ncols]
            ax.view_init(**self.VIEW_ANGLE_3D)
            self.plot_subplot(ax, 0, col, title)

        # Create the slider widget and set its properties
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(slider_ax, 'Frame', 0, len(self.anim_frames) - 1, valinit=0, valstep=1, valfmt='%d')

        # Connect the slider to the update function
        slider.on_changed(lambda val: [self.plot_subplot(axs[i//ncols, i%ncols], val, col, title) for i, (col, title) in enumerate(zip(COLS, COL_TITLES))])

        # Display the plot
        plt.show()

    def update4sp(self, ax, val, col, title):
        ax.cla()
        frame_data = df.loc[df[self.ANIM_COL] == self.anim_frames[val]]

        x = frame_data['x'].values
        y = frame_data['y'].values
        z = frame_data['z'].values

        if col == 'v':
            c = frame_data[col].values
            cmap = sns.diverging_palette(h_neg=150, h_pos=276, s=73, l=30, sep=1, center='light', as_cmap=True)
            norm = mpl.colors.Normalize(vmin=df[col].min(), vmax=df[col].max())
            linewidth = 0.6
            # Normalize opacity based on 'frame'
            min_opacity, max_opacity = 0.3, 1.0
            min_frame, max_frame = df[self.OPACITY_COL].min(), df[self.OPACITY_COL].max()
            alpha = frame_data[self.OPACITY_COL].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity)

        elif col == 'outlier':
            c = frame_data[col].values
            cmap = 'viridis'
            norm = mpl.colors.Normalize(vmin=df[col].min(), vmax=df[col].max())
            linewidth = 0.7
            alpha = 1

        elif col == 'clstr':
            c = frame_data[col].values
            cmap = mpl.colors.ListedColormap(sns.color_palette("pastel", as_cmap=True))
            norm = mpl.colors.Normalize(vmin=df[col].min(), vmax=df[col].max())
            linewidth = 0.7
            alpha = np.where(frame_data[col] == -1, 1, 0.8)

        elif col == ('bball', 'sent'):
            c = np.where(frame_data['sent'], 2, np.where(frame_data['bball'], 1, 0))
            cmap = mpl.colors.ListedColormap(['white', 'orange', 'red'])
            norm = mpl.colors.Normalize(vmin=0, vmax=2)
            linewidth = 0.7
            alpha = np.where(frame_data['bball'] == True, 0.4, 1.0)
            print( pd.concat([frame_data['bball'], pd.Series(alpha, name='alpha')])[:50] )
            #alpha = 1

        sc = ax.scatter(x, y, z, c=c, cmap=cmap, norm=norm, s=20, marker='o', edgecolor='k', linewidths=linewidth, alpha=alpha)

        # Set limits, axes, and title+subtitle
        ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')
        time_str = pd.to_datetime(frame_data['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        ax.set_title(f"{title} Plot\nTime: {time_str} (fr: {frame_data['frame'].max()})")

    def plot4sp(self, df):
        COLS = ['v', 'outlier', 'clstr', ('bball', 'sent')]
        TITLES = ['Velocity', 'Outlier', 'Cluster', 'Bball/Sent']

        ncols = 2
        nrows = 2

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), subplot_kw={'projection': '3d'})
        self.anim_frames = df[self.ANIM_COL].drop_duplicates().sort_values(ascending=True).unique()

        # Loop over axes to set starting conditions
        for row in range(nrows):
            for col in range(ncols):
                ax = axs[row, col]
                ax.set(xlim=self.bounds['x'], ylim=self.bounds['y'], zlim=self.bounds['z'], xlabel='X', ylabel='Y', zlabel='Z')
                ax.view_init(**self.VIEW_ANGLE_3D)


        # Create Colormaps
        # Creating a pastel color map using seaborn and converting it to a matplotlib color map
        pastel_colors = sns.color_palette("pastel", n_colors=20)
        pastel_cmap = mpl.colors.ListedColormap(pastel_colors.as_hex())

        # Adding black for -1
        pastel_cmap_black = pastel_cmap(np.arange(pastel_cmap.N))
        pastel_cmap_black[-1] = (0, 0, 0, 1)
        pastel_cmap_black = mpl.colors.ListedColormap(pastel_cmap_black)

        # Create colorbars
        # For 'v' subplot
        cmap_v = sns.diverging_palette(h_neg=150, h_pos=276, s=73, l=30, sep=1, center='light', as_cmap=True)
        sm_v = plt.cm.ScalarMappable(cmap=cmap_v, norm=plt.Normalize(vmin=df['v'].min(), vmax=df['v'].max()))
        sm_v.set_array([])
        cbar_v = fig.colorbar(sm_v, ax=axs[0, 0], pad=0.1, location='right')
        cbar_v.ax.set_title("Velocity")

        # For 'outlier' subplot
        sm_outlier = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=df['outlier'].min(), vmax=df['outlier'].max()))
        sm_outlier.set_array([])
        cbar_outlier = fig.colorbar(sm_outlier, ax=axs[0, 1], pad=0.1, location='right')
        cbar_outlier.ax.set_title("Outlier")

        ## For 'clstr' subplot
        #cmap_clstr = sns.color_palette("pastel", as_cmap=True)
        #sm_clstr = plt.cm.ScalarMappable(cmap=cmap_clstr, norm=plt.Normalize(vmin=df['clstr'].min(), vmax=df['clstr'].max()))
        #sm_clstr.set_array([])
        #cbar_clstr = fig.colorbar(sm_clstr, ax=axs[1, 0], pad=0.1, location='right')
        #cbar_clstr.ax.set_title("Cluster")
        # For 'clstr' subplot
        sm_clstr = plt.cm.ScalarMappable(cmap=pastel_cmap_black, norm=plt.Normalize(vmin=df['clstr'].min(), vmax=df['clstr'].max()))
        sm_clstr.set_array([])
        cbar_clstr = fig.colorbar(sm_clstr, ax=axs[1, 0], pad=0.1, location='right')
        cbar_clstr.ax.set_title("Cluster")

        self.sm_v = sm_v
        self.sm_outlier = sm_outlier
        self.sm_clstr = sm_clstr

        # No colorbar needed for the 'bball'/'sent' subplot

        # multi-plot lambda-fn
        update_fn = lambda val: [self.update4sp(axs[i // ncols, i % ncols], val, col, title) for i, (col, title) in enumerate(zip(COLS, TITLES))]

        # Create the slider widget and set its properties
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(slider_ax, 'Frame', 0, len(self.anim_frames) - 1, valinit=0, valstep=1, valfmt='%d')

        # Connect the slider to the update function
        slider.on_changed(update_fn)

        # Initialize the plot
        update_fn(0)

        # Display the plot
        plt.show()



if __name__ == "__main__":
    
    #df = DATA_ALL_TAGGED.loc[(DATA_ALL_TAGGED['file'] == 'log2.json') & (DATA_ALL_TAGGED['dir'] == 'jc')]
    # get all stacks where a DF would've been sent.
    sent_stack_idxs = DATA_ALL_TAGGED['stack'].isin(DATA_ALL_TAGGED.loc[DATA_ALL_TAGGED['sent'] == True, 'stack'].unique())  # noqa: E712
    df = DATA_ALL_TAGGED.loc[sent_stack_idxs]

    viz = Viz()
    #ani = viz.viz_slider(df, COLOR1_COL='v', COLOR1_TITLE='Velocity (m/s)', COLOR2_COL='outlier', COLOR2_TITLE='Outlier')
    #ani = viz.plot(df, COLS=['v', 'outlier', 'clstr', 'bball'])
    ani = viz.plot4sp(df)
