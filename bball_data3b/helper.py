import subprocess
import pandas as pd
import numpy as np
import orjson
import hdbscan
#from mayavi import mlab
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def get_repo_dir():
    import subprocess
    repo_dir = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    return repo_dir

def orjson_to_df(filepath: str|list[str]) -> pd.DataFrame:
    from os.path import isfile
    df = pd.DataFrame()
    filepath = [filepath] if isinstance(filepath, str) else filepath
    
    if not isinstance(filepath,list):
        raise TypeError(f"filepaths")
    for f in filepath:
        if not isfile(f):
            raise FileNotFoundError("your file dont exist homie")

    for f in filepath:
        with open(f, 'rb') as f:
            for line in f.readlines():
                d = orjson.loads(line)
                __df__ = pd.DataFrame(d['xyzv'], columns=['x','y','z','v']).assign(ts=d['ts'], frame=d['frame'])
                df = pd.concat([df, __df__[['ts','frame','x','y','z','v']]])
    return df

def mk_stacks(df: pd.DataFrame, STACK_HEIGHT: int, STACK_ON='frame', SORT_COLS_LIST=['frame']) -> pd.DataFrame:
    df_out = df.assign(stack=df.apply(lambda r: np.arange(r[STACK_ON], r[STACK_ON]+(STACK_HEIGHT+1), dtype=np.int64), axis=1)).explode('stack').reset_index(drop=True)#.sort_values([SORT_COLS_LIST])
    df_out = df_out.loc[df_out[STACK_ON].isin(df[STACK_ON].unique())]
    return df_out

def stacks_above_v(df_in: pd.DataFrame, TST_VEL: int|float, N_STACKS: int = 20):
    """ Get only stacks where velocity exceeds threshold. """
    df_out = df_in.copy(deep=True)    
    # Id all rows above threshold
    rows_over_Vms = df_out.loc[df_out['v'].abs() >= TST_VEL].copy()
    # Separate stacks from ID'd rows
    over_Vms_stacks = rows_over_Vms['stack'].unique()
    # Parse DF to include stacks in ID'd list
    df_out = df_out.loc[df_out['stack'].isin(over_Vms_stacks)]
    # Only include the first N stacks for each frame that has a threshold velocity
    f_stacks = df_out.loc[df_out['v'].abs() >= TST_VEL][['frame','stack']].drop_duplicates()
    N_f_stacks = f_stacks.groupby('frame').apply(lambda x: x.iloc[:N_STACKS]).reset_index(drop=True)['stack'].unique()
    # Filter DF again
    df_out = df_out.loc[df_out['stack'].isin(N_f_stacks)]
    return df_out

def stack_over_v(df_in: pd.DataFrame, TST_VEL: int|float): #, N_STACKS: int = 20):
    """ Get only stacks where velocity exceeds threshold. """
    df_out = df_in.copy(deep=True)    
    rows_over_Vms = df_out.loc[df_out['v'].abs() >= TST_VEL]
    over_Vms_stacks = rows_over_Vms['stack'].unique()
    df_out = df_out.loc[df_out['stack'].isin(over_Vms_stacks)].sort_values(['stack','frame','v'])
    return df_out

def do_hdbscan(df_in: pd.DataFrame, MIN_CLSTRS: int, NORM: bool = False) -> pd.DataFrame:
    from warnings import filterwarnings; filterwarnings('ignore', category=RuntimeWarning);

    df_out = pd.DataFrame()
    df_in = df_in.copy(deep=True)
    df_in['v'] = df_in['v'].abs()

    # add normalization (done by column per-stack)
    if NORM:
        df[['stack','x','y','z','v']] = df[['stack','x','y','z','v']].groupby('stack').apply(lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0).drop('stack',axis=1).reset_index('stack')

    for stack,g in df_in.groupby('stack'):
        scan = hdbscan.HDBSCAN(min_cluster_size=MIN_CLSTRS, algorithm='best', alpha=1.0, metric='euclidean')
        g = g.copy(deep=True)
        model = scan.fit(g[['x','y','z','v']])
        g = g.assign(outlier=model.outlier_scores_, clstr=model.labels_)
        df_out = pd.concat([df_out, g])

    df_out['outlier'] = df_out['outlier'].fillna(0)
    
    return df_out

def viz_time(df: pd.DataFrame, ANIM_COL: str = 'stack', interval=0.01, COLOR1_COL: str = 'v', COLOR1_TITLE: str = 'Velocity (m/s)'):
    """ Vizualize Animated, Colored, 3D Scatter plot. Interval-based animation. """
    anim_frames = np.sort(df[ANIM_COL].unique())
    max_bounds = {'x': (df['x'].min(), df['x'].max()),
                  'y': (df['y'].min(), df['y'].max()),
                  'z': (df['z'].min(), df['z'].max())}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colorbar settings
    vmin, vmax = df[COLOR1_COL].min(), df[COLOR1_COL].max()
    sm = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.ax.set_title(COLOR1_TITLE)

    # Set 3D viewing angle
    ax.view_init(elev=df['z'].quantile(0.8), azim=-135)

    for frame in anim_frames:
        ax.clear()
        frame_data = df.loc[df[ANIM_COL] == frame]

        x = frame_data['x'].values
        y = frame_data['y'].values
        z = frame_data['z'].values
        v = frame_data[COLOR1_COL].values

        # Update colorbar limits if necessary
        frame_vmin, frame_vmax = v.min(), v.max()
        if frame_vmin < vmin or frame_vmax > vmax:
            vmin, vmax = min(vmin, frame_vmin), max(vmax, frame_vmax)
            sm.set_clim(vmin, vmax)

        # Calculate the min and max frame values
        min_frame, max_frame = df['frame'].min(), df['frame'].max()

        # Normalize the frame values between min_opacity and max_opacity
        min_opacity, max_opacity = 0.3, 1.0
        normalized_opacity = frame_data['frame'].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity)

        # Use the normalized_opacity values as the alpha parameter for the scatter plot
        sc = ax.scatter(x, y, z, c=v, cmap='viridis', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)

        ax.set_xlim(*max_bounds['x'])
        ax.set_ylim(*max_bounds['y'])
        ax.set_zlim(*max_bounds['z'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set title and subtitle
        time_str = pd.to_datetime(frame_data['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        ax.set_title(f"Radial View of Point Cloud\nTime: {time_str}")

        plt.draw()
        plt.pause(interval)

    plt.show()

def viz_slider(df: pd.DataFrame, ANIM_COL: str = 'stack', COLOR1_COL: str = 'v', COLOR1_TITLE: str = None,
                    COLOR2_COL: str = 'clstr', COLOR2_TITLE: str = None, OPACITY_COL: str = 'frame'):
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
    print(f"len(df)={len(df)}    ANIM_COL={ANIM_COL}")
    print(f"COLOR1_COL: {COLOR1_COL if color1_valid else 'None':>10}    COLOR1_TITLE: {COLOR1_TITLE if color1_valid else 'None':>16}")
    print(f"COLOR2_COL: {COLOR2_COL if color2_valid else 'None':>10}    COLOR2_TITLE: {COLOR2_TITLE if color2_valid else 'None':>16}")

    VIEW_ANGLE_3D = dict(elev=12, azim=-135, roll=0)
    anim_frames = np.sort(df[ANIM_COL].unique())
    max_bounds = {'x': (df['x'].min(), df['x'].max()),
                  'y': (df['y'].min(), df['y'].max()),
                  'z': (df['z'].min(), df['z'].max())}

    def update(val, ax, color_col, color_title):
        #ax.clear()
        ax.cla()
        frame_data = df.loc[df[ANIM_COL] == anim_frames[val]]

        x = frame_data['x'].values
        y = frame_data['y'].values
        z = frame_data['z'].values
        c = frame_data[color_col].values

        # Normalize opacity
        min_opacity, max_opacity = 0.3, 1.0
        min_frame, max_frame = df[OPACITY_COL].min(), df[OPACITY_COL].max()
        normalized_opacity = frame_data[OPACITY_COL].sub(min_frame).div(max_frame - min_frame).mul(max_opacity - min_opacity).add(min_opacity)

        sc = ax.scatter(x, y, z, c=c, cmap='Purples', s=20, marker='o', edgecolor='k', linewidths=0.6, alpha=normalized_opacity)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(*max_bounds['x'])
        ax.set_ylim(*max_bounds['y'])
        ax.set_zlim(*max_bounds['z'])

        # Set title and subtitle
        time_str = pd.to_datetime(frame_data['ts'].max(), unit='ms').strftime('%X.%f')[:-3]
        ax.set_title(f"{color_title} Plot\nTime: {time_str} (fr: {frame_data['frame'].min()})")

    fig = plt.figure()

    if color1_valid and color2_valid:
        print(f"Displaying columns '{COLOR1_COL}' and '{COLOR2_COL}' on two subplots")
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # set mplot3d view angle once
        ax1.view_init(**VIEW_ANGLE_3D)
        ax2.view_init(**VIEW_ANGLE_3D)

        # Create colorbars
        c1_min, c1_max = df[COLOR1_COL].min(), df[COLOR1_COL].max()
        sm1 = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=c1_min, vmax=c1_max))
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, ax=ax1, pad=0.1, location='left')
        cbar1.ax.set_title(COLOR1_TITLE)

        c2_min, c2_max = df[COLOR2_COL].min(), df[COLOR2_COL].max()
        sm2 = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=c2_min, vmax=c2_max))
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=ax2, pad=0.1, location='right')
        cbar2.ax.set_title(COLOR2_TITLE)

        update_fn = lambda val: (update(val, ax1, COLOR1_COL, COLOR1_TITLE),
                                 update(val, ax2, COLOR2_COL, COLOR2_TITLE))

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

        update_fn = lambda val: update(val, ax1, COLOR1_COL, COLOR1_TITLE or COLOR1_COL)

    else:
        print(f"None of the provided columns were found in the dataframe")
        return

    # Create the slider widget and set its properties
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider = Slider(slider_ax, 'Frame', 0, len(anim_frames) - 1, valinit=0, valstep=1, valfmt='%d')

    # Connect the slider to the update function
    slider.on_changed(update_fn)

    # Initialize the plot
    update_fn(0)

    # Display the plot
    plt.show()

repo_dir = get_repo_dir()

if __name__ == "__main__":

    # load data
    #df_lab_lobs = pd.read_csv('./data/03.30-lab_lobs.csv', engine='pyarrow')
    df_jc_pitch = pd.read_csv('./data/03.21-jc_pitch.csv', engine='pyarrow')
    #df = mk_stacks(df_jc_pitch, 15, 'frame', ['stack','frame','ts'])

    #df_ao_pitch = pd.read_csv('./data/03.31-ao_pitch.csv', engine='pyarrow')    
    #df = mk_stacks(df_ao_pitch, 15, 'frame', ['stack','frame','ts'])

    print("=> data loaded + stacks made!")

    #df = mk_stacks(df_jc_pitch, 10, 'frame', ['stack','frame','ts'])
    #dd = stacks_above_v(df, 20)
    #ani = viz_slider(d, COLOR1_COL='v', COLOR1_TITLE='Velocity (m/s)', COLOR2_COL='frame', COLOR2_TITLE='Frame #', OPACITY_COL='frame')

    df = orjson_to_df(['/Users/alex/Documents/neu/linuxmnt/_pymmw/bball_data3b/data/2023.03.21_jc/log2.json'])
    df = df.loc[(df['v'] >= 0) & (df['frame'] <= 1500)]
    df = mk_stacks(df, 20)
    df = stack_over_v(df, 17)
    dd = do_hdbscan(df, 4, False)

    ani = viz_slider(dd, COLOR1_COL='v', COLOR1_TITLE='Velocity (m/s)', COLOR2_COL='outlier', COLOR2_TITLE='Outlier', OPACITY_COL='frame')

    

    print("=> viz done!!")
