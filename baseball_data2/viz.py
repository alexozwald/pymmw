import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from matplotlib import use
from matplotlib.cm import ScalarMappable

plt.rcParams['animation.ffmpeg_path'] = "/opt/homebrew/bin/ffmpeg"
FFwriter = animation.FFMpegWriter(extra_args=['-c:v', 'h264_videotoolbox'])

## viz1() -- Plotly ############################################################

def viz1(df_stack: pd.DataFrame, style: str = '3d'):
    bool_no_clstr = 'clstr' not in df_stack.columns.to_list()
    if bool_no_clstr:
        df_stack.loc[:,'clstr'] = 0

    x_max = df_stack['x'].max()
    y_max = df_stack['y'].max()
    z_max = df_stack['z'].max()
    v_max = df_stack['v'].max()
    x_max = 8 if x_max <= 8 else x_max
    y_max = 8 if y_max <= 8 else y_max
    z_max = 8 if z_max <= 8 else z_max
    v_max = 8 if v_max <= 8 else v_max

    if style == '3d':
        fig = px.scatter_3d(df_stack, 'x', 'y', 'z', animation_frame='stack', color='clstr', hover_data=['v'],
            range_x=[-2,x_max], range_y=[-2,y_max], range_z=[-2,z_max])
    elif style == 'vel':
        fig = px.scatter_3d(df_stack, 'x', 'y', 'z', animation_frame='stack', color=df_stack['v'].abs(), hover_data=['v'],
            range_x=[0,x_max], range_y=[0,y_max], range_z=[-2,z_max])
    fig.show()


## viz2() -- One 3D Colored Mesh (animated) -- BROKEN ##########################

def viz2(df: pd.DataFrame, color_col: str, /, anim_frame: str = 'stack'):
    use('MacOSX')
    # Set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the scatter plot with the first frame of the data
    sc = ax.scatter(df['x'][df[anim_frame] == df[anim_frame].min()],
                    df['y'][df[anim_frame] == df[anim_frame].min()],
                    df['z'][df[anim_frame] == df[anim_frame].min()],
                    c=df[color_col][df[anim_frame] == df[anim_frame].min()],
                    cmap='viridis')

    # Define the update function
    def update(frame):
        # Clear the previous frame
        ax.clear()

        # Plot the new frame
        sc = ax.scatter(df['x'][df[anim_frame] == frame],
                        df['y'][df[anim_frame] == frame],
                        df['z'][df[anim_frame] == frame],
                        c=df[color_col][df[anim_frame] == frame],
                        cmap='viridis')

        # Set the axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Set the title
        ax.set_title(f'Time = {frame}')

        return sc,

    # Create the animation
    frames = df[anim_frame].unique()
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50, repeat=False)
    
    # Show the animation
    plt.show()


## viz3() -- Side-by-Side 3D Mesh Plots (animated) #############################
## viz3() ######################################################################

def viz3(df: pd.DataFrame, color_col_1: str, color_col_2: str, /, anim_frame: str = 'stack', SAVE: str = 'mp4'):
    #use('inline')
    # Set up the figure
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Initialize the scatter plots with the first frame of the data
    sc1 = ax1.scatter(df['x'][df[anim_frame] == df[anim_frame].min()],
                    df['y'][df[anim_frame] == df[anim_frame].min()],
                    df['z'][df[anim_frame] == df[anim_frame].min()],
                    c=df[color_col_1][df[anim_frame] == df[anim_frame].min()],
                    cmap='viridis')
    sc2 = ax2.scatter(df['x'][df[anim_frame] == df[anim_frame].min()],
                    df['y'][df[anim_frame] == df[anim_frame].min()],
                    df['z'][df[anim_frame] == df[anim_frame].min()],
                    c=df[color_col_2][df[anim_frame] == df[anim_frame].min()],
                    cmap='viridis')

    # Set Color Bars (static for whole time)
    # Add colorbar to Subplot 1
    sm1 = ScalarMappable(cmap='viridis')
    sm1.set_array(df[color_col_1])
    fig.colorbar(sm1, ax=ax1)

    # Add colorbar to Subplot 2
    sm2 = ScalarMappable(cmap='viridis')
    sm2.set_array(df[color_col_2])
    fig.colorbar(sm2, ax=ax2)

    # Define the update function
    def update(frame):
        # Clear the previous frames
        ax1.clear()
        ax2.clear()

        # Plot the new frames
        sc1 = ax1.scatter(df['x'][df[anim_frame] == frame],
                        df['y'][df[anim_frame] == frame],
                        df['z'][df[anim_frame] == frame],
                        c=df[color_col_1][df[anim_frame] == frame],
                        cmap='viridis')
        sc2 = ax2.scatter(df['x'][df[anim_frame] == frame],
                        df['y'][df[anim_frame] == frame],
                        df['z'][df[anim_frame] == frame],
                        c=df[color_col_2][df[anim_frame] == frame],
                        cmap='viridis')

        # Set the axis labels
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        # Set the titles
        ax1.set_title(f'Frame = {frame}, {color_col_1}')
        ax2.set_title(f'Frame = {frame}, {color_col_2}')
        
        return sc1, sc2

    # Create the animation
    frames = df[anim_frame].unique()
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=20, repeat=False)

    if 'x265' in SAVE or 'hevc' in SAVE:
        anim.save("./SAVED_PLOT.mp4", writer=animation.FFMpegWriter(extra_args=['-c:v', 'hevc_videotoolbox']))
    elif 'x264' in SAVE or 'mp4' in SAVE:
        anim.save("./SAVED_PLOT.mp4", writer=animation.FFMpegWriter(extra_args=['-c:v', 'h264_videotoolbox']))
    elif SAVE == 'gif':
        anim.save("./SAVED_PLOT.gif")
    else:
        try:
            anim.save("./SAVED_PLOT.mp4", writer=animation.FFMpegWriter())
        except Exception:
            anim.save("./SAVED_PLOT.gif")

    # Show the animation
    plt.show()