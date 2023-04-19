#!/usr/bin/env python3
import os.path
import threading

from collections import deque

import click

from capstone.data import HDBSCAN_Analysis, PYTORCH_GNN_Analaysis
from capstone.plot import Plot3Dv
from capstone.tools import DebugThreadPoolExecutor, ZmqListener, tprint

@click.command(context_settings=dict(help_option_names=['-h','--help'], max_content_width=90))
@click.option("-m", "--mps", type=float, default=17, show_default=True, help="Set the minimum cutoff speed (m/s) at which outliers are tested to be baseballs")
@click.option("-o", "--outlier-score", type=click.FloatRange(0.0, 1.0), default=0.75, show_default=True, help="Set the cutoff minimum outllier score to test for baseballs with")
@click.option("-N", "--normalize", is_flag=True, default=False, show_default=True, help="Flag to normalize the dimensional readings before running HDBSCAN algo!")
@click.option("-a", "--alpha", type=click.FloatRange(0.0, 2.0), default=1.00, show_default=True, help="Set the cutoff minimum outllier score to test for baseballs with")
@click.option("-n", "--min-cluster-size", type=int, default=3, show_default=True, help="Set frame lag for the Live 3D plot")
@click.option("-fSTACK", "--frame-stack-size", type=int, default=20, show_default=True, help="Set the frame stack height of temporal points that will be concat'd")
@click.option("-fB2B", "--min-frames-ball2ball", type=int, default=7, show_default=True, help="Set the minimum # of frames between baseballs (to separate clusters)")
@click.option("-p", "--plot/--no-plot", is_flag=True, default=False, show_default=True, help="▲▽ Enable Live 3D Plot △▼")
@click.option("-fLAG", "--frame-lag", type=int, default=20, show_default=True, help="Set frame lag for the Live 3D plot")
@click.option("-i", "--internet/--no-internet", is_flag=True, default=False, show_default=True, help="Enable sending actual http.POSTs to the server")#default=True
@click.option("-v", "--verbose", count=True, default=False, help="Increase Output Verbosity")
def main(mps: float, outlier_score: float, normalize: bool, alpha: float, min_cluster_size: int, frame_stack_size: int, min_frames_ball2ball: int, plot: bool, frame_lag: int, internet: bool, verbose: int, **kwargs):
    """Launch ZMQ-Reader service to parse, analyze, & process . Plot data live in a colored, 3D Scatter plot.
        
    \b
    Requisite Launch Commands: 
    MacOS:      ./pymmw.py -c /dev/tty.SLAB_USBtoUART -d /dev/tty.SLAB_USBtoUART3
    Linux:      ./pymmw.py -c /dev/ttyUSB0 -d /dev/ttyUSB1
    TEST_SRVR:  python REPO_BASE/bball_data3b/zmq_server.py

    \b
    References:
    [1] HDBSCAN Parameter Selection - https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    [2] TI mmWave Demo - https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/
    [3] TI mmWave Sensing Estimator - https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.2.0/

    \b
    Tested Config:
    ./srvr.py  --no-internet --mps 17 --outlier-score 0.75 --alpha 1.0 --min-cluster-size 3 -fB2B 7 -fSTACK 20 --plot -fLAG 20 [-vv]
    
    """

    # generate
    zmq_listener = ZmqListener(verbose=verbose)
    data_analysis = HDBSCAN_Analysis(
        internet_input=internet,
        mps_input=mps,
        outlier_score_input=outlier_score,
        normalize_input=normalize,
        alpha_score_input=alpha,
        minimum_cluster_size_input=min_cluster_size,
        fr_stack_size_input=frame_stack_size,
        min_fr_b2b_input=min_frames_ball2ball,
        verbose=verbose)
    analysis_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=False, verbose=verbose)
    handle_executor = DebugThreadPoolExecutor(max_workers=1, non_blocking=True, verbose=verbose)
    handle_future, analysis_future = None, None

    def generate():
        """  """

        for df_line in zmq_listener.listen():
            zmq_listener.append_to_buffer(df_line)

            if plot:
                plotter.add_pts(df_line)

            # If analysis_future has a response, submit handle_detections task
            if analysis_future and analysis_future.done():
                input_data_for_handle = analysis_future.result()
                handle_executor.submit(data_analysis.handle_detections, *input_data_for_handle)

            # Submit exec_df_get_bball task when there's no ongoing analysis
            if analysis_future is None or analysis_future.done():
                analysis_future = analysis_executor.submit(data_analysis.exec_df_get_bball, zmq_listener.buffer)
                zmq_listener.buffer = pd.DataFrame()



    if plot:
        plotter = Plot3Dv(frame_lag=frame_lag)

    if verbose > 0:
        tprint(f"=({data_analysis.time})==> Entering loop...")

    for df_line in zmq_listener.listen():
        zmq_listener.append_to_buffer(df_line)

        if plot:
            plotter.add_pts(df_line)

        # If analysis_future has a response, submit handle_detections task
        if analysis_future and analysis_future.done():
            input_data_for_handle = analysis_future.result()
            handle_executor.submit(data_analysis.handle_detections, *input_data_for_handle)

        # Submit exec_df_get_bball task when there's no ongoing analysis
        if analysis_future is None or analysis_future.done():
            analysis_future = analysis_executor.submit(data_analysis.exec_df_get_bball, zmq_listener.buffer)
            zmq_listener.buffer = pd.DataFrame()


if __name__ == "__main__":
    main()
