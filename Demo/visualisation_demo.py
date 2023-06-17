import redis
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Demo.redis_utilis import readRedis
import numpy as np
from SharedParameters.signal_parameters import DATASET_FREQ
from Server.server_params import DECIMATION_FACTOR
from spectrogram import cwt_spectrogram, spectrogram_plot
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, NoNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

global data
r = redis.Redis(host='localhost', port=6379, decode_responses=False)
sampling_frequency = DATASET_FREQ/DECIMATION_FACTOR

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div(html.Img(id='pg_logo', src=app.get_asset_url('pg_logo.jpg'))),
        html.Div(html.Img(id='aitech_logo', src=app.get_asset_url('aitech.jpeg'))),
        html.Div(html.Img(id='eti_logo', src=app.get_asset_url('ETI_logo.jpg')))
    ], id='header', className='image-container'),
    html.H1(id='title', children='BrainBot - EEG Brain-Computer Interface', style={'textAlign':'center'}),
    # html.Div([
    #     html.Button('Left', id='left_control', n_clicks=0, className='control_button'),
    #     html.Button('Right', id='right_control', n_clicks=0, className='control_button'),
    #     html.Button('Forward', id='forward_control', n_clicks=0, className='control_button'),
    #     html.Button('Stop', id='stop_control', n_clicks=0, className='control_button')
    # ], id='control_interface', className='center'),
    dcc.Graph(id='eeg-signal'),
    dcc.Graph(id='spectrogram'),
    dcc.Interval(id='interval_component',
                interval=1000)], id='layout')

@app.callback(
    Output('eeg-signal', 'figure'),
    [Input('interval_component', 'n_intervals')]
)
def update_eeg_graph(value):
    fig = go.Figure()
    sample = readRedis(r, 'eeg_data')
    single_channel = sample[0].flatten()
    data = single_channel
    print(data.shape)
    fig.add_traces([go.Scatter(x=list(range(len(data))), y=data)])
    return fig
@app.callback(
    Output('spectrogram', 'figure'),
    [Input('interval_component', 'n_intervals')]
)
def update_spectrogram(value):
    fig = go.Figure()
    power, times, frequencies, coif = cwt_spectrogram(data, sampling_frequency)
    fig, (ax1, ax2) = plt.subplots(2, 1)

    n_samples = data.shape[0]
    total_duration = n_samples / sampling_frequency
    sampling_times = np.linspace(0, total_duration, n_samples)
    ax1.plot(sampling_times, data, color='b');

    ax1.set_xlim(0, total_duration)
    ax1.set_xlabel('time (s)')
    # ax1.axis('off')
    spectrogram_plot(power, times, frequencies, coif, cmap='jet', norm=LogNorm(), ax=ax2)

    ax2.set_xlim(0, total_duration)
    # ax2.set_ylim(0, 0.5*sampling_frequency)
    ax2.set_ylim(2.0 / total_duration, 0.5 * sampling_frequency)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('frequency (Hz)')
    plotly_fig = mpl_to_plotly(fig)
    plotly_fig.show()
    return plotly_fig
# @app.callback(
#     Output('', ''),
#     [Input('left_control', 'n_clicks'),
#      Input('right_control', 'n_clicks'),
#      Input('forward_control', 'n_clicks'),
#      Input('stop_control', 'n_clicks')]
# )
# def update_control():
#     pass

def main():
    app.run_server(debug=True)

if __name__ == "__main__":
    main()