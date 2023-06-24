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
    html.H1(id='label', children='STOP', style={'textAlign':'left', 'color': 'blue'}),
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
    [Output('eeg-signal', 'figure'),
    Output('spectrogram', 'figure'),
    Output('label', 'children')],
    [Input('interval_component', 'n_intervals')]
)
def update_eeg_graph(value):
    global data
    signal_fig = go.Figure()
    sample = readRedis(r, 'eeg_data')
    label = r.get('label').decode('utf-8')
    print(label)
    single_channel = sample[0].flatten()
    data = single_channel
    signal_fig.add_traces([go.Scatter(x=list(range(len(data))), y=data)])
    power, times, frequencies, coif = cwt_spectrogram(data, sampling_frequency)
    trace = [go.Heatmap(x=times, y=frequencies, z=power, colorscale='Jet')]
    layout = go.Layout(
        title='Spectrogram',
        yaxis=dict(title='Frequency'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    spectrogram = go.Figure(data=trace, layout=layout)
    return signal_fig, spectrogram, label

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