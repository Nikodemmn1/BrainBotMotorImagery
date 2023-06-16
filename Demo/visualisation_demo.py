import redis
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
data = []
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div(html.Img(id='pg_logo', src=app.get_asset_url('pg_logo.jpg'))),
        html.Div(html.Img(id='aitech_logo', src=app.get_asset_url('aitech.jpeg')))
    ], id='header'),
    html.H1(id='title', children='BrainBot - EEG Brain-Computer Interface', style={'textAlign':'center'}),
    html.Div([
        html.Button('Left', id='left_control', n_clicks=0, className='control_button'),
        html.Button('Right', id='right_control', n_clicks=0, className='control_button'),
        html.Button('Forward', id='forward_control', n_clicks=0, className='control_button'),
        html.Button('Stop', id='stop_control', n_clicks=0, className='control_button')
    ], id='control_interface', className='center'),
    dcc.Graph(id='graph-content'),
    dcc.Interval(id='interval_component',
                interval=1)], id='layout')

@app.callback(
    Output('graph-content', 'figure'),
    [Input('interval_component', 'n_intervals')]
)
def update_eeg_graph(value):
    fig = go.Figure()
    sample = r.xrevrange('eeg_data', '+', '-', 1)[0][1]['data']
    data.append(sample)
    fig.add_traces([go.Scatter(x=list(range(len(data))), y=data)])
    return fig
@app.callback(
    Output('', ''),
    [Input('left_control', 'n_clicks'),
     Input('right_control', 'n_clicks'),
     Input('forward_control', 'n_clicks'),
     Input('stop_control', 'n_clicks')]
)
def update_control():
    pass

def main():
    app.run_server(debug=True)

if __name__ == "__main__":
    main()