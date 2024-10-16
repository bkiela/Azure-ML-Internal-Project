import plotly.graph_objects as go



def update_plot(fig, new_data):
    fig.data[0].x = new_data['x']
    fig.data[0].y = new_data['y']

    return fig

fig = go.Figure()
fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])

new_data = {'x': [4, 5, 6], 'y': [7, 8, 9]}

fig.show()

fig.close()


fig = update_plot(fig, new_data)

fig.show()
