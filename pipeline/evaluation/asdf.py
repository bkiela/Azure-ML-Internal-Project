import pandas as pd
import plotly.graph_objects as go
import asyncio
import random

def update_plot(fig, new_data):
    fig.data[0].x = new_data['x']
    fig.data[0].y = new_data['y']

    return fig

fig = go.Figure()
fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])
fig.show()

async def main():
    global fig
    chart_counter = 0
    
    while True:
        new_data = {'x': [random.uniform(0, 10) for _ in range(3)],
                    'y': [random.uniform(0, 10) for _ in range(3)]}
        
        chart_counter += 1
        fig = update_plot(fig, new_data)
        fig.show()
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
