import pandas as pd
import plotly.express as px

def create_scatterplot(scatter_data, output_file):
    fig = px.scatter(title="Interactive Scatter Plot")
    
    for encoded_data, country_region, week in scatter_data:
        df = pd.DataFrame({'x': encoded_data[:, 0], 'y': encoded_data[:, 1], 'Country': country_region})
        
        scatter = fig.add_scatter(x=df['x'], y=df['y'], mode='markers', name=week, text=df['Country'], opacity=0.7)
    
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        showlegend=True,
        legend_title='Week',
    )
    
    fig.write_html(output_file)
    fig.show()
