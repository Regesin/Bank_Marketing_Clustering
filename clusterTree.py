import pandas as pd
import plotly.express as px

def create_cluster_tree(df):

    fig = px.sunburst(df, path=['education', 'job', 'marital', 'y'], values='cluster')

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html