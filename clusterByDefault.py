import pandas as pd
import plotly.graph_objects as go

def stacked_bar_chart_default(df):
    # Group the dataframe by default and cluster
    df_grouped = df.groupby(['default', 'cluster']).size().unstack()

    # Create a stacked bar chart
    fig = go.Figure(data=[
        go.Bar(x=df_grouped.index, y=df_grouped[cluster], name=cluster) for cluster in df_grouped.columns
    ])

    # Update the layout with labels and title
    fig.update_layout(
        barmode='stack',
        xaxis_title='default',
        yaxis_title='Count',
        title='Stacked Bar Chart: Cluster per default'
    )

    # Convert the plot to HTML
    plot_html = fig.to_html(full_html=False)

    return plot_html