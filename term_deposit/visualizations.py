import plotly.express as px


def plot_3d_interactive(data, clusters, title):
    fig = px.scatter_3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        color=clusters.astype(str),
        title=f'3D {title} Visualization',
        labels={'x': f'{title} Component 1', 'y': f'{title} Component 2', 'z': f'{title} Component 3'}
    )
    fig.show()