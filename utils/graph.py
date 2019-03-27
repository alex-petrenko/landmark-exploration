import io
from os.path import join

import networkx as nx
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.utils import ensure_dir_exists


def parse_layout(nx_graph, layout):
    if layout == 'pos':
        pos = nx.get_node_attributes(nx_graph, 'pos')
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(nx_graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(nx_graph)
    else:
        raise Exception('Unknown graph layout: ' + layout)
    return pos


def plot_graph(nx_graph, layout, map_img=None, node_size=80):
    if layout == 'pos':
        for node_name in nx_graph.nodes:
            pos = nx_graph.node[node_name].get('pos')
            if pos is None:
                layout = 'kamada_kawai'
                break

    pos = parse_layout(nx_graph, layout)

    figure = plt.gcf()
    figure.clear()
    if map_img is not None:
        # ax.imshow(map_img)
        import numpy as np
        map_img = np.array(map_img).astype(np.float) / 255
        height = 0
        figure.figimage(map_img, 0, height)
    nx.draw(
        nx_graph, pos, node_size=node_size, node_color=list(range(len(nx_graph.nodes))), edge_color='#cccccc',
        cmap=plt.cm.get_cmap('plasma'), with_labels=True, font_color='#00ff00', font_size=7,
    )
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    return figure


def visualize_graph_tensorboard(nx_graph, tag, layout='pos', map_img=None):
    figure = plot_graph(nx_graph, layout, map_img=map_img)
    w, h = figure.canvas.get_width_height()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    graph_image_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(), height=h, width=w)
    graph_summary = tf.Summary.Value(tag=tag, image=graph_image_summary)

    summary = tf.Summary(value=[graph_summary])
    figure.clear()
    return summary


def visualize_graph_html(nx_graph, output_dir=None, title_text='', layout='kamada_kawai', should_show=False):
    """
    This method visualizes a NetworkX graph using Bokeh.

    :param nx_graph: NetworkX graph with node attributes containing image filenames.
    :param output_dir: Optional output directory for saving html.
    :param title_text: String to be displayed above the visualization.
    :param layout: Which layout function to use.
    :param should_show: Open the browser to look at the graph.
    """
    from bokeh import palettes
    from bokeh.io import output_file, show
    from bokeh.models import Circle, HoverTool, MultiLine, Plot, Range1d, TapTool
    # noinspection PyProtectedMember
    from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, NodesOnly

    pos = parse_layout(nx_graph, layout)

    hover_tool = HoverTool(tooltips='<img src="@imgs" height="200" alt="@imgs" width="200"></img>', show_arrow=False)

    plot = Plot(plot_width=800, plot_height=800, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    if title_text != '':
        plot.title.text = title_text
    plot.title.align = 'center'
    plot.min_border = 0
    plot.outline_line_color = None

    plot.add_tools(hover_tool, TapTool())
    plot.toolbar.logo = None
    plot.toolbar_location = None

    graph_renderer = from_networkx(nx_graph, pos)

    graph_renderer.node_renderer.data_source.data['imgs'] = [n[1]['img'] for n in nx_graph.nodes(data=True)]

    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color=palettes.Spectral4[0], line_color=None)
    graph_renderer.node_renderer.selection_glyph = Circle(size=10, fill_color=palettes.Spectral4[2], line_color=None)
    graph_renderer.node_renderer.hover_glyph = Circle(size=10, fill_color=palettes.Spectral4[1], line_color=None)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color='#CCCCCC', line_alpha=0.8, line_width=1.5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=palettes.Spectral4[2], line_width=2)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesOnly()

    plot.renderers.append(graph_renderer)

    if output_dir:
        ensure_dir_exists(output_dir)
        output_file(join(output_dir, 'visualize_graph.html'))

    if should_show:
        show(plot)
