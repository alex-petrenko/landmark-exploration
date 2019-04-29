import copy
import io
from os.path import join

import cv2
import networkx as nx
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

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


def plot_graph(
        nx_graph, layout, map_img=None, node_size=80, limits=(0, 0, 1856, 1856), topological_map=False, **kwargs,
):
    """Plot the graph with a map image overlaid on it. Give coordinate limits of map in :limits:"""
    if layout == 'pos':
        for node_name in nx_graph.nodes:
            pos = nx_graph.node[node_name].get('pos')
            if pos is None:
                layout = 'kamada_kawai'
                break

    pos = parse_layout(nx_graph, layout)

    if map_img is None:
        figure = plt.gcf()
        figure.clear()
    else:
        map_img = cv2.resize(map_img, dsize=None, fx=2, fy=2)
        width, height = map_img.shape[:2]

        dpi = 120  # can be changed
        figure = plt.figure(num=2, figsize=(height / dpi, width / dpi), dpi=dpi, facecolor='none', edgecolor='k')
        figure.clear()
        figure.figimage(map_img, 0, 0)
        plt.xlim(limits[0], limits[2])
        plt.ylim(limits[1], limits[3])

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.patch.set_facecolor('none')

    draw_func = draw_map if topological_map else nx.draw
    draw_func(
        nx_graph, pos, node_size=node_size, node_color=list(range(len(nx_graph.nodes))), edge_color='#cccccc',
        cmap=plt.cm.get_cmap('plasma'), with_labels=True, font_color='#00ff00', font_size=7, **kwargs,
    )
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    return figure


def draw_map(g, pos=None, ax=None, **kwds):
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor('w')
    if ax is None:
        # noinspection PyProtectedMember
        if cf._axstack() is None:
            ax = cf.add_axes((0, 0, 1, 1))
        else:
            ax = cf.gca()

    if 'with_labels' not in kwds:
        kwds['with_labels'] = 'labels' in kwds

    if pos is None:
        pos = nx.drawing.spring_layout(g)  # default to spring layout

    sparse_map = kwds.get('is_sparse', False)
    only_draw_edges = not sparse_map

    if not only_draw_edges:
        nx.draw_networkx_nodes(g, pos, **kwds)

    edges_normal = [(u, v) for (u, v, d) in g.edges(data=True) if not d.get('loop_closure')]
    edges_shortcuts = [(u, v) for (u, v, d) in g.edges(data=True) if d.get('loop_closure')]

    nx.draw_networkx_edges(g, pos, edgelist=edges_normal, arrows=False, **kwds)

    shortcuts_kwargs = copy.deepcopy(kwds)
    shortcuts_kwargs['alpha'] = 0.4
    shortcuts_kwargs['edge_color'] = 'r'
    shortcuts_kwargs['width'] = 2
    nx.draw_networkx_edges(
        g, pos, edgelist=edges_shortcuts, arrows=False, **shortcuts_kwargs,
    )

    with_labels = not only_draw_edges
    if with_labels:
        nx.draw_networkx_labels(g, pos, **kwds)

    ax.set_axis_off()
    # noinspection PyUnresolvedReferences
    plt.draw_if_interactive()


def visualize_graph_tensorboard(nx_graph, tag, layout='pos', map_img=None, coord_limits=None, is_sparse=False):
    figure = plot_graph(
        nx_graph, layout, map_img=map_img, limits=coord_limits, topological_map=True, is_sparse=is_sparse,
    )
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
