from os.path import join

from matplotlib import pyplot as plt
import networkx as nx
import tensorboardX

from utils.utils import ensure_dir_exists, experiments_dir, project_root


def visualize_graph_tensorboard(graph, log_dir=experiments_dir(),  key='matplotlib/figure'):
    nx.draw(graph, nx.kamada_kawai_layout(graph), node_size=50, node_color=list(graph.nodes), edge_color='#cccccc', cmap=plt.cm.get_cmap('plasma'))
    writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    writer.add_figure(key, plt.gcf())
    writer.close()


def visualize_graph_html(graph, title_text='', layout='kamada_kawai'):
    """
    This method visualizes a NetworkX graph using Bokeh.

    :param graph: NetworkX graph with node attributes containing image filenames.
    :param title_text: String to be displayed above the visualization.
    :param layout: Which layout function to use.
    """
    from bokeh import palettes
    from bokeh.io import output_file, show
    from bokeh.models import Circle, HoverTool, MultiLine, Plot, Range1d, TapTool
    # noinspection PyProtectedMember
    from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, NodesOnly

    if layout == 'pos':
        pos = nx.get_node_attributes(graph, 'pos')
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)

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

    graph_renderer = from_networkx(graph, pos)

    graph_renderer.node_renderer.data_source.data['imgs'] = [n[1]['img'] for n in graph.nodes(data=True)]

    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color=palettes.Spectral4[0], line_color=None)
    graph_renderer.node_renderer.selection_glyph = Circle(size=10, fill_color=palettes.Spectral4[2], line_color=None)
    graph_renderer.node_renderer.hover_glyph = Circle(size=10, fill_color=palettes.Spectral4[1], line_color=None)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color='#CCCCCC', line_alpha=0.8, line_width=1.5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=palettes.Spectral4[2], line_width=2)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesOnly()

    plot.renderers.append(graph_renderer)

    output_dir = join(project_root(), '.visualize')
    ensure_dir_exists(output_dir)
    output_file(join(output_dir, 'visualize_graph.html'))

    show(plot)
