from os.path import join

from bokeh import palettes
from bokeh.io import output_file, show
from bokeh.models import BoxSelectTool, Circle, HoverTool, MultiLine, Plot, Range1d, TapTool
# noinspection PyProtectedMember
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, NodesOnly
import networkx as nx

from utils.utils import project_root, ensure_dir_exists


def visualize_graph(graph, title_text='Graph Visualizer'):
    """
    This method visualizes a NetworkX graph using Bokeh.

    :param graph: NetworkX graph with node attributes containing image filenames.
    :param title_text: String to be displayed above the visualization.
    """
    pos = nx.get_node_attributes(graph, 'pos')

    hover_tool = HoverTool(tooltips='<img src="@imgs" height="200" alt="@imgs" width="200"></img>', show_arrow=False)

    plot = Plot(plot_width=400, plot_height=400, x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1))
    plot.title.text = title_text
    plot.title.align = 'center'

    plot.add_tools(hover_tool, TapTool(), BoxSelectTool())

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
