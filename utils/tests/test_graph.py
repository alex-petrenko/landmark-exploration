from os.path import join
from unittest import TestCase

import networkx as nx

from utils.graph import visualize_graph
from utils.utils import project_root


class TestGraph(TestCase):
    def test_visualize_graph(self):
        imglist = ['elephant.jpg', 'lion.jpg', 'ostrich.jpg', 'snake.jpg']
        img_folder = project_root()
        imglist = [join(img_folder, img) for img in imglist] * 25

        graph = nx.random_geometric_graph(100, 0.15)
        for n in graph.nodes():
            graph.node[n]['img'] = imglist[n]

        visualize_graph(graph, 'Test Visualize Graph')
