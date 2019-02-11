from os.path import join
import random
import unittest
from unittest import TestCase

import networkx as nx

from utils.graph import visualize_graph_html, visualize_graph_tensorboard
from utils.utils import project_root, vis_dir


@unittest.skip('Uses external data')  # TODO
class TestGraph(TestCase):
    def test_visualize_graph_html(self):
        imglist = ['elephant.jpg', 'lion.jpg', 'ostrich.jpg', 'snake.jpg']
        img_folder = vis_dir(project_root())
        imglist = [join(img_folder, img) for img in imglist]

        # graph = nx.random_geometric_graph(100, 0.15)
        # for n in graph.nodes():
        #     graph.node[n]['img'] = imglist[int(n) % len(imglist)]

        graph = nx.DiGraph()
        for i in range(100):
            graph.add_node(i, img=imglist[i % len(imglist)], pos=(i / 100, i / 100))

        for i in range(100):
            nonedges = list(nx.non_edges(graph))
            chosen_nonedge = random.choice(nonedges)
            graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

        # visualize_graph(graph, layout='fruchterman_reingold')
        visualize_graph_html(graph, layout='kamada_kawai')

    def test_graph_tensorboard(self):
        graph = nx.DiGraph()
        for i in range(100):
            graph.add_node(i)

        for i in range(100):
            nonedges = list(nx.non_edges(graph))
            chosen_nonedge = random.choice(nonedges)
            graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

        visualize_graph_tensorboard(graph)
