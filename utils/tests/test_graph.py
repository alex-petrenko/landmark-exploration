import random
import shutil
import time
from os.path import join
from unittest import TestCase

import networkx as nx
import numpy as np
import tensorflow as tf
from PIL import Image

from algorithms.agent import AgentLearner
from utils.graph import visualize_graph_html, visualize_graph_tensorboard
from utils.utils import vis_dir, summaries_dir, log


class TestGraph(TestCase):
    def test_visualize_graph_html(self):
        params = AgentLearner.AgentParams('__test_graph__')
        img_folder = vis_dir(params.experiment_dir())

        num_images = 4
        imglist = range(num_images)
        imglist = [join(img_folder, str(img) + '.png') for img in imglist]

        # generate some test images
        for i, img_file in enumerate(imglist):
            arr = np.zeros((5, 5, 3))
            arr[0, i % 5, i % 3] = 1
            pil_image = Image.fromarray(arr, 'RGB')
            pil_image.save(img_file)

        graph = nx.DiGraph()
        for i in range(100):
            graph.add_node(i, img=imglist[i % len(imglist)], pos=(i / 100, i / 100))

        for i in range(100):
            nonedges = list(nx.non_edges(graph))
            chosen_nonedge = random.choice(nonedges)
            graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

        visualize_graph_html(graph, layout='kamada_kawai', output_dir=img_folder)

        shutil.rmtree(params.experiment_dir())  # this line deletes the image files before they can be viewed!

    def test_graph_tensorboard(self):
        graph = nx.DiGraph()
        for i in range(100):
            graph.add_node(i)

        for i in range(100):
            nonedges = list(nx.non_edges(graph))
            chosen_nonedge = random.choice(nonedges)
            graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

        params = AgentLearner.AgentParams('__test_graph__')
        summary_dir = summaries_dir(params.experiment_dir())

        summary_writer = tf.summary.FileWriter(summary_dir)

        start_summary = time.time()
        summary = visualize_graph_tensorboard(graph, tag='test/graph')
        log.debug('Took %.3f seconds to write graph summary', time.time() - start_summary)
        summary_writer.add_summary(summary, global_step=1)

        shutil.rmtree(params.experiment_dir())
