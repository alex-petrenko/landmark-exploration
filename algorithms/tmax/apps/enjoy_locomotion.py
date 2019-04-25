import random
import sys
import time
from threading import Thread

import cv2
import numpy as np
from pynput.keyboard import Key, Listener

from algorithms.utils.algo_utils import main_observation, EPS
from algorithms.utils.env_wrappers import reset_with_info
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.envs.generate_env_map import generate_env_map
from utils.timing import Timing
from utils.utils import log, min_with_idx

terminate = False
current_actions = []
key_to_action = None


# noinspection PyCallingNonCallable
def on_press(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False

    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action not in current_actions:
            current_actions.append(action)


# noinspection PyCallingNonCallable
def on_release(key):
    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action in current_actions:
            current_actions.remove(action)


def edge_weight(i1, i2, d):
    if d['loop_closure']:
        return 3
    if i2 < i1:
        return 3
    return 1


class Localizer:
    def __init__(self, m, agent):
        self.m = m
        self.agent = agent
        self.current_landmark = None

        self.max_neighborhood = 4
        self.max_neighborhood_dist = 0.4
        self.max_lookahead = 7
        self.confidently_reachable = 0.1

        self.neighbors = [None] * self.m.num_landmarks()
        for i in range(self.m.num_landmarks()):
            self.neighbors[i] = self.m.topological_neighborhood(i, max_dist=self.max_neighborhood)

        self.paths = [None] * self.m.num_landmarks()

        self.max_neighborhood_to_search = 1

    def _ensure_path_to_goal_calculated(self, curr_landmark, final_goal):
        if self.paths[curr_landmark] is not None:
            return self.paths[curr_landmark]

        path = self.m.get_path(curr_landmark, final_goal, edge_weight=edge_weight)
        self.paths[curr_landmark] = path
        return path

    def _distances(self, obs, to_map, to_nodes):
        from_obs = [obs] * len(to_nodes)
        to_obs = [to_map.get_observation(node) for node in to_nodes]

        assert len(to_obs) == len(to_nodes)
        assert len(from_obs) == len(to_nodes)

        distances = self.agent.curiosity.distance.distances_from_obs(
            self.agent.session, obs_first=from_obs, obs_second=to_obs,
        )
        assert len(distances) == len(to_nodes)
        return distances

    def _find_nn_among(self, obs, nodes):
        distances = self._distances(obs, self.m, nodes)
        min_d, min_d_idx = min_with_idx(distances)
        return min_d, min_d_idx

    def localize_neighborhood(self, obs, goal):
        if self.current_landmark is None:
            # we don't know where we are, so we cannot use neighborhood heuristic
            return None

        path = self._ensure_path_to_goal_calculated(self.current_landmark, goal)
        path = path[:self.max_lookahead + 1]
        min_d, min_d_idx = self._find_nn_among(obs, path)

        if min_d > self.max_neighborhood_dist:
            # could not find close enough landmark in the neighborhood
            return None

        nearest_neighbor = path[min_d_idx]
        log.debug('Neighborhood search closest landmark is %d with dist %.3f', nearest_neighbor, min_d)
        self.max_neighborhood_to_search //= 2
        self.max_neighborhood_to_search = max(1, self.max_neighborhood_to_search)
        return nearest_neighbor

    def localize_global(self, obs):
        all_nodes = list(self.m.graph.nodes)
        distances = self._distances(obs, self.m, all_nodes)
        distances = np.asarray(distances)

        if self.current_landmark is None:
            neighborhood_size = self.m.num_landmarks()
        else:
            neighborhood_size = 2

        while True:
            if self.current_landmark is None:
                all_landmarks = list(self.m.graph.nodes)
                prev_neighbors = all_landmarks
            else:
                prev_neighbors = self.m.topological_neighborhood(self.current_landmark, max_dist=neighborhood_size)

            log.warning('Global search in neighborhood of size %d (%d)...', neighborhood_size, len(prev_neighbors))

            candidates = []
            max_distance = 1.0 if len(prev_neighbors) >= self.m.num_landmarks() else self.max_neighborhood_dist

            for i in prev_neighbors:
                if distances[i] > max_distance:
                    continue

                d = distances[self.neighbors[i]]
                dist = 0.5 * (np.median(d) + distances[i])
                candidates.append((dist, distances[i], i))

            neighborhood_size *= 2
            if len(candidates) <= 0:
                continue

            candidates.sort()
            dist, d, nearest_neighbor = candidates[0]

            log.info('Best candidate %d: %.3f %.3f', nearest_neighbor, dist, d)

            if len(prev_neighbors) >= self.m.num_landmarks():
                break

            if d < self.max_neighborhood_dist:
                break

            if neighborhood_size > self.max_neighborhood_to_search:
                nearest_neighbor = self.current_landmark
                self.max_neighborhood_to_search *= 2
                self.max_neighborhood_to_search = min(self.max_neighborhood_to_search, self.m.num_landmarks())
                break

        log.warning(
            'Global search closest landmark is %d with neigh. dist %.3f and dist %.3f',
            nearest_neighbor, dist, d,
        )
        return nearest_neighbor

    def get_next_target(self, obs, final_goal):
        curr_landmark = self.localize_neighborhood(obs, final_goal)

        if curr_landmark is None:
            curr_landmark = self.localize_global(obs)

        self.current_landmark = curr_landmark

        path = self._ensure_path_to_goal_calculated(self.current_landmark, final_goal)
        log.debug('Shortest path to global goal %d is %r', final_goal, path)

        path = path[:self.max_lookahead + 1]
        distances = self._distances(obs, self.m, path)

        target_node = path[0]
        target_d = distances[0]

        confidently_reachable = np.random.random() * 0.1 + 0.05

        if len(path) > 1:
            target_node = path[1]
            target_d = distances[1]
            for i, node in enumerate(path):
                if i <= 1:
                    continue

                if distances[i] > confidently_reachable:
                    break
                target_node = node
                target_d = distances[i]

        log.debug('Selected target node %d, dist %.3f', target_node, target_d)

        return target_node


def display_obs(win_name, obs):
    cv2.imshow(win_name, cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR), (420, 420)))


def test_locomotion(params, env_id):
    def make_env_func():
        e = create_env(env_id, skip_frames=False)
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params)
    agent.initialize()

    env = make_env_func()
    map_img, coord_limits = generate_env_map(make_env_func)

    env_obs, info = reset_with_info(env)
    obs_prev = obs = main_observation(env_obs)
    done = False

    loaded_persistent_map = TopologicalMap.create_empty()
    loaded_persistent_map.maybe_load_checkpoint(params.persistent_map_checkpoint)

    t = Timing()

    frame = 0
    frame_repeat = 4
    action = 0

    final_goal_idx = 92
    m = loaded_persistent_map
    localizer = Localizer(m, agent)

    final_goal_obs = m.get_observation(final_goal_idx)
    display_obs('final_goal', final_goal_obs)
    display_obs('next_target', obs)
    cv2.waitKey()

    next_target = localizer.get_next_target(obs, final_goal_idx)
    next_target_obs = m.get_observation(next_target)

    while not done and not terminate:
        with t.timeit('one_frame'):
            env.render()

            if frame % frame_repeat == 0:
                if random.random() < 0.05:
                    action = env.action_space.sample()
                else:
                    if random.random() < 0.1:
                        deterministic = False
                    else:
                        deterministic = True

                    action = agent.locomotion.navigate(
                        agent.session, [obs_prev], [obs], [next_target_obs], deterministic=deterministic,
                    )

            env_obs, rew, done, info = env.step(action)

            if frame % frame_repeat == 0:
                obs_prev = obs
                obs = main_observation(env_obs)
                next_target = localizer.get_next_target(obs, final_goal_idx)
                next_target_obs = m.get_observation(next_target)

                display_obs('next_target', next_target_obs)
                cv2.waitKey(1)

        took_seconds = t.one_frame
        desired_fps = 40
        wait_seconds = (1.0 / desired_fps) - took_seconds
        wait_seconds = max(0.0, wait_seconds)
        if wait_seconds > EPS:
            time.sleep(wait_seconds)

        frame += 1

    env.render()
    time.sleep(0.2)

    env.close()
    agent.finalize()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    env_id = args.env

    global key_to_action
    if 'dmlab' in env_id:
        from utils.envs.dmlab import play_dmlab
        key_to_action = play_dmlab.key_to_action
    elif 'atari' in env_id:
        key_to_action = atari_utils.key_to_action
    elif 'doom' in env_id:
        key_to_action = doom_utils.key_to_action
    else:
        raise Exception('Unknown env')

    # start keypress listener (to pause/resume execution or exit)
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    status = test_locomotion(params, args.env)

    if not terminate:
        log.debug('Press ESC to exit...')
    listener_thread.join()

    return status


if __name__ == '__main__':
    sys.exit(main())
