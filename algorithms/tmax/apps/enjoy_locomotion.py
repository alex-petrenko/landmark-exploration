import random
import sys
import time
from threading import Thread

import cv2
from pynput.keyboard import Key, Listener

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.navigator import Navigator, NavigatorNaive
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap
from algorithms.utils.algo_utils import main_observation, EPS
from algorithms.utils.env_wrappers import reset_with_info
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log

terminate = False
key_to_action = None
pause = False


# noinspection PyCallingNonCallable
def on_press(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False

    if key == Key.space:
        global pause
        pause = not pause


def on_release(_):
    pass


# Experimental "Localizer", replaced with "Navigator" in TMAX
# def edge_weight(i1, i2, d):
#     if d['loop_closure']:
#         return 3
#     if i2 < i1:
#         return 3
#     return 1
#
#
# class Localizer:
#     def __init__(self, m, agent):
#         self.m = m
#         self.agent = agent
#         self.current_landmark = None
#
#         self.max_neighborhood = 4
#         self.max_neighborhood_dist = 1.0  # should be 0.4-0.5
#         self.max_lookahead = 7
#         self.confidently_reachable = 0.1
#
#         self.neighbors = [None] * self.m.num_landmarks()
#         for i in range(self.m.num_landmarks()):
#             self.neighbors[i] = self.m.topological_neighborhood(i, max_dist=self.max_neighborhood)
#
#         self.paths = [None] * self.m.num_landmarks()
#
#         self.max_neighborhood_to_search = 1
#
#     def _ensure_path_to_goal_calculated(self, curr_landmark, final_goal):
#         if self.paths[curr_landmark] is not None:
#             return self.paths[curr_landmark]
#
#         path = self.m.get_path(curr_landmark, final_goal, edge_weight=edge_weight)
#         self.paths[curr_landmark] = path
#         return path
#
#     def _distances(self, obs, to_map, to_nodes):
#         from_obs = [obs] * len(to_nodes)
#         to_obs = [to_map.get_observation(node) for node in to_nodes]
#
#         assert len(to_obs) == len(to_nodes)
#         assert len(from_obs) == len(to_nodes)
#
#         distances = self.agent.curiosity.distance.distances_from_obs(
#             self.agent.session, obs_first=from_obs, obs_second=to_obs,
#         )
#         assert len(distances) == len(to_nodes)
#         return distances
#
#     def _find_nn_among(self, obs, nodes):
#         distances = self._distances(obs, self.m, nodes)
#         min_d, min_d_idx = min_with_idx(distances)
#         return min_d, min_d_idx
#
#     def localize_neighborhood(self, obs, goal):
#         if self.current_landmark is None:
#             # we don't know where we are, so we cannot use neighborhood heuristic
#             return None
#
#         path = self._ensure_path_to_goal_calculated(self.current_landmark, goal)
#         path = path[:self.max_lookahead + 1]
#         min_d, min_d_idx = self._find_nn_among(obs, path)
#
#         if min_d > self.max_neighborhood_dist:
#             # could not find close enough landmark in the neighborhood
#             return None
#
#         nearest_neighbor = path[min_d_idx]
#         log.debug('Neighborhood search closest landmark is %d with dist %.3f', nearest_neighbor, min_d)
#         self.max_neighborhood_to_search //= 2
#         self.max_neighborhood_to_search = max(1, self.max_neighborhood_to_search)
#         return nearest_neighbor
#
#     def localize_global(self, obs):
#         all_nodes = list(self.m.graph.nodes)
#         distances = self._distances(obs, self.m, all_nodes)
#         distances = np.asarray(distances)
#
#         if self.current_landmark is None:
#             neighborhood_size = self.m.num_landmarks()
#         else:
#             neighborhood_size = 2
#
#         while True:
#             if self.current_landmark is None:
#                 all_landmarks = list(self.m.graph.nodes)
#                 prev_neighbors = all_landmarks
#             else:
#                 prev_neighbors = self.m.topological_neighborhood(self.current_landmark, max_dist=neighborhood_size)
#
#             log.warning('Global search in neighborhood of size %d (%d)...', neighborhood_size, len(prev_neighbors))
#
#             candidates = []
#             max_distance = 1.0 if len(prev_neighbors) >= self.m.num_landmarks() else self.max_neighborhood_dist
#
#             for i in prev_neighbors:
#                 if distances[i] > max_distance:
#                     continue
#
#                 d = distances[self.neighbors[i]]
#                 dist = 0.5 * (np.median(d) + distances[i])
#                 candidates.append((dist, distances[i], i))
#
#             neighborhood_size *= 2
#             if len(candidates) <= 0:
#                 continue
#
#             candidates.sort()
#             dist, d, nearest_neighbor = candidates[0]
#
#             log.info('Best candidate %d: %.3f %.3f', nearest_neighbor, dist, d)
#
#             if len(prev_neighbors) >= self.m.num_landmarks():
#                 break
#
#             if d < self.max_neighborhood_dist:
#                 break
#
#             if neighborhood_size > self.max_neighborhood_to_search:
#                 nearest_neighbor = self.current_landmark
#                 self.max_neighborhood_to_search *= 2
#                 self.max_neighborhood_to_search = min(self.max_neighborhood_to_search, self.m.num_landmarks())
#                 break
#
#         log.warning(
#             'Global search closest landmark is %d with neigh. dist %.3f and dist %.3f',
#             nearest_neighbor, dist, d,
#         )
#         return nearest_neighbor
#
#     def get_next_target(self, obs, final_goal):
#         curr_landmark = self.localize_neighborhood(obs, final_goal)
#
#         if curr_landmark is None:
#             curr_landmark = self.localize_global(obs)
#
#         self.current_landmark = curr_landmark
#
#         path = self._ensure_path_to_goal_calculated(self.current_landmark, final_goal)
#         log.debug('Shortest path to global goal %d is %r', final_goal, path)
#
#         path = path[:self.max_lookahead + 1]
#         distances = self._distances(obs, self.m, path)
#
#         target_node = path[0]
#         target_d = distances[0]
#
#         confidently_reachable = np.random.random() * 0.1 + 0.1
#
#         if len(path) > 1:
#             target_node = path[1]
#             target_d = distances[1]
#             for i, node in enumerate(path):
#                 if i <= 1:
#                     continue
#
#                 if distances[i] > confidently_reachable:
#                     break
#                 target_node = node
#                 target_d = distances[i]
#
#         log.debug('Selected target node %d, dist %.3f', target_node, target_d)
#
#         return target_node


def display_obs(win_name, obs):
    cv2.imshow(win_name, cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR), (420, 420)))


def test_locomotion(params, env_id):
    def make_env_func():
        e = create_env(env_id, skip_frames=True)
        e.seed(0)
        return e

    # params = params.load()
    # params.ensure_serialized()

    params.num_envs = 1
    # params.naive_locomotion = True

    agent = AgentTMAX(make_env_func, params)

    agent.initialize()

    env = make_env_func()

    env_obs, info = reset_with_info(env)
    obs_prev = obs = main_observation(env_obs)
    done = False

    if params.persistent_map_checkpoint is not None:
        loaded_persistent_map = TopologicalMap.create_empty()
        loaded_persistent_map.maybe_load_checkpoint(params.persistent_map_checkpoint)
    else:
        agent.tmax_mgr.initialize([obs], [info], 1)
        loaded_persistent_map = agent.tmax_mgr.dense_persistent_maps[-1]

    m = loaded_persistent_map

    t = Timing()

    log.info('Num landmarks: %d', m.num_landmarks())
    final_goal_idx = 49

    log.info('Locomotion goal is %d', final_goal_idx)

    # localizer = Localizer(m, agent)

    final_goal_obs = m.get_observation(final_goal_idx)
    cv2.namedWindow('next_target')
    cv2.moveWindow('next_target', 800, 100)
    cv2.namedWindow('final_goal')
    cv2.moveWindow('final_goal', 1400, 100)
    display_obs('next_target', obs)
    display_obs('final_goal', final_goal_obs)
    cv2.waitKey(1)

    # localizer.current_landmark = 0
    # next_target = localizer.get_next_target(obs, final_goal_idx)
    # next_target_obs = m.get_observation(next_target)

    frame = 0

    if params.naive_locomotion:
        navigator = NavigatorNaive(agent)
    else:
        navigator = Navigator(agent)

    navigator.reset(0, m)

    next_target, next_target_d = navigator.get_next_target(
        [m], [obs], [final_goal_idx], [frame],
    )
    next_target, next_target_d = next_target[0], next_target_d[0]
    next_target_obs = m.get_observation(next_target)

    while not done and not terminate:
        with t.timeit('one_frame'):
            env.render()
            if not pause:
                if random.random() < 0.5:
                    deterministic = False
                else:
                    deterministic = True

                if params.naive_locomotion:
                    action = navigator.replay_action([0])[0]
                else:
                    action = agent.locomotion.navigate(
                        agent.session, [obs_prev], [obs], [next_target_obs], deterministic=deterministic,
                    )[0]

                env_obs, rew, done, info = env.step(action)

                log.info('Action is %d', action)
                obs_prev = obs
                obs = main_observation(env_obs)

                next_target, next_target_d = navigator.get_next_target(
                    [m], [obs], [final_goal_idx], [frame],
                )
                next_target, next_target_d = next_target[0], next_target_d[0]
                if next_target is None:
                    log.error('We are lost!')
                else:
                    log.info('Next target is %d with distance %.3f!', next_target, next_target_d)
                    display_obs('next_target', next_target_obs)
                    cv2.waitKey(1)

                if next_target is not None:
                    next_target_obs = m.get_observation(next_target)

                log.info('Frame %d...', frame)

        took_seconds = t.one_frame
        desired_fps = 10
        wait_seconds = (1.0 / desired_fps) - took_seconds
        wait_seconds = max(0.0, wait_seconds)
        if wait_seconds > EPS:
            time.sleep(wait_seconds)

        if not pause:
            frame += 1

    log.info('After loop')

    env.render()
    time.sleep(0.05)

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
