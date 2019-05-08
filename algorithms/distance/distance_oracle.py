import math

from algorithms.distance.distance import DistanceNetwork
from utils.utils import log


class DistanceOracle(DistanceNetwork):
    def __init__(self, env, params):
        super().__init__(env, params)

    @staticmethod
    def _default_pos():
        return {'agent_x': 0.0, 'agent_y': 0.0}

    def distances(self, session, obs_first_encoded, obs_second_encoded, infos_first=None, infos_second=None):
        if len(obs_first_encoded) <= 0:
            return []

        if infos_first is None or infos_second is None:
            # fall back to standard distance net
            return super().distances(session, obs_first_encoded, obs_second_encoded)

        assert len(infos_first) == len(infos_second)

        far_distance = 250.0

        d = []
        for i in range(len(infos_first)):
            try:
                pos1, pos2 = infos_first[i]['pos'], infos_second[i]['pos']
            except (KeyError, TypeError):
                log.warning('No coordinate information provided!')
                pos1 = pos2 = self._default_pos()

            x1, y1 = pos1['agent_x'], pos1['agent_y']
            x2, y2 = pos2['agent_x'], pos2['agent_y']

            ground_truth_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            ground_truth_distance = max(0.0, ground_truth_distance)  # just in case, to avoid numerical issues

            # linear interpolation
            # 0 == 0.0
            # >=300 == 1.0
            distance_normalized = min(ground_truth_distance / far_distance, 1.0)
            d.append(distance_normalized)

        return d

    def distances_from_obs(
            self, session, obs_first, obs_second, hashes_first=None, hashes_second=None,
            infos_first=None, infos_second=None
    ):
        if infos_first is None or infos_second is None:
            # fall back to standard distance net
            return super().distances_from_obs(session, obs_first, obs_second, hashes_first, hashes_second)
        else:
            return self.distances(
                session, obs_first, obs_second,
                infos_first=infos_first, infos_second=infos_second,
            )

    def train(self, buffer, env_steps, agent, timing=None):
        """We do not train the oracle."""
        return 0
