class CuriosityModule:
    def initialize(self, session):
        raise NotImplementedError

    def generate_bonus_rewards(self, session, observations, next_obs, actions, dones, infos):
        raise NotImplementedError

    def train(self, buffer, env_steps, agent):
        raise NotImplementedError

    def set_trajectory_buffer(self, trajectory_buffer):
        raise NotImplementedError

    def is_initialized(self):
        raise NotImplementedError

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        raise NotImplementedError
