import torch
import torch.nn as nn
from collections import OrderedDict
import copy

class PPOModelWrapper(nn.Module):
    """
    PPO model wrapper that bundles the actor and critic models into a single
    unified nn.Module. Parameters are exposed under the `actor.xxx` and
    `critic.xxx` namespaces so that the two sub-models can be saved, loaded,
    and aggregated together while staying individually addressable.
    """

    def __init__(self, actor_model, critic_model=None):
        super(PPOModelWrapper, self).__init__()

        # Register the actor and critic sub-models.
        self.actor = actor_model
        if critic_model is not None:
            self.critic = critic_model
        else:
            # If no critic is supplied, fall back to a deep copy of the actor.
            self.critic = copy.deepcopy(actor_model)

    def forward(self, *args, **kwargs):
        # By default the forward pass is delegated to the actor.
        return self.actor(*args, **kwargs)

    def get_actor_output(self, *args, **kwargs):
        """Run the actor's forward pass and return its output."""
        return self.actor(*args, **kwargs)

    def get_critic_output(self, *args, **kwargs):
        """Run the critic's forward pass and return its output."""
        return self.critic(*args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict so that every parameter is exported with an
        explicit `actor.` or `critic.` prefix, keeping the two sub-models'
        parameters disambiguated in the combined state dictionary.
        """
        state_dict = OrderedDict()

        # Add the actor parameters (with the `actor.` prefix).
        actor_state_dict = self.actor.state_dict(prefix='', keep_vars=keep_vars)
        for key, value in actor_state_dict.items():
            state_dict[f'actor.{key}'] = value

        # Add the critic parameters (with the `critic.` prefix).
        if hasattr(self, 'critic') and self.critic is not None:
            critic_state_dict = self.critic.state_dict(prefix='', keep_vars=keep_vars)
            for key, value in critic_state_dict.items():
                state_dict[f'critic.{key}'] = value

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle the prefixed parameter layout,
        routing `actor.*` keys to the actor and `critic.*` keys to the critic.
        """
        actor_state_dict = OrderedDict()
        critic_state_dict = OrderedDict()

        # Split the combined dict into separate actor and critic parameter sets.
        for key, value in state_dict.items():
            if key.startswith('actor.'):
                clean_key = key[6:]  # strip the 'actor.' prefix
                actor_state_dict[clean_key] = value
            elif key.startswith('critic.'):
                clean_key = key[7:]  # strip the 'critic.' prefix
                critic_state_dict[clean_key] = value

        # Load the actor parameters.
        missing_keys_actor = []
        unexpected_keys_actor = []
        if actor_state_dict:
            missing_keys_actor, unexpected_keys_actor = self.actor.load_state_dict(
                actor_state_dict, strict=strict
            )

        # Load the critic parameters.
        missing_keys_critic = []
        unexpected_keys_critic = []
        if critic_state_dict and hasattr(self, 'critic'):
            missing_keys_critic, unexpected_keys_critic = self.critic.load_state_dict(
                critic_state_dict, strict=strict
            )

        # Merge the missing and unexpected key lists, restoring the prefixes so
        # the caller sees keys in the same namespaced form they were given.
        missing_keys = [f'actor.{k}' for k in missing_keys_actor] + \
                      [f'critic.{k}' for k in missing_keys_critic]
        unexpected_keys = [f'actor.{k}' for k in unexpected_keys_actor] + \
                         [f'critic.{k}' for k in unexpected_keys_critic]

        # Return a result object compatible with nn.Module.load_state_dict.
        from torch.nn.modules.module import _IncompatibleKeys
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def to(self, device):
        """Override `to` so both sub-models are moved to the target device."""
        self.actor = self.actor.to(device)
        if hasattr(self, 'critic'):
            self.critic = self.critic.to(device)
        return super().to(device)

    def train(self, mode=True):
        """Override `train` so both sub-models switch training mode together."""
        self.actor.train(mode)
        if hasattr(self, 'critic'):
            self.critic.train(mode)
        return super().train(mode)

    def eval(self):
        """Override `eval` so both sub-models switch to evaluation mode together."""
        self.actor.eval()
        if hasattr(self, 'critic'):
            self.critic.eval()
        return super().eval()
