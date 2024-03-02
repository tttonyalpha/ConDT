

class Embeddings(nn.Module):
    def __init__(self, state_dim, act_dim, h_dim):
        super().__init__()
        self.embed_rtg = nn.Sequential(  # nonlinear layer as mentioned in paper
            nn.Linear(1, 2 * h_dim),
            nn.ReLU(),  # nn.GELU() \
            nn.Linear(2 * h_dim, h_dim),
        )

        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        self.apply(self._init_weights)

    def forward(self, rtg, states, actions):
        latent_rtg = self.embed_rtg(rtg.unsqueeze(-1))
        latent_states = self.embed_state(states) * latent_rtg
        latent_actions = self.embed_action(actions) * latent_rtg
        return latent_states, latent_actions

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, Embeddings):
            pass
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


class DTsa(nn.Module):
    def __init__(self, latent_state_action_dim, h_dim):
        super().__init__()
        self.embed_latent_state_action = torch.nn.Linear(
            2 * h_dim, latent_state_action_dim
        )

    def forward(self, latent_state, latent_action):
        state_action = torch.cat([latent_state, latent_action], dim=-1)
        latent_state_action = self.embed_latent_state_action(state_action)

        return latent_state_action
