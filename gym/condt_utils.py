from tqdm import tqdm
import matplotlib.pyplot as plt


class BinarizedDataset(IterableDataset):
    def __init__(
        self, dataset: SequenceDataset, n_bins: int = 10, smoothing_factor: float = 0.05
    ):
        self.n_bins = n_bins
        self.rsa = []  # array of tuples: return to-go, state, action

        state_std = dataset.state_std
        state_mean = dataset.state_mean
        reward_scale = dataset.reward_scale

        for traj in tqdm(dataset.dataset):
            states = traj["observations"]
            actions = traj["actions"]
            returns_to_go = traj["returns"]

            states = (states - state_mean) / state_std
            returns_to_go = returns_to_go * reward_scale  # returns_to_go

            traj_rsa = np.array(list(zip(returns_to_go, states, actions)))
            self.rsa.append(traj_rsa)

        # check atari colab for more efficient realization
        self.rsa = np.concatenate(self.rsa, axis=0)
        sorted_indices = np.argsort(self.rsa[:, 0])
        # np array sortet by reward to-go  it's inefficient but can afford now
        self.rsa = self.rsa[sorted_indices]

        self.bins, self.bin_edges = self.bin_indices()

        self.bins_probs = np.array(
            [el[1] - el[0] + 1 if el[0] is not None else 0 for el in self.bins]
        )  # number of elements per bin
        self.bins_probs = self.bins_probs / self.bins_probs.sum()
        # Laplacian smoothing: make distribution more unoform
        self.bins_probs = (self.bins_probs + smoothing_factor) / (
            np.sum(self.bins_probs) + len(self.bins_probs) * smoothing_factor
        )

    def bin_indices(self):
        rtg_min, rtg_max = self.rsa[0][0], self.rsa[-1][0]  # for sorted array

        bins = np.linspace(
            rtg_min, rtg_max, self.n_bins + 1
        )  # split on n_bins equal chunks by rtg as mentioned in paper
        digitized = np.digitize(self.rsa[:, 0], bins + 1)

        indices = [
            (np.min(np.where(digitized == i)), np.max(np.where(digitized == i)))
            if i in digitized
            else (None, None)
            for i in range(1, self.n_bins + 1)
        ]
        return indices, bins

    def __iter__(
        self,
    ):  # sampling bin number then sampling anchor and posirive from bin
        while True:
            bin_idx = np.random.choice(
                np.arange(self.n_bins), p=self.bins_probs)
            l, r = self.bins[bin_idx]

            anchor_idx, pos_idx = random.randint(l, r), random.randint(l, r)
            while pos_idx == anchor_idx:  # if pos_idx and anchor_idx same
                pos_idx = random.randint(l, r)

            anchor = self.rsa[anchor_idx]
            pos = self.rsa[pos_idx]

            rtg = torch.tensor([anchor[0], pos[0]], dtype=torch.float32)
            states = torch.from_numpy(np.stack([anchor[1], pos[1]]))
            actions = torch.from_numpy(np.stack([anchor[2], pos[2]]))

            yield rtg, states, actions, bin_idx


def SimRCLRL(latent_state_action, t):  # (B, 2, latent_sa_dim)
    anc = latent_state_action[:, 0]
    pos = latent_state_action[:, 1]

    anc_pos_sim = F.cosine_similarity(
        anc, pos, dim=-1)  # anchor - positive sim
    mask = torch.eye(
        anc.shape[0], dtype=torch.bool, device=anc_pos_sim.device
    )  # mask diagonal

    pairvise_sim_anc_pos = torch.exp((
        F.cosine_similarity(anc[None, :, :], pos[:, None, :], dim=-1) / t)
    )  # anchor - negative sim
    pairvise_sim_anc_anc = torch.exp((
        F.cosine_similarity(anc[None, :, :], anc[:, None, :], dim=-1) / t)
    )  # pos - negative sim

    pairvise_sim_anc_pos.masked_fill_(mask, 0)  # or 0
    pairvise_sim_anc_anc.masked_fill_(mask, 0)

    loss = (
        -anc_pos_sim / t
        + (pairvise_sim_anc_pos + pairvise_sim_anc_anc).sum(axis=1).log()
    ).mean()  # in paper - .sum() ??
    return loss


def mean_cossim(latent_state_action):  # (B, 2, latent_sa_dim)
    anc = latent_state_action[:, 0]
    pos = latent_state_action[:, 1]

    mean_pos_sim = F.cosine_similarity(anc, pos, dim=-1).mean()
    mask = torch.eye(anc.shape[0], dtype=torch.bool,
                     device=mean_pos_sim.device)

    pairvise_sim_anc_pos = F.cosine_similarity(
        anc[None, :, :], pos[:, None, :], dim=-1)
    pairvise_sim_anc_anc = F.cosine_similarity(
        anc[None, :, :], anc[:, None, :], dim=-1)

    pairvise_sim_anc_pos.masked_fill_(mask, 0)
    pairvise_sim_anc_anc.masked_fill_(mask, 0)

    mean_neg_sim = (pairvise_sim_anc_pos + pairvise_sim_anc_anc).mean() / 2
    return mean_pos_sim, mean_neg_sim
