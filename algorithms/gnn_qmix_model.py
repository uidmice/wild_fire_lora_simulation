
class GNNQmixModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, g):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.obs_size = _get_size(obs_space)
        self.emb_size = 16
        self.rnn_hidden_dim = model_config["lstm_cell_size"]


        self.gnn = GraphSAGE(self.obs_size, 64, self.emb_size, g)
        self.qmix = RNN(self.emb_size, self.rnn_hidden_dim, num_outputs)

        self.n_agents = model_config["n_agents"]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            self.qmix.fc1.weight.new(self.n_agents, self.rnn_hidden_dim).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        emb = self.gnn(input_dict['obs_flat'].float())
        q, h = self.qmix(emb, hidden_state[0])
        return q, [h]