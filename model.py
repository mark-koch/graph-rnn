import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size=None, edge_feature_len=1):
        """
        Arguments:
            input_size: Length of the padded adjacency vector
            embedding_size: Size of the input embedding fed to the GRU
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            output_size: Size of the final output. Set to None if the
                output layer should be skipped.
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
        """
        super().__init__()
        self.input_size = input_size
        self.edge_feature_len = edge_feature_len
        self.linear_in = nn.Linear(input_size * edge_feature_len, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None
        self.hidden = None

    def reset_hidden(self):
        """ Resets the hidden state to 0. """
        # By setting to None, PyTorch will automatically use a zero tensor.
        # This way we do not need to know the batch size in this function.
        self.hidden = None

    def forward(self, x, x_lens=None):
        """
        Arguments:
            x: Input tensor of shape [batch, seq_len, input_size, edge_feature_len].
                Should be an  adjacency vector describing the connectivity of the
                previously generated node.
            x_lens: List of sequence lengths (i.e. number of graph nodes) of
                each batch entry. Should be on the CPU. This is used to pack
                the input to get rid of padding and increase efficiency.
                Set to 'None' to disable packing.

        Returns: The final hidde state of the GRU of shape [batch, seq_len,
            hidden_size].
        """
        # Flatten edge features
        x = torch.flatten(x, 2, 3)  # [batch, seq_len, input_size * edge_feature_len]

        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_dim]

        # Pack data to increase efficiency during training. Also see comment in the training code
        if x_lens is not None:
            x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        x, self.hidden = self.gru(x, self.hidden)  # Packed [batch, seq_len, hidden_size]

        # Unpack (reintroduces padding)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        if self.linear_out1:
            x = self.relu(self.linear_out1(x))
            x = self.linear_out2(x)

        return x


class EdgeLevelRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, edge_feature_len=1):
        """
        Arguments:
            embedding_size: Size of the input embedding fed to the GRU
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

    def set_first_layer_hidden(self, h):
        """
        Sets the hidden state of the first GRU layer. The hidden state of all
        other layers will be reset to 0. This should be set to the output of
        the graph-level RNN.

        Arguments:
            h: Hidden vector of shape [batch, hidden_size]
        """
        # Prepare zero tensor for all layers except the first
        zeros = torch.zeros([self.num_layers-1, h.shape[-2], h.shape[-1]], device=h.device)
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        self.hidden = torch.cat([h, zeros], dim=0)  # [num_layers, batch_size, hidden_size]

    def forward(self, x, x_lens=None, return_logits=False):
        """
        Arguments:
            x: Input tensor of shape [batch, seq_len, edge_feature_len].
            x_lens: List of sequence lengths (i.e. number of graph nodes) of
                each batch entry. Should be on the CPU. This is used to pack
                the input to get rid of padding and increase efficiency.
                Set to 'None' to disable packing.
            return_logits: Set to True to ouput the logits without activation

        Returns: The next edge prediction of shape [batch, seq_len, edge_feature_len].
        """
        assert self.hidden is not None, "Hidden state not set!"
        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_size_in]

        # Pack data to increase efficiency. Also see comment in the training code
        if x_lens is not None:
            x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        x, self.hidden = self.gru(x, self.hidden)  # [batch, seq_len, hidden_size]

        # Unpack (reintroduces padding)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.relu(self.linear_out1(x))  # [batch, seq_len, embedding_size_out]
        x = self.linear_out2(x)  # [batch, seq_len, edge_feature_len]
        if not return_logits:
            x = self.sigmoid(x)
        return x


class EdgeLevelMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, edge_feature_len=1):
        """
        Arguments:
            input_size: Size of the hidden state outputted by the graph-level RNN
            hidden_size: Size of the hidden layer
            output_size: Number of edges probabilities to output
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
        """
        super().__init__()
        self.edge_feature_len = edge_feature_len
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size * edge_feature_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        """
        Arguments:
            x: Input tensor of shape [batch, seq_len, input_size]. Should be the
                hidden GRU state outputted by the graph-level RNN.
        return_logits: Set to True to ouput the logits without activation

        Returns: The next edge prediction of shape [batch, seq_len, 1].
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        if not return_logits:
            x = self.sigmoid(x)

        # Reshape x to get edge features into separate dimension
        x = torch.reshape(x, [x.shape[0], x.shape[1], -1, self.edge_feature_len])  # [batch, seq_len, input_size, edge_feature_len]

        return x
