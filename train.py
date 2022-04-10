import argparse
import yaml
import torch
import os
import time
import datetime

from data import GraphDataSet
from extension_data import DirectedGraphDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP


def train_mlp_step(graph_rnn, edge_mlp, data, criterion, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with MLP edge model. """
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    s, lens = data['x'].float().to(device), data['len'].cpu()

    # If s does not have edge features, just add a dummy dimension 1
    # to the end
    if len(s.shape) == 3:
        s = s.unsqueeze(3)

    # Teacher forcing: We want the input to be one node offset from the target.
    # Therefore, we introduce an SOS token (the authors choose the value 1) to
    # the input tensor:
    #       x = [1,  S0, S1, S2, ...,     S(n-1), Sn]
    #       y = [S0, S1, S2, ..., S(n-1), Sn    ,  0]
    one_frame = torch.ones([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    zero_frame = torch.zeros([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    x = torch.cat((one_frame, s[:, :, :]), dim=1)
    y = torch.cat((s[:, :, :], zero_frame), dim=1)

    lens = lens+1

    graph_rnn.reset_hidden()
    hidden = graph_rnn(x, lens)
    y_pred = edge_mlp(hidden, return_logits=use_edge_features)  # Edge features use cross entropy loss which reqires logits

    y = pack_padded_sequence(y, lens, batch_first=True, enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    # Edge features use Cross Entroy loss which wants the features in
    # the second dimension, so we have to swap them around
    if use_edge_features:
        y = torch.swapaxes(y, 1, 3)
        y_pred = torch.swapaxes(y_pred, 1, 3)

    loss = criterion(y_pred, y)
    loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    return loss.item()


def train_rnn_step(graph_rnn, edge_rnn, data, criterion, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with RNN edge model. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    seq, lens = data['x'].float().to(device), data['len'].cpu()

    # If s does not have edge features, just add a dummy dimension 1
    # to the end
    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)

    # Add SOS token to the node-level RNN input to prevent it from looking
    # into the future.
    one_frame = torch.ones([seq.shape[0], 1, seq.shape[2], seq.shape[3]], device=device)
    x_node_rnn = torch.cat((one_frame, seq[:, :-1, :]), dim=1)

    # Compute hidden graph-level representation
    graph_rnn.reset_hidden()
    hidden = graph_rnn(x_node_rnn, lens)

    # While this hidden graph representation can be computed via a single
    # invocation of the NodeRNN, the training for the EdgeRRN is a bit more
    # involved. Given a graph sequence
    #
    #   [(S11), (S21, S22), (S31, S32, S33), ..., (Sn1, ..., Snn)]
    #
    # we need to run the EdgeRNN n times to generate each of the adjacency
    # vectors (S11'), (S21', S22'), (S31', S31', S33'), ... and compute the
    # loss for each of them. However, we can speed up the training by splitting
    # the vectors in batches and performing the computations simultaneously:
    #
    #       1. time step: Feed batch [S11, S21, S31, ..., Sn1]
    #       2. time step: Feed batch [ - , S22, S32, ..., Sn2]
    #       ...
    #       n. time step: Feed batch [ - ,  - ,  - , ..., Snn]
    #
    # Essentially, we want to reshape [batch, nodes, edges] into [batch*nodes,
    # edges, 1] which is compatible with the EdgeRNN.

    # Incidentally, this can be implemented using the PyTorch packing function.
    # Suppose we have a batch of zero padded graph sequences, e.g.
    #
    #   seq = [[(A11, -, -, -), (A21, A22, -, -), (A31, A32, A33, -), (A41, A42, A43, A44)]    <- graph A
    #          [(B11, -, -, -), (B21, B22, -, -), ( - ,  - ,  - , -), ( - ,  - ,  - ,  - )]    <- graph B
    #          [(C11, -, -, -), (C21, C22, -, -), (C31, C32, C33, -), ( - ,  - ,  - ,  - )]]   <- graph C
    #
    # encoding graph A with 5 nodes, graph B with 3 nodes and graph C with 4
    # nodes. First, packing this (along the node axis) yields
    #
    #   seq_packed = [(A11,  - ,  - ,  - ), (B11,  - ,  - , -), (C11,  - , -, -),   <- batch 1
    #                 (A21, A22,  - ,  - ), (B21, B22,  - , -), (C21, C22, -, -),   <- batch 2
    #                 (A31, A32, A33,  - ), (C31, C32, C33, -),                     <- batch 3
    #                 (A41, A42, A43, A44)]                                         <- batch 4
    #
    # Note that we don't need to sort by sequence length since PyTorch can do
    # this itself now:
    seq_packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False).data

    # Packing once more finally yields the desired effect:
    #
    #   [A11, B11, C11, A21, B21, C21, A31, C31, A41,   <- batch 1
    #    A22, B22, C22, A31, C32, A42,                  <- batch 2
    #    A33, C33, A43,                                 <- batch 3
    #    A44]                                           <- batch 4
    #
    # However, we won't do this packing here since the EdgeRNN also performs
    # packing, so we can delegate it there.

    # We now need to compute the sequence lengths of `seq_packed`.
    # TODO: Do this more efficiently
    seq_packed_len = []
    m = graph_rnn.input_size
    for l in lens:
        for i in range(1, l + 1):
            seq_packed_len.append(min(i, m))
    seq_packed_len.sort()

    # Add nex axis to tensor to be compatible with EdgeRNN input shape.
    # This is no longer needed since seq has edge_features dimension
    # by default.
    # seq_packed = seq_packed.unsqueeze(2)  # [batch, seq_len, 1]

    # Add SOS token to the edge-level RNN input to prevent it from looking
    # into the future.
    one_frame = torch.ones([seq_packed.shape[0], 1, seq_packed.shape[2]], device=device)
    x_edge_rnn = torch.cat((one_frame, seq_packed[:, :-1, :]), dim=1)
    y_edge_rnn = seq_packed

    # We need to set the hidden state of the first EdgeRNN layer to the
    # previously computed hidden representation. Since we feed node-packed
    # data to the EdgeRNN, we also need to pack the hidden representation:
    hidden_packed = pack_padded_sequence(hidden, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed)

    # Compute edge probabilities
    y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len, return_logits=use_edge_features)  # Edge features use cross entropy loss which reqires logits

    y_edge_rnn = pack_padded_sequence(y_edge_rnn, seq_packed_len, batch_first=True, enforce_sorted=False)
    y_edge_rnn, _ = pad_packed_sequence(y_edge_rnn, batch_first=True)

    # Edge features use Cross Entroy loss which wants the features in
    # the second dimension, so we have to swap them around
    if use_edge_features:
        y_edge_rnn = torch.swapaxes(y_edge_rnn, 1, 2)
        y_edge_rnn = torch.argmax(y_edge_rnn, dim=1)  # One hot to class labels
        y_edge_rnn_pred = torch.swapaxes(y_edge_rnn_pred, 1, 2)

    loss = criterion(y_edge_rnn_pred, y_edge_rnn)
    loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', required=False, default=0, type=int,
                        help='Id of the GPU to use')
    args = parser.parse_args()

    base_path = os.path.dirname(args.config_file)

    # Load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.join(base_path, config['train']['checkpoint_dir']), exist_ok=True)
    os.makedirs(os.path.join(base_path, config['train']['log_dir']), exist_ok=True)


    # Create models
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=config['model']['EdgeRNN']['hidden_size'],
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(device)
        step_fn = train_rnn_step
    else:
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=None,  # No output layer needed
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelMLP(input_size=config['model']['GraphRNN']['hidden_size'],
                                  output_size=config['data']['m'],
                                  **config['model']['EdgeMLP']).to(device)
        step_fn = train_mlp_step

    # If we use directed graphs we need edge features, requiring
    # Cross Entropy Loss
    use_edge_features = 'edge_feature_len' in config['model']['GraphRNN'] \
                        and config['model']['GraphRNN']['edge_feature_len'] > 1
    if use_edge_features:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss().to(device)

    optim_node_model = torch.optim.Adam(list(node_model.parameters()), lr=config['train']['lr'])
    optim_edge_model = torch.optim.Adam(list(edge_model.parameters()), lr=config['train']['lr'])

    scheduler_node_model = MultiStepLR(optim_node_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])
    scheduler_edge_model = MultiStepLR(optim_edge_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])

    # Tensorboard
    writer = SummaryWriter(os.path.join(base_path, config['train']['log_dir']))

    global_step = 0

    # Restore from checkpoint
    if args.restore_path:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        global_step = state["global_step"]
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        criterion.load_state_dict(state["criterion"])

    if 'mode' in config['model'] and 'directed' in config['model']['mode']:
        dataset = DirectedGraphDataSet(**config['data'])
    else:
        dataset = GraphDataSet(**config['data'])
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'])

    node_model.train()
    edge_model.train()

    done = False
    loss_sum = 0
    start_step = global_step
    start_time = time.time()
    while not done:
        for data in data_loader:
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            loss = step_fn(node_model, edge_model, data, criterion, optim_node_model,
                           optim_edge_model, scheduler_node_model, scheduler_edge_model,
                           device, use_edge_features)
            loss_sum += loss

            # Tensorboard
            writer.add_scalar('loss', loss, global_step)

            if global_step % config['train']['print_iter'] == 0:
                running_time = time.time() - start_time
                time_per_iter = running_time / (global_step - start_step)
                eta = (config['train']['steps'] - global_step) * time_per_iter
                print("[{}] loss={} time_per_iter={:.4f}s eta={}"
                      .format(global_step,
                              loss_sum / config['train']['print_iter'],
                              time_per_iter,
                              datetime.timedelta(seconds=eta)))
                loss_sum = 0

            if global_step % config['train']['checkpoint_iter'] == 0 or global_step+1 > config['train']['steps']:
                state = {
                    "global_step": global_step,
                    "config": config,
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    "criterion": criterion.state_dict()
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(base_path, config['train']['checkpoint_dir'],
                                               "checkpoint-{}.pth".format(global_step)))

    writer.close()
