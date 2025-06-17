from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F

class LinkPredModel(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.3,
            negative_slope: float = 0.2,
            dot_product: bool = True,
    ):
        super(LinkPredModel, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        if not dot_product:
            self.classifier = torch.nn.Linear(2 * hidden_dim, 1)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dot_product = dot_product

    def forward(self, x, edge_index, edge_label_index):

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        if self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index)

        x_src = x[edge_label_index[0]]
        x_trg = x[edge_label_index[1]]

        if self.dot_product:
            out = torch.sum(x_src * x_trg, dim=1)
        else:
            x = torch.cat([x_src, x_trg], dim=1)
            out = self.classifier(x).squeeze()

        return out

    def loss(self, preds, link_label):
        return self.loss_fn(preds, link_label.type(preds.dtype))

def merge_edge_labels(data):
    edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    edge_label = torch.cat([data.pos_edge_label, data.neg_edge_label], dim=0)
    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    return data

def train(model, data_train, data_val, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_label_index)
    loss = model.loss(out, data_train.edge_label)
    loss.backward()
    optimizer.step()

    loss_train, acc_train = test(model, data_train)
    loss_val, acc_val = test(model, data_val)
    return loss_train, acc_train, loss_val, acc_val

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    loss = model.loss(out, data.edge_label.type(out.dtype))
    probs = torch.sigmoid(out)
    preds = (probs > 0.5).float()
    correct = (preds == data.edge_label).sum().item()
    acc = correct / data.edge_label.size(0)
    return loss.item(), acc

@torch.no_grad()
def predict_new_edges(model, data, threshold=0.9):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    probs = torch.sigmoid(out)
    mask = probs > threshold
    predicted_edges = data.edge_label_index[:, mask].T.cpu().tolist()
    predicted_scores = probs[mask].cpu().tolist()

    new_links = []
    for (src, tgt), score in zip(predicted_edges, predicted_scores):
        new_links.append({"source": src, "target": tgt, "score": score})

    return new_links
