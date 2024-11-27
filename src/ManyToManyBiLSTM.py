import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ManyToManyBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(ManyToManyBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiplied by 2 for bidirectional

    def forward(self, inputs, lengths):
        # Initialize hidden and cell states
        h_0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_dim, device=device)

        # Pack and pass through LSTM
        packed_inputs = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_inputs, (h_0, c_0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Fully connected layer
        return self.fc(output)
        

    def predict_states(self, inputs, lengths, num_predictions=50, temperature=1.0, use_gumbel=True):
        logits = self.forward(inputs, lengths)
        logits = logits.unsqueeze(1).expand(-1, num_predictions, -1, -1)

        if use_gumbel:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=logits.device)))
            sampled_logits = logits + gumbel_noise
        else:
            sampled_logits = logits

        prob_predictions = torch.sigmoid(sampled_logits / temperature)
        return prob_predictions.squeeze(-1)


def train_one_epoch(model, data_loader, criterion, optimizer, clip_value=1.0):
    model.train()
    total_loss = 0
    for inputs, targets, lengths in data_loader:
        target_score = inputs[:, :, 0]  # Shape: [batch_size, seq_length]
        predictions = model.predict_states(inputs, lengths, num_predictions=50)
        loss = criterion(predictions, targets, target_score, lengths)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value) # Clip gradients to prevent exploding gradients
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, criterion, num_predictions=20, temperature=1.0, use_gumbel=True):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, lengths in data_loader:
            target_score = inputs[:, :, 0] # Shape: [batch_size, seq_length]
            predictions = model.predict_states(inputs, lengths, num_predictions=num_predictions,
                                               temperature=temperature, use_gumbel=use_gumbel)
            loss = criterion(predictions, targets, target_score, lengths)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def custom_loss_function(predictions, targets, target_score, lengths, alpha=1.0, beta=1.0):
    device = predictions.device
    batch_size, num_predictions, seq_length = predictions.shape

    # Mask to handle variable sequence lengths
    mask = torch.arange(seq_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [batch_size, seq_length]

    # Expand targets for multiple predictions
    targets_expanded = targets.unsqueeze(1).expand(-1, num_predictions, -1)  # [batch_size, num_predictions, seq_length]

    # Compute Binary Cross-Entropy Loss
    bce_loss = F.binary_cross_entropy(predictions, targets_expanded, reduction='none')  # [batch_size, num_predictions, seq_length]
    bce_loss = bce_loss.mean(dim=1)  # Average over num_predictions: [batch_size, seq_length]
    bce_loss = (bce_loss * mask).sum() / mask.sum()  # Average over valid elements

    # Compute Score Alignment Loss (Mean Squared Error)
    avg_predictions = predictions.mean(dim=1)  # Average over num_predictions: [batch_size, seq_length]
    score_loss = F.mse_loss(avg_predictions, target_score, reduction='none')
    score_loss = (score_loss * mask).sum() / mask.sum()

    # Total loss
    total_loss = bce_loss * alpha + score_loss * beta
    return total_loss


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, clip_value=1.0):
    train_loss_list, test_loss_list = [], []
    for epoch in range(num_epochs):
        train_loss= train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss = evaluate_model(model, test_loader, criterion, num_predictions=50, temperature=1, use_gumbel=True)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    return model, train_loss_list, test_loss_list


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def collate_batch(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(i) for i in inputs], dtype=torch.int)

    sorted_lengths, sorted_indices = lengths.sort(descending=True)

    # Sort inputs and targets based on sorted_lengths
    inputs = [inputs[i] for i in sorted_indices]
    targets = [targets[i] for i in sorted_indices]

    # Pad sequences
    inputs_padded = pad_sequence([torch.tensor(i, dtype=torch.float) for i in inputs], batch_first=True)
    targets_padded = pad_sequence([torch.tensor(t, dtype=torch.float) for t in targets], batch_first=True)

    return inputs_padded.to(device), targets_padded.to(device), sorted_lengths.to(device)


