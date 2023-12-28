import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAgent(nn.Module):
    def __init__(self):
        super(ChessAgent, self).__init__()
        # Convolutional layers for the board
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for the board
        self.fc1_board = nn.Linear(8 * 8 * 16, 128)

        # Additional inputs layer
        self.fc1_additional = nn.Linear(6, 32)

        # Combined layers
        self.fc2 = nn.Linear(128 + 32, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, board, additional_inputs):
        # Process the board
        x_board = F.relu(self.conv1(board))
        x_board = F.relu(self.conv2(x_board))
        x_board = x_board.view(-1, 8 * 8 * 16)  # Flatten

        x_board = F.relu(self.fc1_board(x_board))

        # Process the additional inputs
        x_additional = F.relu(self.fc1_additional(additional_inputs))

        # Combine and process further
        x_combined = torch.cat((x_board, x_additional), dim=1)
        x_combined = F.relu(self.fc2(x_combined))
        x_combined = self.out(x_combined)
        return x_combined
        
    def get_position_evaluation(self, board_state, additional_inputs):
        # Convert the board state to a tensor and run it through the network
        state_tensor = board_state.unsqueeze(0)
        additional_inputs_tensor = torch.FloatTensor(additional_inputs.copy()).unsqueeze(0).to("cuda")
        with torch.no_grad():  # Ensure no gradients are calculated
            return self.forward(state_tensor, additional_inputs_tensor).item()
        