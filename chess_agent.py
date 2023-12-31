import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAgent(nn.Module):
    def __init__(self):
        super(ChessAgent, self).__init__()
        # Convolutional layers for the board
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 1x1 Convolution for dimension matching in residual connection
        self.match_dimensions = nn.Conv2d(1, 64, kernel_size=1)

        # Fully connected layers for the board
        self.fc1_board = nn.Linear(64 * 8 * 8, 256)

        # Additional inputs layer
        self.fc1_additional = nn.Linear(6, 32)

        # Combined layers
        self.fc2 = nn.Linear(256 + 32, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, board, additional_inputs):
        # Reshape the board input to [batch_size, channels, height, width]
        # Assuming board is a flattened 8x8 board with 1 channel
        board = board.view(-1, 1, 8, 8)  # Reshape to [batch_size, 1, 8, 8]

        # Process the board with residual connection
        identity = board
        x_board = F.relu(self.conv1(board))
        x_board = F.relu(self.conv2(x_board))
        x_board = F.relu(self.conv3(x_board))
        x_board = self.conv4(x_board)
        identity = self.match_dimensions(identity)
        x_board = F.relu(x_board + identity)

        x_board = x_board.view(-1, 64 * 8 * 8)  # Flatten

        x_board = F.relu(self.fc1_board(x_board))

        # Process the additional inputs
        x_additional = F.relu(self.fc1_additional(additional_inputs))

        # Combine and process further
        x_combined = torch.cat((x_board, x_additional), dim=1)
        x_combined = F.relu(self.fc2(x_combined))

        # Output layer with tanh activation for output range of [-1, 1]
        x_combined = torch.tanh(self.out(x_combined))

        return x_combined

    def get_position_evaluation(self, board_state, additional_inputs):
        # Convert the board state to a tensor and run it through the network
        additional_inputs_tensor = torch.tensor(additional_inputs.copy(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():  # Ensure no gradients are calculated
            return self.forward(board_state, additional_inputs_tensor).item()
        
    def get_multiple_position_evaluations(self, board_states, additional_inputs):
        states_tensor = torch.stack(board_states)
        add_states_tensor = torch.stack(additional_inputs)

        with torch.no_grad():  # Ensure no gradients are calculated
            result = self.forward(states_tensor, add_states_tensor).tolist()
        
            return result
        