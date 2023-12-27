import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ChessEnvironment import ChessEnvironment
import time

class ChessAgent(nn.Module):
    def __init__(self):
        super(ChessAgent, self).__init__()
        # Convolutional layers with fewer filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) # 8 filters in the first layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 16 filters in the second layer

        # Fully connected layers with fewer neurons
        self.fc1 = nn.Linear(8 * 8 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1792)  # Output layer for all possible moves
    
    def forward(self, x):
        # Reshape x to (batch_size, channels, height, width)
        x = x.view(-1, 1, 8, 8)  # Adjust shape based on your input data format

        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.out(x)
        move_probs = F.log_softmax(x, dim=1)
        
        return move_probs
        
    def select_move(self, board_state, legal_moves):
        """
        Selects a move based on the current board state and legal moves.
        """
        # Convert the board state to a tensor and run it through the network
        state_tensor = torch.FloatTensor(board_state.copy()).unsqueeze(0).to("cuda")
        with torch.no_grad():  # Ensure no gradients are calculated
            move_probs = self.forward(state_tensor).squeeze()

        # Mask and renormalize the probabilities for legal moves
        mask = torch.zeros_like(move_probs)
        for move in legal_moves:
            mask[move] = 1
        masked_move_probs = move_probs.masked_fill(mask == 0, float('-inf'))
        probs = torch.exp(masked_move_probs)
        sum_masked_probs = probs.sum()
        renormalized_probs = probs / sum_masked_probs

        # Sample a move based on the renormalized probabilities
        move_index = torch.multinomial(renormalized_probs, 1).item()
        return move_index
        
    def select_move_deterministic(self, board_state, legal_moves):
        """
        Selects a move based on the current board state and legal moves.
        """
        # Convert the board state to a tensor and run it through the network
        state_tensor = torch.FloatTensor(board_state.copy()).unsqueeze(0).to("cuda")
        with torch.no_grad():  # Ensure no gradients are calculated
            move_probs = self.forward(state_tensor).squeeze()

        # Mask and renormalize the probabilities for legal moves
        mask = torch.zeros_like(move_probs)
        for move in legal_moves:
            mask[move] = 1
        masked_move_probs = move_probs.masked_fill(mask == 0, float('-inf'))
        probs = torch.exp(masked_move_probs)
        sum_masked_probs = probs.sum()
        renormalized_probs = probs / sum_masked_probs

        # Sample a move based on the renormalized probabilities
        move_index = torch.argmax(renormalized_probs).item()
        return move_index

def train(agent, env, optimizer, loss_function, opponent, num_games):
    is_white = True
    steps = 0
    session_loss = 0
    for game in range(num_games):
        state = env.reset()
        done = False
        game_loss = 0
        game_steps = 0
        
        # Opponent is white and makes first move
        if not is_white:
            legal_moves = env.get_legal_moves_encoded()
            opp_move = opponent.select_move(state, legal_moves)
            state, done = env.step(opp_move)
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state.copy()).unsqueeze(0).to("cuda")

            # Forward pass to get move probabilities
            move_probabilities = agent(state_tensor).squeeze()

            # Mask illegal moves and sample a move
            legal_moves = env.get_legal_moves_encoded()
            move_index = mask_and_sample(move_probabilities, legal_moves)

            # Make the move and get the new state and calculate reward
            score_before = env.calculate_score()
            next_state, done = env.step(move_index)
                
            # Opponent makes move
            if (not done):
                legal_moves = env.get_legal_moves_encoded()
                opp_move = opponent.select_move(next_state, legal_moves)
                next_state, done = env.step(opp_move)
                
            score_after = env.calculate_score()
            # Reward is calculated based on the board score after opponent's move
            reward = score_after - score_before
            
            # Change symbol for black
            if not is_white:
                reward = -reward

            # Compute loss
            target = torch.tensor([reward], dtype=torch.float).to("cuda")
            loss = loss_function(move_probabilities[move_index].unsqueeze(0), target)
            game_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            game_steps += 1
        
        is_white = not is_white
        
        steps += game_steps
        session_loss += game_loss
    
    return session_loss / steps

def mask_and_sample(move_probabilities, legal_moves):
    # Create a mask for legal moves
    mask = torch.zeros_like(move_probabilities)
    for move in legal_moves:
        mask[move] = 1
        
    # Apply the mask and re-normalize
    masked_move_probabilities = move_probabilities.masked_fill(mask == 0, float('-inf'))
    
    # Convert log probabilities to probabilities
    probs = torch.exp(masked_move_probabilities)
    
    # Renormalize the probabilities so they sum up to 1
    sum_masked_probs = probs.sum()
    renormalized_probs = probs / sum_masked_probs
    
    if torch.any(renormalized_probs.isnan()) or torch.any(renormalized_probs.isinf()) or torch.any(renormalized_probs < 0):
        print("Invalid probabilities detected")
        print(legal_moves)
        print(len(renormalized_probs[renormalized_probs > 0]), len(legal_moves))

    # Sample from these probabilities
    sample_index = torch.multinomial(renormalized_probs, 1)
    
    return sample_index.item()

def save_model(model, games, is_latest):
    filename = f"chess_model_games_{games}.pt"
    torch.save(model.state_dict(), filename)
    
def load_model(model, filename):
    # model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    model.name = filename
    model.rating = 0
    model.to("cuda")

def do_training():
    env = ChessEnvironment()
    agent = ChessAgent()
    agent.to("cuda")
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    opponent = ChessAgent()
    load_model(opponent, f"chess_model_games_latest.pt")
    
    loop_index = 1
    # Don't want it to go indefinitely yet
    while loop_index < 10:
        start = time.time()
        avg_loss = train(agent, env, optimizer, loss_function, opponent, 100)
        end = time.time()
        print(avg_loss, end-start)
        save_model(agent, 10 * loop_index, True)
        loop_index += 1


#do_training()