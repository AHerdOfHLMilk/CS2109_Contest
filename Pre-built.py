import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from IPython.display import HTML
from base64 import b64encode


config_dict = {
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'n_filters': 128,              # Number of convolutional filters used in residual blocks
    'n_res_blocks': 8,             # Number of residual blocks used in network
    'exploration_constant': 2,     # Exploration constant used in PUCT calculation
    'temperature': 1.25,           # Selection temperature. A greater temperature is a more uniform distribution
    'dirichlet_alpha': 1.,         # Alpha parameter for Dirichlet noise. Larger values mean more uniform noise
    'dirichlet_eps': 0.25,         # Weight of dirichlet noise
    'learning_rate': 0.001,        # Adam learning rate
    'training_epochs': 100,         # How many full training epochs
    'games_per_epoch': 100,        # How many self-played games per epoch
    'minibatch_size': 128,         # Size of each minibatch used in learning update 
    'n_minibatches': 4,            # How many minibatches to accumulate per learning step
    'mcts_start_search_iter': 30,  # Number of Monte Carlo tree search iterations initially
    'mcts_max_search_iter': 150,   # Maximum number of MCTS iterations
    'mcts_search_increment': 1,    # After each epoch, how much should search iterations be increased by
    'early_stop_visits': 75  # Add this line
    }

# Convert to a struct esque object
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

config = Config(config_dict)\

class Connect4:
    "Connect 4 game engine, containing methods for game-related tasks."
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def get_next_state(self, state, action, to_play=1):
        "Play an action in a given state and return the resulting board."
        # Pre-condition checks
        assert self.evaluate(state) == 0
        assert np.sum(abs(state)) != self.rows * self.cols
        assert action in self.get_valid_actions(state)
        
        # Identify next empty row in column
        row = np.where(state[:, action] == 0)[0][-1]
        
        # Apply action
        new_state = state.copy()
        new_state[row, action] = to_play
        return new_state

    def get_valid_actions(self, state):
        "Return a numpy array containing the indices of valid actions."
        # If game over, no valid moves
        if self.evaluate(state) != 0:
            return np.array([])
        
        # Identify valid columns to play
        cols = np.sum(np.abs(state), axis=0)
        return np.where((cols // self.rows) == 0)[0]

    def evaluate(self, state):
        "Evaluate the current position. Returns 1 for player 1 win, -1 for player 2 and 0 otherwise."
        # Kernels for checking win conditions
        kernel = np.ones((1, 4), dtype=int)
        
        # Horizontal and vertical checks
        horizontal_check = convolve2d(state, kernel, mode='valid')
        vertical_check = convolve2d(state, kernel.T, mode='valid')

        # Diagonal checks
        diagonal_kernel = np.eye(4, dtype=int)
        main_diagonal_check = convolve2d(state, diagonal_kernel, mode='valid')
        anti_diagonal_check = convolve2d(state, np.fliplr(diagonal_kernel), mode='valid')
        
        # Check for winner
        if any(cond.any() for cond in [horizontal_check == 4, vertical_check == 4, main_diagonal_check == 4, anti_diagonal_check == 4]):
            return 1
        elif any(cond.any() for cond in [horizontal_check == -4, vertical_check == -4, main_diagonal_check == -4, anti_diagonal_check == -4]):
            return -1

        # No winner
        return 0  

    def step(self, state, action, to_play=1):
        "Play an action in a given state. Return the next_state, reward and done flag."
        # Get new state and reward
        next_state = self.get_next_state(state, action, to_play)
        reward = self.evaluate(next_state)
        
        # Check for game termination
        done = True if reward != 0 or np.sum(abs(next_state)) >= (self.rows * self.cols - 1) else False
        return next_state, reward, done

    def encode_state(self, state):
        "Convert state to tensor with 3 channels."
        encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def reset(self):
        "Reset the board."
        return np.zeros([self.rows, self.cols], dtype=np.int8)
    
class ResNet(nn.Module):
    "Complete residual neural network model."
    def __init__(self, game, config):
        super().__init__()

        # Board dimensions
        self.board_size = (game.rows, game.cols)
        n_actions = game.cols  # Number of columns represent possible actions
        n_filters = config.n_filters
        
        self.base = ConvBase(config)  # Base layers

        # Policy head for choosing actions
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//4 * self.board_size[0] * self.board_size[1], n_actions)
        )

        # Value head for evaluating board states
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//32 * self.board_size[0] * self.board_size[1], 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.base(x) 
        x_value = self.value_head(x)
        x_policy = self.policy_head(x)
        return x_value, x_policy

class ConvBase(nn.Module):
    "Convolutional base for the network."
    def __init__(self, config):
        super().__init__()
        
        n_filters = config.n_filters
        n_res_blocks = config.n_res_blocks

        # Initial convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )

        # List of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )

    def forward(self, x):
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x

class ResidualBlock(nn.Module):
    "Residual block, the backbone of a ResNet."
    def __init__(self, n_filters):
        super().__init__()

        # Two convolutional layers, both with batch normalization
        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass x through layers and add skip connection
        output = self.relu(self.batch_norm_1(self.conv_1(x)))
        output = self.batch_norm_2(self.conv_2(output))
        return self.relu(output + x)
    
class ConvBaseWithDilations(nn.Module):
    "Convolutional base with dilated convolutions for broader spatial capture."
    def __init__(self, config):
        super().__init__()
        
        n_filters = config.n_filters
        n_res_blocks = config.n_res_blocks

        # Initial convolutional layer without dilation
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )

        # Alternating residual blocks with standard and dilated convolutions
        self.res_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            if i % 2 == 0:
                # Add a standard residual block
                self.res_blocks.append(ResidualBlock(n_filters))
            else:
                # Add a residual block with dilation (e.g., dilation=2)
                self.res_blocks.append(DilatedResidualBlock(n_filters, dilation=2))

    def forward(self, x):
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x

class DilatedResidualBlock(nn.Module):
    "Residual block with dilated convolutions to increase receptive field."
    def __init__(self, n_filters, dilation=2):
        super().__init__()
        
        # Two convolutional layers with batch normalization and specified dilation
        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters)
        
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass x through layers and add skip connection
        output = self.relu(self.batch_norm_1(self.conv_1(x)))
        output = self.batch_norm_2(self.conv_2(output))
        return self.relu(output + x)

class ResNetWithDilations(nn.Module):
    "Complete ResNet model with base containing dilated convolutions."
    def __init__(self, game, config):
        super().__init__()

        # Board dimensions
        self.board_size = (game.rows, game.cols)
        n_actions = game.cols  # Number of columns represent possible actions
        n_filters = config.n_filters
        
        # Using modified convolutional base with dilated convolutions
        self.base = ConvBaseWithDilations(config)

        # Policy head for choosing actions
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//4 * self.board_size[0] * self.board_size[1], n_actions)
        )

        # Value head for evaluating board states
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//32 * self.board_size[0] * self.board_size[1], 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.base(x) 
        x_value = self.value_head(x)
        x_policy = self.policy_head(x)
        return x_value, x_policy
    
class MCTS:
    def __init__(self, network, game, config):
        "Initialize Monte Carlo Tree Search with a given neural network, game instance, and configuration."
        self.network = network
        self.game = game
        self.config = config
        self.early_stop_visits = config.early_stop_visits  # Default threshold for early stopping
        self.base_temperature = config.temperature  # Starting temperature

    def search(self, state, total_iterations, temperature=None):
        "Performs a search for the desired number of iterations, returns an action and the tree root."
        # Create the root
        root = Node(None, state, 1, self.game, self.config)

        # Initial expansion of the root with Dirichlet noise for exploration
        valid_actions = self.game.get_valid_actions(state)
        state_tensor = torch.tensor(self.game.encode_state(state), dtype=torch.float).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            self.network.eval()
            value, logits = self.network(state_tensor)
        action_probs = F.softmax(logits.view(self.game.cols), dim=0).cpu().numpy()
        
        # Adding Dirichlet noise for exploration
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.game.cols)
        action_probs = ((1 - self.config.dirichlet_eps) * action_probs) + self.config.dirichlet_eps * noise
        action_probs /= np.sum(action_probs)  # Normalize probabilities

        for action, prob in zip(valid_actions, action_probs[valid_actions]):
            child_state = -self.game.get_next_state(state, action)
            root.children[action] = Node(root, child_state, -1, self.game, self.config)
            root.children[action].prob = prob

        root.n_visits = 1
        root.total_score = value.item()

        for i in range(total_iterations):
            current_node = root
            temp_factor = self.adjust_temperature(i, total_iterations)

            # Early stopping check
            if current_node.n_visits >= self.early_stop_visits:
                break

            # Selection phase: Traverse tree to select a node
            while not current_node.is_leaf():
                current_node = current_node.select_child()

            # Expansion phase: Expand a leaf node
            if not current_node.is_terminal():
                current_node.expand()
                state_tensor = torch.tensor(self.game.encode_state(current_node.state), dtype=torch.float).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    self.network.eval()
                    value, logits = self.network(state_tensor)
                    value = value.item()

                mask = np.full(self.game.cols, False)
                mask[valid_actions] = True
                action_probs = F.softmax(logits.view(self.game.cols)[mask], dim=0).cpu().numpy()

                for child, prob in zip(current_node.children.values(), action_probs):
                    child.prob = prob
            else:
                value = self.game.evaluate(current_node.state)

            # Backpropagation with weighted values
            self.weighted_backpropagate(current_node, value)

        # Select the best action based on adjusted temperature
        return self.select_action(root, temp_factor), root

    def adjust_temperature(self, iteration, total_iterations):
        "Dynamically adjusts temperature based on the current iteration and total iterations."
        progress = iteration / total_iterations
        return self.base_temperature * (1 - progress) + 0.1 * progress  # Example linear decay toward 0.1

    def weighted_backpropagate(self, node, value):
        "Backpropagate with weights applied based on node depth."
        depth = 0
        while node is not None:
            weight = 1 / (1 + depth)  # Weight decreases with depth
            node.total_score += weight * value
            node.n_visits += 1
            value = -value  # Alternate value sign for opponent's perspective
            node = node.parent
            depth += 1

    def select_action(self, root, temperature):
        "Select an action from the root based on visit counts, adjusted by temperature."
        action_counts = {key: val.n_visits for key, val in root.children.items()}
        if temperature == 0:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            distribution = np.array([*action_counts.values()]) ** (1 / temperature)
            return np.random.choice([*action_counts.keys()], p=distribution / sum(distribution))

class Node:
    def __init__(self, parent, state, to_play, game, config):
        "Represents a node in the MCTS, holding the game state and statistics for MCTS to operate."
        self.parent = parent
        self.state = state
        self.to_play = to_play
        self.config = config
        self.game = game

        self.prob = 0
        self.children = {}
        self.n_visits = 0
        self.total_score = 0

    def expand(self):
        "Create child nodes for all valid actions. If state is terminal, evaluate and set the node's value."
        # Get valid actions
        valid_actions = self.game.get_valid_actions(self.state)

        # If there are no valid actions, state is terminal, so get value using game instance
        if len(valid_actions) == 0:
            self.total_score = self.game.evaluate(self.state)
            return

        # Create a child for each possible action
        for action in zip(valid_actions):
            # Make move, then flip board to perspective of next player
            child_state = -self.game.get_next_state(self.state, action)
            self.children[action] = Node(self, child_state, -self.to_play, self.game, self.config)

    def select_child(self):
        "Select the child node with the highest PUCT score."
        best_puct = -np.inf
        best_child = None
        for child in self.children.values():
            puct = self.calculate_puct(child)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child

    def calculate_puct(self, child):
        "Calculate the PUCT score for a given child node."
        # Scale Q(s,a) so it's between 0 and 1 so it's comparable to a probability
        # Using 1 - Q(s,a) because it's from the perspectve of the child – the opposite of the parent
        exploitation_term = 1 - (child.get_value() + 1) / 2
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + self.config.exploration_constant * exploration_term

    def backpropagate(self, value):
        "Update the current node and its ancestors with the given value."
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            # Backpropagate the negative value so it switches each level
            self.parent.backpropagate(-value)

    def is_leaf(self):
        "Check if the node is a leaf (no children)."
        return len(self.children) == 0

    def is_terminal(self):
        "Check if the node represents a terminal state."
        return (self.n_visits != 0) and (len(self.children) == 0)

    def get_value(self):
        "Calculate the average value of this node."
        if self.n_visits == 0:
            return 0
        return self.total_score / self.n_visits
    
    def __str__(self):
        "Return a string containing the node's relevant information for debugging purposes."
        return (f"State:\n{self.state}\nProb: {self.prob}\nTo play: {self.to_play}" +
                f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}" +
                f"\nTotal score: {self.total_score}")
    
class AlphaZero:
    def __init__(self, game, config, verbose=True):
        self.network = ResNet(game, config).to(config.device)
        self.mcts = MCTS(self.network, game, config)
        self.game = game
        self.config = config

        # Losses and optimizer
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, weight_decay=0.0001)

        # Pre-allocate memory on GPU
        state_shape = game.encode_state(game.reset()).shape
        self.max_memory = config.minibatch_size * config.n_minibatches
        self.state_memory = torch.zeros(self.max_memory, *state_shape).to(config.device)
        self.value_memory = torch.zeros(self.max_memory, 1).to(config.device)
        self.policy_memory = torch.zeros(self.max_memory, game.cols).to(config.device)
        self.current_memory_index = 0
        self.memory_full = False

        # MCTS search iterations
        self.search_iterations = config.mcts_start_search_iter
        
        # Logging
        self.verbose = verbose
        self.total_games = 0

    def train(self, training_epochs):
        "Train the AlphaZero agent for a specified number of training epochs."
        # For each training epoch
        for _ in range(training_epochs):

            # Play specified number of games
            for _ in range(self.config.games_per_epoch):
                self.self_play()
            
            # At the end of each epoch, increase the number of MCTS search iterations
            self.search_iterations = min(self.config.mcts_max_search_iter, self.search_iterations + self.config.mcts_search_increment)

    def self_play(self):
        "Perform one episode of self-play."
        state = self.game.reset()
        done = False
        while not done:
            # Search for a move
            action, root = self.mcts.search(state, self.search_iterations)

            # Value target is the value of the MCTS root node
            value = root.get_value()

            # Visit counts used to compute policy target
            visits = np.zeros(self.game.cols)
            for child_action, child in root.children.items():
                visits[child_action] = child.n_visits
            # Softmax so distribution sums to 1
            visits /= np.sum(visits)

            # Append state + value & policy targets to memory
            self.append_to_memory(state, value, visits)

            # If memory is full, perform a learning step
            if self.memory_full:
                self.learn()

            # Perform action in game
            state, _, done = self.game.step(state, action)

            # Flip the board
            state = -state

        # Increment total games played
        self.total_games += 1

        # Logging if verbose
        if self.verbose:
            print("\rTotal Games:", self.total_games, "Items in Memory:", self.current_memory_index, "Search Iterations:", self.search_iterations, end="")

    def append_to_memory(self, state, value, visits):
        """
        Append state and MCTS results to memory buffers.
        Args:
            state (array-like): Current game state.
            value (float): MCTS value for the game state.
            visits (array-like): MCTS visit counts for available moves.
        """
        # Calculate the encoded states
        encoded_state = np.array(self.game.encode_state(state))
        encoded_state_augmented = np.array(self.game.encode_state(state[:, ::-1]))

        # Stack states and visits
        states_stack = np.stack((encoded_state, encoded_state_augmented), axis=0)
        visits_stack = np.stack((visits, visits[::-1]), axis=0)

        # Convert the stacks to tensors
        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.config.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.config.device)
        value_tensor = torch.tensor(np.array([value, value]), dtype=torch.float).to(self.config.device).unsqueeze(1)

        # Store in pre-allocated GPU memory
        self.state_memory[self.current_memory_index:self.current_memory_index + 2] = state_tensor
        self.value_memory[self.current_memory_index:self.current_memory_index + 2] = value_tensor
        self.policy_memory[self.current_memory_index:self.current_memory_index + 2] = visits_tensor

        # Increment index, handle overflow
        self.current_memory_index = (self.current_memory_index + 2) % self.max_memory

        # Set memory filled flag to True if memory is full
        if (self.current_memory_index == 0) or (self.current_memory_index == 1):
            self.memory_full = True


    def learn(self):
        "Update the neural network by extracting minibatches from memory and performing one step of optimization for each one."
        self.network.train()

        # Create a randomly shuffled list of batch indices
        batch_indices = np.arange(self.max_memory)
        np.random.shuffle(batch_indices)

        for batch_index in range(self.config.n_minibatches):
            # Get minibatch indices
            start = batch_index * self.config.minibatch_size
            end = start + self.config.minibatch_size
            mb_indices = batch_indices[start:end]

            # Slice memory tensors
            mb_states = self.state_memory[mb_indices]
            mb_value_targets = self.value_memory[mb_indices]
            mb_policy_targets = self.policy_memory[mb_indices]

            # Network predictions
            value_preds, policy_logits = self.network(mb_states)

            # Loss calculation
            policy_loss = self.loss_cross_entropy(policy_logits, mb_policy_targets)
            value_loss = self.loss_mse(value_preds.view(-1), mb_value_targets.view(-1))
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory_full = False
        self.network.eval()
    
class Evaluator:
    "Class to evaluate the policy network's performance on simple moves."
    def __init__(self, alphazero, num_examples=500, verbose=True):
        self.network = alphazero.network
        self.game = alphazero.game
        self.config = alphazero.config
        self.accuracies = []
        self.num_examples = num_examples
        self.verbose = verbose

        # Generate and prepare example states and actions for evaluation
        self.generate_examples()

    def select_action(self, state):
        "Select an action based on the given state, will choose a winning or blocking moves."
        valid_actions = self.game.get_valid_actions(state)
        
        # Check for a winning move
        for action in valid_actions:
            next_state, reward, _ = self.game.step(state, action)
            if reward == 1:
                return action

        # Check for a blocking move
        flipped_state = -state
        for action in valid_actions:
            next_state, reward, _ = self.game.step(flipped_state, action)
            if reward == 1:
                return action

        # Default to random action if no winning or blocking move
        return random.choice(valid_actions)

    def generate_examples(self):
        "Generate and prepare example states and actions for evaluation."
        winning_examples = self.generate_examples_for_condition('win')
        blocking_examples = self.generate_examples_for_condition('block')

        # Prepare states and actions for evaluation
        winning_example_states, winning_example_actions = zip(*winning_examples)
        blocking_example_states, blocking_example_actions = zip(*blocking_examples)

        target_states = np.concatenate([winning_example_states, blocking_example_states], axis=0)
        target_actions = np.concatenate([winning_example_actions, blocking_example_actions], axis=0)

        encoded_states = [self.game.encode_state(state) for state in target_states]
        self.X_target = torch.tensor(np.stack(encoded_states, axis=0), dtype=torch.float).to(self.config.device)
        self.y_target = torch.tensor(target_actions, dtype=torch.long).to(self.config.device)

    def generate_examples_for_condition(self, condition):
        "Generate examples based on either 'win' or 'block' conditions."
        examples = []
        while len(examples) < self.num_examples:
            state = self.game.reset()
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.game.step(state, action, to_play=1)
                
                if condition == 'win' and reward == 1:
                    examples.append((state, action))
                    break
                
                if done:
                    break
                
                state = next_state

                # Flipping the board for opponent's perspective
                action = self.select_action(-state)
                next_state, reward, done = self.game.step(state, action, to_play=-1)
                
                if condition == 'block' and reward == -1:
                    examples.append((-state, action))
                    break
                
                if done:
                    break
                
                state = next_state
        return examples

    def evaluate(self):
        "Evaluate the policy network's accuracy and append it to self.accuracies."
        with torch.no_grad():
            self.network.eval()
            _, logits = self.network(self.X_target)
            pred_actions = logits.argmax(dim=1)
            accuracy = (pred_actions == self.y_target).float().mean().item()
        
        self.accuracies.append(accuracy)
        if self.verbose:
            print(f"Initial Evaluation Accuracy: {100 * accuracy:.1f}%")

game = Connect4()
alphazero = AlphaZero(game, config)
evaluator = Evaluator(alphazero)

# Evaluate pre training
evaluator.evaluate()

# Main training/eval loop
for _ in range(config.training_epochs):
    alphazero.train(1)
    evaluator.evaluate()

# Save trained weights
torch.save(alphazero.network.state_dict(), 'alphazero-network-weights.pth')