from game_utils import initialize, step, get_valid_col_id

c4_board = initialize()
print(c4_board.shape)

print(c4_board)

get_valid_col_id(c4_board)

step(c4_board, col_id=2, player_id=1, in_place=True)

print(c4_board)

step(c4_board, col_id=2, player_id=2, in_place=True)
step(c4_board, col_id=2, player_id=1, in_place=True)
step(c4_board, col_id=2, player_id=2, in_place=True)
step(c4_board, col_id=2, player_id=1, in_place=True)
step(c4_board, col_id=2, player_id=2, in_place=True)
print(c4_board)

print(get_valid_col_id(c4_board))

step(c4_board, col_id=2, player_id=1, in_place=True)

class ZeroAgent(object):
    def __init__(self, player_id=1):
        pass
    def make_move(self, state):
        return 0

# Step 1
agent1 = ZeroAgent(player_id=1) # Yours, Player 1
agent2 = ZeroAgent(player_id=2) # Opponent, Player 2

# Step 2
contest_board = initialize()

# Step 3
p1_board = contest_board.view()
p1_board.flags.writeable = False
move1 = agent1.make_move(p1_board)

# Step 4
step(contest_board, col_id=move1, player_id=1, in_place=True)

from simulator import GameController, HumanAgent
from connect_four import ConnectFour

board = ConnectFour()
game = GameController(board=board, agents=[HumanAgent(1), HumanAgent(2)])
game.run()

## Task 1.1 Make a valid move

class AIAgent(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        pass
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_1_1():
    from utils import check_step, actions_to_board
    
    # Test case 1
    res1 = check_step(ConnectFour(), 1, AIAgent)
    assert(res1 == "Pass")
    
    # Test case 2
    res2 = check_step(actions_to_board([0, 0, 0, 0, 0, 0]), 1, AIAgent)
    assert(res2 == "Pass")
    
    # Test case 3
    res2 = check_step(actions_to_board([4, 3, 4, 5, 5, 1, 4, 4, 5, 5]), 1, AIAgent)
    assert(res2 == "Pass")

## Task 2.1: Defeat the Baby Agent

class AIAgent(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        pass
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_2_1():
    assert(True)
    # Upload your code to Coursemology to test it against our agent.

## Task 2.2: Defeat the Base Agent

class AIAgent(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        pass
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_2_2():
    assert(True)
    # Upload your code to Coursemology to test it against our agent.


if __name__ == '__main__':
    test_task_1_1()
    test_task_2_1()
    test_task_2_2()