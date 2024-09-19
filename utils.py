import time
import timeout_decorator
from connect_four import ConnectFour

WRAPPED_TIME_LIMIT = 2
TIME_LIMIT = 1

def wrap_test(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            return "Code not implemented: Implement your code."
        except timeout_decorator.TimeoutError as e:
            return f"Out of time: Your agent took too long, exceeds {TIME_LIMIT} second(s)."
        except Exception as e:
            return f"Failed, reason: {str(e)}"
    return inner

# sequence of actions to connect4 board
def actions_to_board(seq_actions):
    tc_board = ConnectFour()
    for i, col_id in enumerate(seq_actions):
        current_player_id = (i % 2) + 1
        tc_board.step((col_id, current_player_id))
    return tc_board

@wrap_test
@timeout_decorator.timeout(WRAPPED_TIME_LIMIT)
def check_step(board, player_id, AgentClazz):
    message = "Pass"
    start = time.process_time()
    try:
        agent = AgentClazz(player_id=player_id)
        col_id = agent.make_move(board.get_state())
        board.step((col_id, player_id))
    except ValueError as e:
        message = str(e)
    end = time.process_time()
    move_time = end - start
    
    if move_time > TIME_LIMIT:
        message = f"Out of time: Your agent took too long, exceeds {TIME_LIMIT} second(s)."
    
    return message
