import events as e
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUMBER_OF_ACTIONS = 6
NUMBER_OF_FEATURE = 12

# Mini-batch
BATCH_SIZE = 128  # Smaller batch size for faster updates
TRANSITION_BUFFER_SIZE_MAX = 1024
# Q-learning
STEP_ALPHA = 0.1  # Lower step size for cautious learning
DECAY_GAMMA_VALUE = 0.95  # Slightly lower gamma for shorter-term focus
N_STEP_LEARNING_RATE = True
N_STEP = 3
PRIORITY_LEARNING = False
PRIORITY_RATIO = 0.2

# Epsilon-greedy policy
EPSILON_START_VALUE = 0.5  # Higher initial exploration rate
EPSILON_END_VALUE = 0.05  # Lower final exploration rate
EPSILON_DECAY_VALUE = 0.999  # Slightly slower epsilon decay

# Models
FINAL_MODEL_NAME = "model-1000"
SAVE_MODEL = 100
TRAINING_ROUNDS = 200  # Increased training rounds for more learning

