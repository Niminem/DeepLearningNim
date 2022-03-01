## parameters for the model, agent, and environment

const

    ## for model
    HIDDEN_LAYER_SIZE* = 100       ## *hyperparameter* / size of Hidden layer
    STATE_SIZE* = 1                ## size of state at any given time in environment / size of Input layer
    ACTION_SIZE* = 3               ## amount of actions the agent can take / size of Output layer

    ## for agent
    LEARNING_RATE* = 0.006'f32     ## *hyperparameter* / learning rate (size of step for gradient descent)
    MAX_MEMORY* = 2000             ## maximum number of states to remember / size of memory (oldest states are then replaced)
    BATCH_SIZE* = 60               ## number of states to train on in a batch
    GAMMA* = 0.99                  ## *hyperparameter* / discount factor of future rewards (emphasis on short-term rewards)

    ## for environment
    EPISODES* = 3000               ## for limiting the number of games in the training