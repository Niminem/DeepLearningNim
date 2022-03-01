import sequtils
import parameters

include agent
include environment

var
    record, score, totalscore = 0
    agent = DQNAgent()
    game = initShowerEnv()

var episode = 1
while episode != EPISODES:
    # get old state
    var state_old = game.state

    # get move
    var state_old_fixed = state_old.float32 * 0.01.float32
    var final_move = get_action(agent, state_old_fixed)

    # perform move and get ---> new state, the reward, and whether the game is over
    var (state_new, reward, done) = game.step(maxIndex(final_move))

    # update score
    score += reward

    # train short memory / current step
    train_step(state_old_fixed, maxIndex(final_move), reward, state_new.float32 * 0.01.float32, done)

    # remember
    agent.remember(state_old_fixed, maxIndex(final_move), reward, state_new.float32 * 0.01.float32, done)

    if done:

        echo "episode: " & $episode & " score: " & $score & " record: " & $record
        echo "total score: " & $totalscore

        # increment number of games
        agent.n_games += 1
        game.game_no += 1

        # increment episode and total score
        total_score += score
        episode += 1

        # update record
        if score > record: record = score

        # reset game
        game.reset()
        score = 0

        # train long memory
        train_long_memory(agent)