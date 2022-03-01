import arraymancer, random
import parameters


# randomize()

# autograd context / neuralnet graph
let ctx* = newContext Tensor[float32]

# define agent type
type
    DQNAgent* = object
        memory*: seq[tuple[
            state: float32, action: int, reward: int, nextState: float32, done: bool]]
        epsilon*: float
        n_games*: int
        n_steps*: int

# define model type
network ctx, LinearQNet:
    layers:
        hidden: Linear(STATE_SIZE, HIDDEN_LAYER_SIZE)
        output: Linear(HIDDEN_LAYER_SIZE, ACTION_SIZE)
    forward x:
        x.hidden.relu.output

let model* = ctx.init(LinearQNet)
var optim* = model.optimizerAdam(learningRate = LEARNING_RATE)


# procedure to create memory for agent
proc remember*(agent: var DQNAgent, state: float32, action: int, reward: int,
            nextState: float32, done: bool) =
    agent.memory.add((state, action, reward, nextState, done))

    if agent.memory.len > MAX_MEMORY: agent.memory.delete(0)


# procedure for agent to act
proc get_action*(agent: var DQNAgent, state: float32): seq[int] =

    result = @[0,0,0]
    agent.epsilon = 100 - agent.n_games.tofloat
    if rand(0 .. 200) < agent.epsilon.toInt:
        var move = rand(0 .. 2)
        result[move] = 1
    else:
        ctx.no_grad_mode:
            let state0 = ctx.variable([[state]].toTensor(), requiresGrad = false)
            let qValues = model.forward(state0).value
        result[qValues.argmax(1)[0,0]] = 1

    agent.n_steps += 1


proc train_step*(state: float32, action: int, reward: int, nextState: float32, done: bool) =

    let stateAsTensor = ctx.variable([[state]].toTensor(), requiresGrad = false)
    let pred = model.forward(stateAsTensor)

    ctx.no_grad_mode:
        let nextStateAsTensor = ctx.variable([[nextState]].toTensor(), requiresGrad = false)
        let test = model.forward(nextStateAsTensor)

    var target = pred.value.clone()
    var q_new: float32 = (reward.toFloat).float32
    if not done:
        q_new = (reward.toFloat + GAMMA * test.value.max).float32
    target[0, action] = q_new

    var loss = mse_loss(pred, target)
    backprop(loss)
    optim.update()


proc trainSteps*(states: seq[float32], actions: seq[int], rewards: seq[int], nextStates: seq[float32], dones: seq[bool]) =

    var teststates: seq[seq[float32]]
    for state in states:
        teststates.add(@[state])
    var testnextstate: seq[seq[float32]]
    for state in nextStates:
        testnextstate.add(@[state])

    let stateTensor = teststates.toTensor()
    let stateAsTensor = ctx.variable(stateTensor, requires_grad=false)
    let pred = model.forward(stateAsTensor)

    var target = pred.value.clone()

    for i in 0 .. dones.high:
        var q_new: float32 = (rewards[i].toFloat).float32
        if not dones[i]:
            let cl = testnextstate[i].toTensor().reshape(1,1)
            ctx.no_grad_mode:
                let nextStateAsTensor = ctx.variable(cl, requires_grad=false)
                q_new = rewards[i].toFloat + GAMMA * model.forward(nextStateAsTensor).value.max
                target[i, actions[i]] = q_new

    var loss = mse_loss(pred, target)
    loss.backprop()
    optim.update()


proc train_long_memory*(agent: var DQNAgent) =

    let batchsize = if agent.memory.len >= BATCH_SIZE: BATCH_SIZE else: agent.memory.len
    var batch: seq[tuple[state: float32, action: int,
                reward: int, nextState: float32, done: bool]]

    for i in 1 .. batchsize:
        var samp = sample(agent.memory)
        batch.add(samp)

    var
        states: seq[float32]
        actions: seq[int]
        rewards: seq[int]
        nextStates: seq[float32]
        dones: seq[bool]

    for samp in batch:
        states.add(samp.state)
        actions.add(samp.action)
        rewards.add(samp.reward)
        nextStates.add(samp.nextState)
        dones.add(samp.done)

    trainSteps(states, actions, rewards, nextStates, dones)