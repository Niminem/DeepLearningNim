type
    ShowerEnv* = object
        state*: int
        shower_length*: int
        game_no*: int

proc initShowerEnv*(): ShowerEnv =
    result.shower_length = 60
    result.state = 38 + rand(-3 .. 3)

proc step*(env: var ShowerEnv, action: int): (int,int,bool) =
    # appy action
    env.state += action - 1
    # reduce shower length by 1
    env.shower_length -= 1
    # calculate reward
    var reward = -1
    if env.state >= 37 and env.state <= 39: reward = 1
    # check if shower is done
    var done = false
    if env.shower_length <= 0: done = true
    # apply temperature noise
    env.state += rand(-1 .. 1)
    if env.state < 0: env.state = 0
    if env.state > 100: env.state = 100
    #return step information
    result = (env.state, reward, done)

proc render*(env: var ShowerEnv) =
    echo "Shower Temp: " & $env.state

proc reset*(env: var ShowerEnv) =
    # reset temperature
    env.state = 38 + rand(-3 .. 3)
    # reset shower length
    env.shower_length = 60