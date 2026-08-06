implement a simple train script to train dynamic

The horizontal length of the LSTM during training is limited to the unrolled steps lunroll = 5, but it will be larger in MCTS as the dynamics process can go deeper. Therefore, we reset the hidden state of LSTM after ζ = 5 steps of recurrent inference, where ζ is the valid horizontal length.

for now, the acton can be just random sampled.

there're two loss here.

one scalar_reward_loss and consist_loss_func

before impl consist_loss_func, you should SimSiam way with projector and predictor

basically like this
L2(sg(P1(st+1)), P2(P1(sˆt+1)))

after finish your work, run the script to actually train the network, make sure everything works fine.
