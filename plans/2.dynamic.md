dynamic network predict next hidden states given current states and actions

we add an extra residual link in the dynamics part to keep the information of historical hidden states during recurrent inference. The design of the dynamics network is listed here:

• Concatenate the input states and input actions into 65 planes.
• 1 convolution with stride 2 and 64 output planes. (BN)
• A residual link: add up the output and the input states. (ReLU)
• 1 residual block with 64 planes.

you can also check the code in efficientzero repos.

you should also impl reward prediction in dynanmic network. Considering the stability of the prediction part, we set the weights and bias of the last layer to zero in prediction networks. As for the reward prediction network, it predicts the sum of the rewards, namely value prefix: rt, ht+1 = R(sˆt+1, ht), where rt is the predicted sum of rewards, h0 is zero-initialized and hidden size of LSTM is 512.

• 1 1x1convolution and 16 output planes. (BN + ReLU)
• Flatten. • LSTM with 512 hidden size. (BN + ReLU)
• 1 fully connected layers and 32 output dimensions. (BN + ReLU)
• 1 fully connected layers and 601 output dimensions.
