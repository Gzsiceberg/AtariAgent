implement representation network

here're the details of the representation network:

• 1 convolution with stride 2 and 32 output planes, output resolution 48x48. (BN + ReLU)
• 1 residual block with 32 planes.
• 1 residual downsample block with stride 2 and 64 output planes, output resolution 24x24.
• 1 residual block with 64 planes.
• Average pooling with stride 2, output resolution 12x12. (BN + ReLU)
• 1 residual block with 64 planes.
• Average pooling with stride 2, output resolution 6x6. (BN + ReLU)
• 1 residual block with 64 planes.

where the kernel size is 3 × 3 for all operations.


you can also check the code in efficientzero repos.