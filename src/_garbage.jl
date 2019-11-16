"""
    RNNActionCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

"""
#           -----------------------------------
#          |     /--> W_1*[o_{t+1};h_t]-\      |
#          |    /                        \     |
#   h_t    |   /                          \    | h_{t+1}
# -------->|-O------> W_2*[o_{t+1};h_t]----------------->
#          | | \                          /    |
#          | |  \                        /     |
#          | |   \--> W_3*[o_{t+1};h_t]-/      |
#           -|---------------------------------
#            | (o_{t+1}, a_t)
#            |
