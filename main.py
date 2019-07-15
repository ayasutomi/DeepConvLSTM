from deepConvLSTM import deepConvLSTM
import data_handler as data
import os
# from keras.optimizers import Adam
import preprocess_data
import tensorflow as tf

if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 18
    batch_size = 85
    num_features = 113
    sliding_window_length = 24
    sliding_window_step = 12
    # timesteps = 24
    rnn_size = 258
    max_len = 150
    learn_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    exist = os.path.isfile('./data/oppChallenge_gestures.data')
    if not exist:
        # Preprocess original OPPORTUNITY zip file
        preprocess_data.generate_data("./data/OpportunityUCIDataset.zip","oppChallenge_gestures.data",'gestures')
    
    # Load dataset
    inputs, labels, test_inputs, test_labels = data.load_data("./data/oppChallenge_gestures.data")

    # Divide data into sliding windows of length "sliding_window_length" with steps of "sliding_window_step"
    inputs, labels = data.opp_sliding_window(inputs, labels, sliding_window_length, sliding_window_step)
    test_inputs, test_labels = data.opp_sliding_window(test_inputs, test_labels, sliding_window_length, sliding_window_step)

    # Create Placeholders of shape (n_x, n_y)
    X = tf.placeholder(tf.float32, [None, sliding_window_length, 1, num_features], name="X")
    Y = tf.placeholder(tf.float32, [None, n_classes], name="Y")

    # Initialize parameters
    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    # Construct model
    model = deepConvLSTM(n_classes, X.shape )
    logits = model._create_model(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(inputs.shape(0) / batch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = data.random_mini_batches(inputs, labels, batch_size, seed)

            for step,minibatch in enumerate(minibatches):

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

                print("Step " + str(step) + ", Minibatch Cost= " + \
                  "{:.4f}".format(minibatch_cost) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
                
            print("Optimization Finished!")

    # Calculate accuracy for 256 inputs
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_inputs[:256],
                                      Y: test_labels[:256]}))

