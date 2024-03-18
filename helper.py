from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt

def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training
    # Calculate accuracy

    accuracy = np.mean(model.predict(X) == Y)

    print(f"Accuracy: {accuracy * 100:.2f}%")


    if train_flag:
        # Hyperparameters
        learning_rate = 0.005
        epochs = 1000

        # Lists to store the cost for plotting
        costs = []
        j=0
        for epoch in range(epochs):

            w_acc1 = np.zeros((150, 487))
            w_acc2 = np.zeros((7, 150))
            b_acc1 = np.zeros((150, 1))
            b_acc2 = np.zeros((7, 1))

            loss_acc = np.zeros((1000,))
            i=0
            for row_X, row_Y in zip(X, Y):

                a2 = model.forward(X)
                loss = compute_loss(a2[i],row_Y)
                i += 1
                loss_acc = loss_acc + loss

                dW1, db1, dW2, db2 = model.backward(X, Y)

                w_acc1 = w_acc1 + dW1
                w_acc2 = w_acc2 + dW2

                b_acc1 = b_acc1 + db1
                b_acc2 = b_acc2 + db2


            cost = loss_acc/1000
            model.W1 = model.W1 - (learning_rate/1000) * np.array(w_acc1).T
            model.W2 = model.W2 - (learning_rate / 1000) * w_acc2
            model.b1 = model.b1 - (learning_rate / 1000) * b_acc1
            model.b2 = model.b2 - (learning_rate / 1000) * b_acc2

            # Store the cost for plotting
            costs.append(cost)

            # Print the cost every 100 epochs

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
        # Plot the cost function
        costs = [abs(x) for x in costs]

        plt.plot(costs)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function Over Epochs')

        plt.savefig("plot"+str(j)+".png")
        j+=1


        # Calculate accuracy after training
        accuracy = np.mean(model.predict(X) == Y)
        print(f"Accuracy after training: {accuracy * 100:.2f}%")

    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.

    if train_flag:
        # Hyperparameters
        learning_rate = 0.005
        epochs = 1000
        batch_size = 64

        # Lists to store the cost for plotting
        costs = []
        j = 99

        for epoch in range(epochs):
            w_acc1 = np.zeros((150, 487))
            w_acc2 = np.zeros((7, 150))
            b_acc1 = np.zeros((150, 1))
            b_acc2 = np.zeros((7, 1))

            loss_acc = 0

            # Shuffle the data for each epoch
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation, :]
            Y = Y[permutation, :]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size, :]
                Y_batch = Y[i:i + batch_size, :]

                a2 = model.forward(X_batch)
                loss = compute_loss(a2, Y_batch)
                loss_acc += loss

                dW1, db1, dW2, db2 = model.backward(X_batch, Y_batch)

                w_acc1 += dW1
                w_acc2 += dW2
                b_acc1 += db1
                b_acc2 += db2

            # Update parameters using accumulated gradients
            model.W1 -= (learning_rate / batch_size) * w_acc1
            model.W2 -= (learning_rate / batch_size) * w_acc2
            model.b1 -= (learning_rate / batch_size) * b_acc1
            model.b2 -= (learning_rate / batch_size) * b_acc2

            # Store the cost for plotting
            costs.append(loss_acc / (X.shape[0] / batch_size))

            # Print the cost every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss_acc / (X.shape[0] / batch_size)}")
                # Plot the cost function
                plt.plot(costs)
                plt.xlabel('Epochs')
                plt.ylabel('Cost')
                plt.title('Cost Function Over Epochs')

                plt.savefig("plot" + str(j) + ".png")
                j += 1

            # Calculate accuracy after training
        accuracy = np.mean(model.predict(X) == Y)
        print(f"Accuracy after training: {accuracy * 100:.2f}%")

    #########################################################################
