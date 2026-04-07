"""
This file contains a training function for the PyTorchClassifier model.

E6692 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FILE TO COMPLETE THE ASSIGNMENT
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(model, device, training_set, validation_set, epochs, optimizer, model_save_path):
    """
    This function defines the training process for a classification model.
    Refer to this for help and how to use PyTorch optimization functions properly:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    params:
        model (nn.Module): the PyTorch classification model

        device (torch.device): GPU specification

        training_set (torch.utils.data.Dataset): the classification training dataset

        validation_set (torch.utils.data.Dataset): the classification validation dataset

        batch_size (int): the batch size. This determines how many iterations it
                          takes to go through an epoch and how many images are included
                          in the gradient calculation at one time.

        optimizer (torch.optim): the optimization function (i.e. Adam, SGD, RMSprop, etc.)

        epochs (int): the number of passes through the entire dataset

        model_save_path (string): path to save the model with the highest validation
                            accuracy.

    returns:
        train_loss_history (list of floats): the training loss by iteration

        train_accuracy_history (list of floats): the training accuracy by iteration

        val_loss_history (list of floats): the validation loss by iteration

        val_accuracy_history (list of floats): the validation accuracy by iteration

    """
    # wipe GPU cache
    torch.cuda.empty_cache()
    
    # initialize empty lists for storing training and validation loss and accuracy
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    # initialize highest validation accuracy value
    highest_val_accuracy = 0.0
    
    # iterate through epochs
    for epoch in range(epochs):
        # initialize training loss and iteration values
        train_i = 0
        train_sum_loss = 0.0
        train_error_count = 0.0
        print("Epoch: {}".format(epoch + 1))
        
        # make sure model is in train mode (dropout activated, weights not frozen, etc.)
        model.train()
        
        # define iterator
        i = 0
        
        # iterate through training epoch
        for images, labels in iter(training_set):
            
            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            # set the gradients of batch to zero
            optimizer.zero_grad()

            # complete forward pass of batch through model
            outputs = model(images)

            # calculate the loss of the output
            loss = F.cross_entropy(outputs, labels)
            
            # compute the gradients of the pass
            loss.backward()

            # update the model weights
            optimizer.step()
            
            # update the training loss and accuracy histories
            train_error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
            count = len(labels.flatten())
            train_i += count
            train_sum_loss += float(loss)
            train_loss = train_sum_loss / train_i
            train_accuracy = 1.0 - train_error_count / train_i

            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')
                
            i += 1
            
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
            

            
        # initialize validation loss and iteration values
        val_i = 0
        val_sum_loss = 0.0
        val_error_count = 0.0

        # switch model to evaluation mode
        model.eval()
        
        # iterate through validation epoch
        for images, labels in iter(validation_set):

            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            # complete forward pass of batch through model
            outputs = model(images)

            # calculate loss
            loss = F.cross_entropy(outputs, labels)

            # update the validation loss and accuracy histories
            val_error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
            count = len(labels.flatten())
            val_i += count
            val_sum_loss += float(loss)
            val_loss = val_sum_loss / val_i
            val_accuracy = 1.0 - val_error_count / val_i

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        # if validation accuracy is higher than previous highest, save model weights and update highest val accuracy value
        if val_accuracy_history[-1] > highest_val_accuracy:
            print("Saving model.")
            highest_val_accuracy = val_accuracy_history[-1]
            torch.save(model.state_dict(), model_save_path)

        # print losses and accuracies each epoch
        print("Training Loss: {}\t Training Accuracy: {}".format(round(train_loss, 4), round(train_accuracy, 4)))
        print("Validation Loss: {}\t Validation Accuracy: {}".format(round(val_loss, 4), round(val_accuracy, 4)))
        
    # return training and validation loss and accuracy histories
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history
            
    
def plot_loss_and_accuracy(loss, accuracy, title, labels):
    """
    This function plots the loss and accuracy histories on the same plot and
    displays it in the jupyter cell output.

    params:
        loss (list of float): loss history
        accuracy (list of float): accuracy history
        title (string): title of the plot
        labels (list of string): list of plot labels
    """
    _, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    ax1.plot(loss, color='blue')
    ax2.plot(accuracy, color='orange')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend([labels[0]], loc='best')
    ax2.set_ylabel("Accuracy")
    ax2.legend([labels[1]], loc='best')

    plt.title(title)
    plt.show()