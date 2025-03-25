import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import random_split

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO - define the layers of the network you will use
        self.conv1 = nn.Conv2d(3,32,3, padding = 1)
        self.conv2 = nn.Conv2d(32,64,3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)

        self.fc1 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,100)




    
    def forward(self, x):
        # TODO - define the forward pass of the network you will use
        x = self.pool(F.relu(self.conv1(x))) # this does self conv1 + pool
        x = self.pool(F.relu(self.conv2(x)))  # same, but now conv2

        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        #x = x.view(-1, 128*2*2) #flattening the tensor
        #print("Shape before flatten:", x.shape) # chatGPT suggested to add this line to help debbugging, it did
        x = torch.flatten(x,start_dim = 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        


        return x

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels) 

        loss.backward()
        optimizer.step()
          
        running_loss += loss.item()   ### TODO
        _, predicted = torch.max(outputs, dim = 1)    ### TODO ### AI disclosure, had to check what function to use to find predicted
        #I conceptually understood that the max between our multiclass would be the predicted, jsut not sure.

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs,labels) ### TODO -- loss calculation

            running_loss += loss.item()  ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs, dim = 1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "MyModel",   # Change name when using a different model
        "batch_size": 64, # run batch size finder to find optimal batch size
        "learning_rate": 0.001,
        "epochs": 5,  # Train for longer in a real scenario, seems validation plateus around here (30)
        "num_workers": 8, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################
    # based off of this:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1), ## this and below are to be tested, checking
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Example normalization
    ])

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])   ### TODO -- BEGIN SOLUTION
    # disclaimer - I asked gpt if this is to be similar to our transform_train, which it told me yes

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=None)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))   ### TODO -- Calculate training set size
    val_size = int(0.2 * len(trainset))     ### TODO -- Calculate validation set size

    trainset, valset = random_split(trainset, [train_size,val_size])  ### TODO -- split into training and validation sets
    
    trainset.dataset.transform = transform_train
    valset.dataset.transform = transform_test
    ### TODO -- define loaders and test set --
    trainloader = DataLoader(trainset,batch_size=CONFIG["batch_size"], shuffle = True, num_workers= CONFIG["num_workers"])
    valloader = DataLoader(valset,batch_size=CONFIG["batch_size"], num_workers = CONFIG["num_workers"], shuffle = False)

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size = CONFIG["batch_size"],shuffle = False)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = SimpleCNN()   # instantiate your model ### TODO
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()   ### TODO -- define loss criterion
    optimizer = optim.Adam(model.parameters(), lr = CONFIG["learning_rate"])   ### TODO -- define optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10)  # Add a scheduler   ### TODO -- you can optionally add a LR scheduler


    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()
    # these following lines were suggested by chatgpt to ensure local accuracy is reflected on kaggle
    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))
    model.eval()
    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    #all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    #submission_df_ood = eval_ood.create_ood_df(all_predictions)
    #submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
if __name__ == '__main__':
    main()
