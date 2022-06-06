import torch

def get_optimizer(model, config):
    

    optimizer_type = config["optimizer_type"]
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]

   
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

   
    return optimizer