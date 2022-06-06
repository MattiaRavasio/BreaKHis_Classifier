import torch


def predict_labels(model_output):

    return torch.argmax(model_output, dim=1)


def compute_loss(model, model_output, target_labels, is_normalize = True):

    loss = model.loss_criterion(input = model_output, target=target_labels)
    
    try:
        loss /= model.ouput.shape[0]*is_normalize
    except:
        pass 

    return loss
