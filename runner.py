import os

import matplotlib.pyplot as plt
import torch.utils
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, roc_auc_score

from dl_utils import compute_loss, predict_labels


class Trainer:


    def __init__(
        self,
        data_dir,
        model,
        optimizer,
        model_dir,
        train_data_transforms,
        test_data_transforms,
        batch_size=100,
        load_from_disk=True,
        cuda=False,
    ):
        self.model_dir = model_dir

        self.model = model

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
        self.train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_data_transforms)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.test_dataset = ImageFolder(os.path.join(data_dir, "test"), transform=test_data_transforms)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )
        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []
        self.last_pred_values = []
        self.last_true_values = []

        self.model.train()

   

    def train(self, num_epochs):
        """
        The main train loop
        """
        self.model.train()

        train_loss, train_acc = self.evaluate(split="train")
        val_loss, val_acc = self.evaluate(split="test")

        self.train_loss_history.append(train_loss)
        self.train_accuracy_history.append(train_acc)
        self.validation_loss_history.append(val_loss)
        self.validation_accuracy_history.append(val_acc)

        print(
            "Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}".format(
                0, self.train_loss_history[-1], self.validation_loss_history[-1]
            )
        )

        for epoch_idx in range(num_epochs):
            self.model.train()
            for _, batch in enumerate(self.train_loader):
                if self.cuda:
                    input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
                else:
                    input_data, target_data = Variable(batch[0]), Variable(batch[1])

                output_data = self.model(input_data)
                loss = compute_loss(self.model, output_data, target_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss, train_acc = self.evaluate(split="train")
            val_loss, val_acc = self.evaluate(split="test")

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)

            print(
                "Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}".format(
                    epoch_idx + 1, self.train_loss_history[-1], self.validation_loss_history[-1]
                )
            )

    def evaluate(self, split="test"):
        """
        Get the loss and accuracy on the test/train dataset
        """
        self.model.eval()

        num_examples = 0
        num_correct = 0
        loss = 0

        if split == "test":
            self.last_pred_values = []
            self.last_true_values = []

        for _, batch in enumerate(self.test_loader if split == "test" else self.train_loader):
            if self.cuda:
                input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            else:
                input_data, target_data = Variable(batch[0]), Variable(batch[1])

            output_data = self.model(input_data)

            num_examples += input_data.shape[0]
            loss += float(compute_loss(self.model, output_data, target_data, is_normalize=False))
            predicted_labels = predict_labels(output_data)

            if split == "test":
                self.last_pred_values+=predicted_labels.cpu().tolist() 
                self.last_true_values+=target_data.cpu().tolist() 
                
            num_correct += torch.sum(predicted_labels == target_data).cpu().item()

        self.model.train()

        return loss / float(num_examples), float(num_correct) / float(num_examples), 

    def plot_loss_history(self):
        """
        Plots the loss history
        """
        plt.figure()
        ep = range(len(self.train_loss_history))

        plt.plot(ep, self.train_loss_history, "-b", label="training")
        plt.plot(ep, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()

    def plot_accuracy(self):
        """
        Plots the accuracy history
        """
        plt.figure()
        ep = range(len(self.train_accuracy_history))
        plt.plot(ep, self.train_accuracy_history, "-b", label="training")
        plt.plot(ep, self.validation_accuracy_history, "-r", label="validation")
        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.show()

    def confusion_matrix(self):
        cm = confusion_matrix(self.last_true_values,self.last_pred_values)

        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]

        accuracy = (TP+TN)/np.sum(cm)
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        precision = TP/(TP+FP)
        negative_pred_val = TN/(TN+FN)
        recall = sensitivity
        f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
        
        print(f'0: Benign, 1: Malignant')
        print()
        print(f'Confusion Matrix:')
        print(f'                       predicted ')
        print(f'                      0       1         ')
        print(f'                  ---------------- ')
        print(f'            t  0  |  {TN}  |  {FP}  | ')
        print(f'            r     ---------------- ')
        print(f'            u  1  |  {FN}   |  {TP} | ')
        print(f'            e     ---------------- ')
        print()
        print(f'Accuracy: {accuracy}')
        print(f'Sensitivty: {sensitivity}')
        print(f'Specificity: {specificity}')
        print(f'Precision: {precision}')
        print(f'Negative predictive value:{negative_pred_val}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')
        


