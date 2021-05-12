import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from utils import save_plot
from data import give_iters
import matplotlib.pyplot as plt
import seaborn as sns


def do_train(model, train_iter, DEVICE, criterion, optimizer, all_losses, all_acc):
    epoch_losses = []
    epoch_acc = []
    for batch_idx, (X, y) in enumerate(train_iter):
        model.train()
        # print(X.size())
        # print(y.size())
        data = X.to(device=DEVICE)
        targets = (y - 1).to(device=DEVICE)
        # print(data.size())
        # print(targets.size())
        # assert 1 == 0
        data = torch.transpose(data, 0, 1)
        # print("hey ",data.size())
        scores = model(data)
        # print(scores.size())
        _, acc1 = torch.max(scores, axis=1)
        correct_pred = (acc1 == targets).float().cpu()

        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_acc.append(np.mean(correct_pred.tolist()))
        print("    Loss: " + str(np.mean(epoch_losses))[:6] + "       Acc: " + str(np.mean(epoch_acc) * 100)[:6] + '%')

    mean_acc = np.mean(epoch_acc) * 100
    mean_loss = np.mean(epoch_losses)
    all_losses.extend(epoch_losses)
    all_acc.append(mean_acc)
    print("Loss: " + str(mean_loss)[:6] + "       Acc: " + str(mean_acc)[:6] + '%')
    return model


def do_val(model, val_iter, DEVICE, criterion, val_all_losses, val_all_acc):
    val_epoch_losses = []
    val_epoch_acc = []
    for batch_idx, (X, y) in enumerate(val_iter):
        model.eval()
        with torch.no_grad():
            data = X.to(device=DEVICE)
            targets = (y - 1).to(device=DEVICE)
            data = torch.transpose(data, 0, 1)
            # print("val data: ", data.size())
            scores = model(data)
            # print("scores data: ", scores.size())
            # print("targets data: ", targets.size())
            _, acc1 = torch.max(scores, axis=1)
            correct_pred = (acc1 == targets).float().cpu()

            val_loss = criterion(scores, targets)

            val_epoch_losses.append(val_loss.item())
            val_epoch_acc.append(np.mean(correct_pred.tolist()))

    val_mean_acc = np.mean(val_epoch_acc) * 100
    val_mean_loss = np.mean(val_epoch_losses)
    val_all_losses.extend(val_epoch_losses)
    val_all_acc.append(val_mean_acc)
    print("Val Loss: " + str(val_mean_loss)[:6] + "   Val Acc: " + str(val_mean_acc)[:6] + '%')


def start_train_val(model, DEVICE, num_epochs, criterion, optimizer, train_iter, val_iter, nf):

    all_losses = []
    all_acc = []
    val_all_losses = []
    val_all_acc = []

    for epoch in range(num_epochs):
        print('Epoch (' + str(epoch + 1) + ' / ' + str(num_epochs) + '):')

        model = do_train(model, train_iter, DEVICE, criterion, optimizer, all_losses, all_acc)
        do_val(model, val_iter, DEVICE, criterion, val_all_losses, val_all_acc)

    save_plot(all_losses, nf + 'losses.png')
    save_plot(all_acc, nf + 'acc.png')
    save_plot(val_all_losses, nf + 'val_losses.png')
    save_plot(val_all_acc, nf + 'val_acc.png')
    return model


def give_class_report(model, DEVICE, val_iter, nf):
    # val_iter = give_iters(val_set.__len__(), DEVICE, [val_set])[0]
    all_val = []
    all_y = []
    for batch_idx, (X, y) in enumerate(val_iter):
        model.eval()
        data = X.to(device=DEVICE)
        data = torch.transpose(data, 0, 1)
        val = model(data).tolist()
        all_val.extend(np.argmax(val, axis=1))
        all_y.extend((y - 1).tolist())
    report = classification_report(all_y, all_val, output_dict=True)
    cf_matrix = confusion_matrix(all_y, all_val)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in range(5)],
                         columns=[i for i in range(5)])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.savefig(nf + 'confusion_matrix.png')
    print('')
    print('')
    print(report)
    print('')
    print('')
    df = pd.DataFrame(report).transpose()
    df.to_csv(nf + 'classification_report.csv', index=True)
