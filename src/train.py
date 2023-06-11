import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN
# from ctc_decoder import ctc_decode


def train_model(model, num_epochs, train_loader, valid_loader, optimizer, criterion, device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_best_loss = 0.
        train_loss = 0.
        train_count = 0
        for train_data in train_loader:
            model.train()
            images, targets = train_data[0].to(device), train_data[1].to(device)
            target_lengths = train_data[2].to(device)
            pred = model(images)
            batch_size = images.size(0)
            input_lengths = torch.LongTensor([pred.size(0)] * batch_size)
            pred = torch.nn.functional.log_softmax(pred, dim=2)

            
            target_lengths = torch.flatten(target_lengths)
            loss = criterion(pred, targets, input_lengths, target_lengths)
            # logits = model(images)
            # log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            # batch_size = images.size(0)
            # input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            # loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_size = train_data[0].size(0)
            train_loss += loss
            train_count += train_size
        train_cur_loss = train_loss / train_count
        print(f'epoch {epoch} - train loss: {train_cur_loss}')
        if train_cur_loss > train_best_loss:
            train_best_loss = train_cur_loss
            save_model_path = f'weights/best_weights_10_1.pt'
            torch.save(model.state_dict(), save_model_path)

        if epoch % 10 == 0:
            model.eval()
            eval_count = 0
            eval_loss = 0
            eval_correct = 0

            with torch.no_grad():
                for data in valid_loader:
                    images, targets = data[0].to(device), data[1].to(device)
                    target_lengths = data[2].to(device)
                    preds = model(images)
                    batch_size = images.size(0)
                    input_lengths = torch.LongTensor([preds.size(0)] * batch_size)
                    preds = torch.nn.functional.log_softmax(preds, dim=2).cpu()       
                    
                    
                    loss = criterion(preds, targets, input_lengths, target_lengths)
                    eval_count += batch_size
                    eval_loss += loss.item()
                    
                    targets = targets.cpu().numpy().tolist()
                    target_lengths = target_lengths.cpu().numpy().tolist()
                    preds = np.transpose(preds.cpu().numpy(), (1, 0, 2))
                    for pred, target, target_length in zip(preds, targets, target_lengths):
                        pred = np.argmax(pred, axis=-1)
                        prediction = []
                        previous = None
                        for l in pred:
                            if l != previous:
                                prediction.append(l)
                                previous = l
                        prediction = [predict for predict in prediction if predict != 10]
                        target = [t for t in target if t != 10]
                        target_length = target_length[0]
                        print(prediction, target)
                        if len(prediction) == len(target):
                            if (np.array(prediction) == np.array(target)).all():
                                eval_correct += 1

            loss = eval_loss / eval_count
            accuracy = eval_correct / eval_count
            print(f'epoch {epoch} - valid: loss={loss}, accuracy={accuracy}')
    save_model_path = f'weights/last_weights.pt'
    torch.save(model.state_dict(), save_model_path)

if __name__ == '__main__':
    num_epochs = 200
    train_batch_size = 16
    eval_batch_size = 32
    lr = 0.0005

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device - {device}')

    train_dataset = CapchaDataset((3, 5), samples=500)
    valid_dataset = CapchaDataset((3, 5), samples=100)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=eval_batch_size)
    img_height, img_width = train_dataset[0][0].shape
    img_channel = 1

    num_class = 11 # CapchaDataset.num_classes()
    model = CRNN(img_channel, img_height, img_width, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256)
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # criterion = CTCLoss(reduction='sum', zero_infinity=True, blank=10)
    criterion = CTCLoss(blank=10)
    criterion.to(device)
    train_model(model, num_epochs, train_loader, valid_loader, optimizer, criterion, device)