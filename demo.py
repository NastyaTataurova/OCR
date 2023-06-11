import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import CapchaDataset
from src.model import CRNN


if __name__ == '__main__':
    weights_path = 'weights/last_weights.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = CapchaDataset((3, 5), samples=1)
    test_loader = DataLoader(dataset=test_dataset)
    img_channel = 1
    img_height, img_width = test_dataset[0][0].shape
    num_class = 11 #CapchaDataset.num_classes
    model = CRNN(img_channel, img_height, img_width, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, targets = data[0].to(device), data[1].to(device)
            pred = model(images)
            pred = torch.nn.functional.log_softmax(pred, dim=2)    
            pred = np.transpose(pred.cpu().numpy(), (1, 0, 2))[0]
            pred = np.argmax(pred, axis=-1)
            prediction = []
            previous = None 
            for l in pred:
                if l != previous:
                    prediction.append(l)
                    previous = l
            prediction = [predict for predict in prediction if predict != 10]
            targets = targets.cpu().numpy()[0]
            target = [t for t in targets if t != 10]
            print(f'Prediction: {prediction}, Real: {target}')
            plt.imshow(images[0].cpu().numpy())
            plt.title(f'Prediction: {str(prediction)}, Real: {str(target)}')
            plt.show()    