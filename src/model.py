import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256):
        super(CRNN, self).__init__()

        self.cnn = self.DCNN(img_channel)
        self.maps2sequence = nn.Linear(512 * (img_height // 16 - 1), map_to_seq_hidden)
        self.lstm_1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.lstm_2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * rnn_hidden, num_class)

    def DCNN(self, img_channel):
        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        # Convolution
        conv_kernels = [3, 3, 3, 3, 3, 3, 2]
        conv_strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]
        # MaxPooling
        pool_kernels = [(2, 2), (2, 2), False, (2, 1), False, (2, 1), False]
        pool_strides = [2, 2, False, (2, 1), False, (2, 1), False]
        # BatchNormalization
        batchnorm = [False, False, False, False, True, True, False]

        cnn = nn.Sequential()
        for i in range(len(conv_kernels)):
            cnn.add_module(
                f'Convolution_{i}',
                nn.Conv2d(channels[i], channels[i+1], conv_kernels[i], conv_strides[i], paddings[i])              
            )
            if batchnorm[i]:
                cnn.add_module(f'BatchNormalization_{i}', nn.BatchNorm2d(channels[i+1]))
            cnn.add_module(f'ReLU_{i}', nn.ReLU(inplace=True))
            if pool_strides[i]:
                cnn.add_module(
                f'MaxPooling_{i}',
                nn.MaxPool2d(kernel_size=pool_kernels[i], stride=pool_strides[i])       
            )
        return cnn
    
    def Transcription_layer(self, ):
        ...

    def forward(self, images):
        # convolutional layers, which extract a feature sequence from the input image
        images = torch.unsqueeze(images, dim=1)
        convolutional_output = self.cnn(images)
        # convolutional feature map
        batch, channel, height, width = convolutional_output.size()
        convolutional_feature_maps = convolutional_output.view(batch, channel * height, width)
        convolutional_feature_maps = convolutional_feature_maps.permute(2, 0, 1)
        # feature sequence
        feature_sequence = self.maps2sequence(convolutional_feature_maps)
        # recurrent layers, which predict a label distribution for each frame
        lstm1, _ = self.lstm_1(feature_sequence)
        lstm_output, _ = self.lstm_2(lstm1)
        output = self.linear(lstm_output)
        return output 
