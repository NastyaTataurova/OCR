# OCR
 The project is made as part of a deep learning course at ITMO University. The goal was to write and train a recognition model to recognize numbers in a picture. Architecture of the model from the [paper](https://arxiv.org/abs/1507.05717). 
 
## Example
 
 ![image_2023-06-11_12-35-51](https://github.com/NastyaTataurova/OCR/assets/49210968/e27620c0-bcb9-497c-b8d1-c387fdb21328)

## Dataset
Images are generated from random numbers from MNIST dataset.

## Metrics
As a quality metric, we chose accuracy to see what is the proportion of correct forecasts.

## Inference
To start you need to clone the repository and run demo.py
```console
conda create --name ocr python==3.9
pip install -r requirements.txt
python demo.py
```
