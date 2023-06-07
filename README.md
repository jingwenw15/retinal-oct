# Retinal OCT Disease Prediction - PyTorch


Citation: Adapted from [CS230-Code-Examples/PyTorch/Vision](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision).


All model training and evaluation is controlled in train.py and train_distill.py.

train.py is for training models from scratch and for transfer learning.

train_distill.py is for knowledge distillation experiments. 

The model folder contains all the models used (including custom net, transfer learning models VGG16 and ResNet18, MobileNetV2). 

preprocess_data.py contains the data preprocessing code to split the dataset. 

build_dataset.py contains the dataloader code as well as data balancing to resolve the data imbalance issue. 

analyze_predictions.py evaluates secondary metrics for the test set predictions of models, and provides visualizations (confusion matrix). 
