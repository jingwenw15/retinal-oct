import numpy as np

def analyze(filename): 
    f = open(filename, 'r')
    total, correct = 0, 0 
    cm = np.zeros((4, 4)) # [pred, actual]

    for line in f: 
        name, output, label = line.strip().split(',')
        output = int(output)
        label = int(label)
        if output == label:
            correct += 1
        total += 1
        cm[output, label] += 1
    cm = cm / np.sum(cm, axis=0)
    print(correct / total)
    print(cm)

analyze('predictions/vgg_dev.csv')