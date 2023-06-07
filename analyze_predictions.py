import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

'''
Analyze prediction csv files on the dev and test sets using secondary metrics. 
Also provide visualizations. 
'''


def analyze(filename): 
    f = open(filename, 'r')
    total, correct = 0, 0 
    conf_matrix = np.zeros((4, 4)) # [actual, pred]

    for line in f: 
        name, output, label = line.strip().split(',')
        output = int(output)
        label = int(label)
        if output == label:
            correct += 1
        total += 1
        conf_matrix[label, output] += 1

    cm_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1 = (2 * precision * recall) / (precision + recall)

    print('Accuracy by class', cm_acc)
    print('Precision', precision, 'Avg', np.sum(precision) / 4)
    print('Recall', recall, 'Avg', np.sum(recall) / 4)
    print('F1 score', f1, 'Avg', np.sum(f1) / 4)
    print('Confusion matrix', conf_matrix)
    print('Accuracy', correct / total)

    sbn.heatmap(conf_matrix, annot=True, fmt='.0f')
    plt.show()
    

analyze('predictions/mobilenet_knowledge_test_s_mb2_t_vgg.csv')
analyze('predictions/vgg_test.csv')
analyze('predictions/custom_test_net_4.csv')