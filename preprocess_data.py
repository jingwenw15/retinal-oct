import os
import random

# Dataset source: https://data.mendeley.com/datasets/rscbjbr9sj/2 

'''
NOTE: images are named like CNV-something-something.jpeg -> split on the dash 
and take the first element of the split list as the label 
'''

# Step 1: combine all the photos together so I can resplit the dataset myself
# os.mkdir('data/CNV')
# os.mkdir('data/DME')
# os.mkdir('data/DRUSEN')
# os.mkdir('data/NORMAL')
for folder in ['train', 'test']:
    for subfolder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']: 
        for file in os.listdir(os.path.join('data', folder, subfolder)):
            os.rename(os.path.join('data', folder, subfolder, file), os.path.join('data', subfolder, file))

# Step 2: count number of images for each category 
total = 0
for folder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
    num_items = len(os.listdir(os.path.join('data', folder)))
    print('{folder}: {count}'.format(folder=folder, count=num_items))
    total += num_items
print("Total number of images in dataset: ", total)

# Step 3: split into train/dev/test
# os.mkdir('data/train')
# os.mkdir('data/val')
# os.mkdir('data/test')
for folder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
    files = os.listdir(os.path.join('data', folder))
    val_test_selection = random.sample(files, 1000)
    dev_filenames, test_filenames = val_test_selection[:500], val_test_selection[500:]
    for cur_file in dev_filenames:
        os.rename(os.path.join('data', folder, cur_file), os.path.join('data', 'val', cur_file))
    for cur_file in test_filenames:
        os.rename(os.path.join('data', folder, cur_file), os.path.join('data', 'test', cur_file)) 
    remaining_files = [f for f in files if f not in val_test_selection]
    for cur_file in remaining_files:
        os.rename(os.path.join('data', folder, cur_file), os.path.join('data', 'train', cur_file)) 

'''
Train: 79609
Dev: 2000
Test: 2000

Train: 
CNV: 36217
DME: 10423
NORMAL: 25348
DRUSEN: 7621
'''
# for dev and test: take 500 from each category
