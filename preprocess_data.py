import os

# Dataset source: https://data.mendeley.com/datasets/rscbjbr9sj/2 

'''
TODO: images are named like CNV-something-something.jpeg -> split on the dash 
and take the first element of the split list as the label 
'''

# Step 1: combine all the photos together so I can rebalance the dataset myself
for folder in ['train', 'val', 'test']:
    for subfolder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']: 
        for file in os.listdir(os.path.join('data', folder, subfolder)):
            os.rename(os.path.join(folder, subfolder, file), os.path.join(subfolder, file))

# Step 2: count number of images for each category 
total = 0
for folder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
    num_items = len(os.listdir(os.path.join('data', folder)))
    print('{folder}: {count}'.format(folder=folder, count=num_items))
    total += num_items
print("Total number of images in dataset: ", total)

# CNV: 37216
# DME: 11422
# DRUSEN: 8620
# NORMAL: 26347
# TOTAL: 83605 

# If we keep roughly 10,000 images in each category; 
# CNV: 13000
# DME: 11422
# DRUSEN: 8620 
# NORMAL: 12000
# TOTAL: 45042 

'''
Train: 
CNV: 12000
DME: 10000
DRUSEN: 7800 
NORMAL: 11000

Dev: 
CNV: 
DME:
DRUSEN:
NORMAL: 

Test:
CNV:
DME:
DRUSEN:
NORMAL: 
'''