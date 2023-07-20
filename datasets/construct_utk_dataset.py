import shutil
import os
import numpy as np
root = '../../datasets/gan_UTKFaceHQ/'

dataset = 'UTKFaceHQ'

young = []
old = []
left = []
middle = []
right = []
for fname in sorted(os.listdir(root)):
    age = int(fname.split('_')[0])
    if age<20 and age>10:
        left.append(os.path.join(root, fname))
    if age>20 and age<25:
        young.append(os.path.join(root, fname))
    if age>25 and age<55:
        middle.append(os.path.join(root, fname))
    if age>55 and age<70:
        old.append(os.path.join(root, fname))
    if age>70 and age<80:
        right.append(os.path.join(root, fname))


young = np.array(young)
old = np.array(old)



rand_index = np.random.permutation(len(young))
trainA = young[rand_index][:-500]
testA = young[rand_index][-500:]

rand_index = np.random.permutation(len(old))
trainB = old[rand_index][:-500]
testB = old[rand_index][-500:]


trainA_path = os.path.join(dataset, 'trainA')
trainB_path = os.path.join(dataset, 'trainB')
testA_path = os.path.join(dataset, 'testA')
testB_path = os.path.join(dataset, 'testB')

left_path = os.path.join(dataset, 'left')
right_path = os.path.join(dataset, 'right')
middle_path = os.path.join(dataset, 'middle')

for p in [trainA_path, trainB_path, testA_path, testB_path, left_path, middle_path, right_path]:
    os.makedirs(p, exist_ok=True)

for p in trainA:
    pname = p.split('/')[-1]
    final_name = os.path.join(trainA_path, pname)
    shutil.copy(p, final_name)
for p in trainB:
    pname = p.split('/')[-1]
    final_name = os.path.join(trainB_path, pname)
    shutil.copy(p, final_name)
for p in testA:
    pname = p.split('/')[-1]
    final_name = os.path.join(testA_path, pname)
    shutil.copy(p, final_name)
for p in testB:
    pname = p.split('/')[-1]
    final_name = os.path.join(testB_path, pname)
    shutil.copy(p, final_name)

for p in left:
    pname = p.split('/')[-1]
    final_name = os.path.join(left_path, pname)
    shutil.copy(p, final_name)
for p in middle:
    pname = p.split('/')[-1]
    final_name = os.path.join(middle_path, pname)
    shutil.copy(p, final_name)
for p in right:
    pname = p.split('/')[-1]
    final_name = os.path.join(right_path, pname)
    shutil.copy(p, final_name)








