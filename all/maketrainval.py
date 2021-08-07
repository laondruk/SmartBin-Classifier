import os
import random
 
trainval_percent = 0.66 # The proportion of training verification in the annotation file    
train_percent = 0.5 # The proportion of training in training verification
xmlfilepath ='./index' # The path of the annotation file, the format is .xml
txtsavepath ='.' # The path where each generated txt is stored
total_xml = os.listdir(xmlfilepath)
print(total_xml)
num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
 
ftrainval = open('./trainval.txt','w') # used to write training verification sequence
ftest = open('./test.txt', 'w')
ftrain = open('./train.txt', 'w')
fval = open('./val.txt', 'w')
 
for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
