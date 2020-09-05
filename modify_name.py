#coding: utf-8

import os
path = os.getcwd()

mm = 1
####### modify dir name ####
dir_train =os.path.join(path,"TIMIT","train_spkr")
if not os.path.exists(dir_train):
    os.makedirs(dir_train)
    for dr in os.listdir(os.path.join(path, "TIMIT", "TRAIN")):
        if dr == ".DS_Store":
            continue
        for d in os.listdir(os.path.join(path, "TIMIT", "train",dr)):
            if d == ".DS_Store":
                continue
            else:
                if len(str(mm)) == 1:
                    n_d = "SP000" + str(mm)
                    os.rename(os.path.join(path,"TIMIT", "train", dr,d), os.path.join(path,"TIMIT", "train_spkr", n_d))
                    mm += 1
                elif len(str(mm)) == 2:
                    n_d = "SP00" + str(mm)
                    os.rename(os.path.join(path,"TIMIT", "train",dr, d), os.path.join(path,"TIMIT", "train_spkr", n_d))
                    mm += 1
                else:
                    n_d = "SP0" + str(mm)
                    os.rename(os.path.join(path,"TIMIT", "train",dr, d), os.path.join(path,"TIMIT","train_spkr", n_d))
                    mm += 1
print("Train over!")

dir_test =os.path.join(path,"TIMIT","test_spkr")
if not os.path.exists(dir_test):
    os.makedirs(dir_test)
    for dr in os.listdir(os.path.join(path, "TIMIT", "test")):
        for d in os.listdir(os.path.join(path, "TIMIT", "test",dr)):
            n_d = "SP0" + str(mm)
            os.rename(os.path.join(path,"TIMIT","test", dr,d), os.path.join(path,"TIMIT","test_spkr",n_d))
            mm += 1
print("Test over!")


####### modify wav name ######
for root, dirs, files in os.walk(os.path.join(path, "timit","train_spkr")):
    print("="*20)
    print("root: {0}, dir:{1}".format(root,dirs))
    print("SpeakerID " + root[-6:])
    n=0
    for name in files:
        if name == '.DS_Store':
            continue
        if(name.endswith(".PHN")):
            os.remove(os.path.join(root, name))
        elif(name.endswith(".TXT")):
            os.remove(os.path.join(root, name))
        elif(name.endswith(".WRD")):
            os.remove(os.path.join(root, name))
        else:
            new_name = root[-6:] + "W0" + str(n) + ".wav"
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
            n += 1
    print(os.listdir(root))