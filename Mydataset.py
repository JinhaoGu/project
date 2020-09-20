import torch.utils.data as Data
from tqdm import tqdm
class MyDataset(Data.Dataset):
        def __init__(self,filepath):
            number = 0
            with open(filepath,"rb") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f,desc="load training dataset"):
                number+=1
        self.number = number
        self.fopen = open(filepath,'rb')
        def __len__(self):
            return self.number
        def __getitem__(self,index):
            x,y = self.fopen.__next__()# 自定义transform()对训练数据进行预处理
            data = list(zip(x,y))#transform(line)
            return data