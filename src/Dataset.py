import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class LSTMdataset(Dataset):
    def __init__(self, playlists):
        self.BOM = {}
        self.playlists = playlists

        for playlist in self.playlists:
            for song in playlist:
                if song not in self.BOM.keys():
                    self.BOM[song] = len(self.BOM.keys())

        self.data = self.generate_sequence(self.playlists)

    def generate_sequence(self, playlists):
        seq = []

        for playlist in playlists:
            ply_bom = [self.BOM[song] for song in playlist]

            data = [([ply_bom[i], ply_bom[i+1], ply_bom[i+2]]) for i in range(len(ply_bom)-2)]

            seq.extend(data)
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = np.array(self.data[idx][0])
        label = np.array(self.data[idx][1], dtype=np.int64)
        return data, label

    
def build_dataloaders(dataset, batch_size = 64):
    #dataset 생성
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # random_split을 통해 train과 test 셋 나누기
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader로 묶기 (필요 시 batch_size 설정)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    #dataloader 생성

    print('Train loader loaded! \n size : {len(train_loader.dataset)} \n Test loader loaded! \n size : {len(test_loader.dataset)}')
    print(len(train_dataset.dataset.BOM))
    return train_dataset, test_dataset, train_loader, test_loader
