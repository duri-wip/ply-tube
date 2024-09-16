import numpy as np
from torch.utils.data import Dataset

class LSTMdataset(Dataset):
    def __init__(self, playlists):
        self.BOM = {}
        self.playlists = playlists

        for playlist in self.playlists:
            for song in playlist:
                if song not in self.BOM:
                    self.BOM[song] = len(self.BOM)

        self.data = self.generate_sequence(self.playlists)

    def generate_sequence(self, playlists):
        seq = []

        for playlist in playlists:
            ply_bom = [self.BOM[song] for song in playlist]

            data = [ply_bom[i:i+3] for i in range(len(ply_bom)-2)]

            seq.extend(data)
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = np.array(self.data[idx][:2])
        label = np.array(self.data[idx][2], dtype=np.int64)
        return data, label