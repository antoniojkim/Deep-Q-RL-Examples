
import torch
import numpy as np


class BaseModel(torch.nn.Module):

    def __init__(self, state_dict_path=None, verbose=False):
        super(BaseModel, self).__init__()

    
    def load_weights(self, state_dict_path: str):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()

    def save_weights(self, state_dict_path: str):
        torch.save(self.state_dict(), state_dict_path)


    def forward(self, x):
        return x


    def predict(self, x, device=None):
        if device is None:
            return self.forward(torch.tensor(torch.from_numpy(np.array(x)), dtype=torch.double)).detach().numpy()

        return self.forward(torch.tensor(torch.from_numpy(np.array(x)), dtype=torch.double).to(device)).cpu().detach().numpy()


def argmax(x):
    return np.random.choice(np.flatnonzero(x == x.max()))


class Memory():

    def __init__(self, max_size, buckets=True):
        self.max_size = max_size
        self.buckets = buckets
        self.memory = []

    def __getitem__(self, i):
        return self.memory[i]

    def __len__(self):
        return len(self.memory)

    def learn(self, items):
        if self.buckets:
            self.memory.append(items)
        else:
            self.memory.extend(items)

        if self.max_size is not None:
            excess = len(self.memory) - self.max_size
            if excess > 0: del self.memory[:excess]

    def sample(self, num):
        indices = list(np.random.randint(len(self.memory), size=num))
        if not self.buckets:
            return [self.memory[i] for i in indices]
        else:
            indices = [(i, np.random.randint(len(self.memory[i]))) for i in indices]
            return [self.memory[i][j] for i, j in indices]


