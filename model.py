
import torch
import numpy as np


class Memory():

    def __init__(self, max_size, buckets=True):
        self.max_size = max_size
        self.buckets = buckets
        self.memory = []

    def __getitem__(self, i):
        return self.memory[i]

    def __len__(self):
        return len(self.memory)

    def remember(self, items):
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


class BaseModel(torch.nn.Module):

    def __init__(self, max_size=None, buckets = True, 
                batch_size=1, mini_batch_size = 64, discount=0.95,
                verbose=False, 
                **kwargs):

        super(BaseModel, self).__init__()

        self.memory = Memory(max_size=max_size, buckets=buckets)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.discount = discount

        

    
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

    def experience_replay(self, optimizer, criterion, device):
        if len(self.memory) > 0:
            batches = []
            for _ in range(self.batch_size): # np.random.randint(1, 2)
                sample = self.memory.sample(self.mini_batch_size)

                values = self.predict([s[0] for s in sample], device)
                predictions = self.predict([s[4] for s in sample], device)

                batch_input = []
                batch_label = []
                for s, value, prediction in zip(sample, values, predictions):
                    state, action, reward, terminal, next_state = s
                    batch_input.append(state)

                    label = value
                    label[action] = reward+(self.discount*np.amax(prediction) if not terminal else 0)
                    batch_label.append(label)

                batches.append((np.array(batch_input), np.array(batch_label)))
                
                if len(batches) > 0:
                    running_loss = 0
                    for batch_input, batch_label in batches:
                        input_tensor = torch.from_numpy(batch_input).double().to(device)
                        label_tensor = torch.from_numpy(batch_label).double().to(device)
                
                        optimizer.zero_grad()
                        outputs = self.forward(input_tensor)
                        loss = criterion(outputs, label_tensor)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                        if running_loss > 1000000000:
                            print(outputs.cpu().detach().numpy())
                            print([[float(x[0]), float(x[1])] for x in list(batch_label)])
                            exit(1)

                    return running_loss / len(batches)


def argmax(x):
    return np.random.choice(np.flatnonzero(x == x.max()))

