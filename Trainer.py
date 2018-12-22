
import torch
import numpy as np

class Memory():

    def __init__(self, max_memory_size, buckets=True, **kwargs):
        self.max_memory_size = max_memory_size
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

        if self.max_memory_size is not None:
            excess = len(self.memory) - self.max_memory_size
            if excess > 0: del self.memory[:excess]

    def sample(self, num):
        indices = list(np.random.randint(len(self.memory), size=num))
        if not self.buckets:
            return [self.memory[i] for i in indices]
        else:
            indices = [(i, np.random.randint(len(self.memory[i]))) for i in indices]
            return [self.memory[i][j] for i, j in indices]


class Trainer:

    def __init__(self, model, device, **kwargs):
        self.model = model
        self.device = device

        self.memory = Memory(**kwargs)
        self.init_trainer(**kwargs)
        self.init_batch(**kwargs)

    
    def init_trainer(self, learning_rate, momentum=0, weight_decay=0, loss="MSELoss", **kwargs):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.criterion = getattr(torch.nn, loss)()


    def init_batch(self, batch_size, mini_batch_size, discount=0.95, **kwargs):
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.discount = discount

    
    def experience_replay(self):
        if len(self.memory) > 0:
            batches = []
            for _ in range(self.batch_size): # np.random.randint(1, 2)
                sample = self.memory.sample(self.mini_batch_size)

                values = self.model.predict([s[0] for s in sample], self.device)
                predictions = self.model.predict([s[4] for s in sample], self.device)

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
                        input_tensor = torch.from_numpy(batch_input).double().to(self.device)
                        label_tensor = torch.from_numpy(batch_label).double().to(self.device)
                
                        self.optimizer.zero_grad()
                        outputs = self.model.forward(input_tensor)
                        loss = self.criterion(outputs, label_tensor)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()

                        if running_loss > 1000000000:
                            print(outputs.cpu().detach().numpy())
                            print([[float(x[0]), float(x[1])] for x in list(batch_label)])
                            exit(1)

                    return running_loss / len(batches)



def argmax(x):
    return np.random.choice(np.flatnonzero(x == x.max()))