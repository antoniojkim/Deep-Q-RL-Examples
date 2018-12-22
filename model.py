
import torch
import numpy as np




class BaseModel(torch.nn.Module):

    def __init__(self, max_size=None, buckets = True, 
                batch_size=1, mini_batch_size = 64, discount=0.95,
                verbose=False, 
                **kwargs):

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



