import torch
import pickle

def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        tensor = pickle.load(f)
    return torch.tensor(tensor)

def compute_cosine_similarity(tensor_file_1, tensor_file_2):
    tensor_1 = load_tensor_from_pickle(tensor_file_1).float() 
    tensor_2 = load_tensor_from_pickle(tensor_file_2).float()
    
    tensor_1 = tensor_1.squeeze(0) 
    tensor_2 = tensor_2.squeeze(0)
    
    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1, tensor_2, dim=1)
    return cosine_sim.mean()

if __name__ == '__main__':
    similarity = compute_cosine_similarity('/home/venky/CogVLM/pkl_outputs/2024-02-29/representation_02-46-05.pkl', '/home/venky/CogVLM/pkl_outputs/2024-02-29/representation_02-39-50.pkl')
    print("Cosine similarity:", similarity)
