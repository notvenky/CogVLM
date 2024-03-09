import torch
import pickle
import h5py

def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        tensor = pickle.load(f)
    return torch.tensor(tensor)

def load_tensor_from_hdf5(tensor_file, key):
    with h5py.File(tensor_file, 'r') as f:
        tensor = torch.tensor(f[key][:])
    return tensor

def compute_cosine_sim_for_key(tensor_file_1, tensor_file_2, key):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, key)
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, key)
    
    # Adjust slicing based on tensor shape
    if tensor_1.size(2) > 268:
        if 'self' in key:
            tensor_1 = tensor_1[:, :, :268, :268]
            tensor_2 = tensor_2[:, :, :268, :268]
        elif 'cross' in key:
            tensor_1 = tensor_1[:, :, :268, :6400]
            tensor_2 = tensor_2[:, :, :268, :6400]
    
    # Reshape tensors for cosine similarity
    tensor_1_flat = tensor_1.reshape(tensor_1.size(0) * tensor_1.size(1), -1)
    tensor_2_flat = tensor_2.reshape(tensor_2.size(0) * tensor_2.size(1), -1)
    
    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1_flat, tensor_2_flat, dim=1)
    
    return cosine_sim.mean()

def flatscore(tensor_file_1, tensor_file_2, key):
    # Load tensors from the HDF5 files
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, key)
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, key)

    # Squeeze to remove any singleton dimension (optional, depending on your data structure)
    tensor_1 = tensor_1.squeeze()
    tensor_2 = tensor_2.squeeze()

    # Flatten the tensors
    tensor_1_flat = tensor_1.view(-1)
    tensor_2_flat = tensor_2.view(-1)

    # Pad the shorter tensor to match the size of the longer one
    if tensor_1_flat.size(0) < tensor_2_flat.size(0):
        tensor_1_flat = torch.nn.functional.pad(tensor_1_flat, (0, tensor_2_flat.size(0) - tensor_1_flat.size(0)))
    else:
        tensor_2_flat = torch.nn.functional.pad(tensor_2_flat, (0, tensor_1_flat.size(0) - tensor_2_flat.size(0)))

    # Compute cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1_flat.unsqueeze(0), tensor_2_flat.unsqueeze(0), dim=1)

    # Return the computed cosine similarity as a scalar value
    return cosine_sim.item()

if __name__ == '__main__':
    keys = ['cross_attn_weights', 'self_attn_weights']
    # keys = ['hidden_states']
    # keys = ['output_embeddings']
    for _ in keys:
        similarity = compute_cosine_sim_for_key('/home/venky/CogVLM/logs/2024-03-07/run_d4_dist_2024-03-07_10-40-55/d4_dist_intermediate_representations.h5', '/home/venky/CogVLM/logs/2024-03-07/run_d1_2024-03-07_10-37-04/d1_intermediate_representations.h5', _)
        print(f"Cosine similarity for {_}:", similarity)
    print("Overall Similarity Score:", similarity)
