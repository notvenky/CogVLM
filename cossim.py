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

def compute_cosine_sim_for_hidden_states(tensor_file_1, tensor_file_2, key):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, key)
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, key)
    
    tensor_1 = tensor_1[:, 258:, :]
    tensor_2 = tensor_2[:, 258:, :]
    # import ipdb; ipdb.set_trace()

    tensor_1_flat = tensor_1.reshape(tensor_1.size(0) * tensor_1.size(1), -1)
    tensor_2_flat = tensor_2.reshape(tensor_2.size(0) * tensor_2.size(1), -1)

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1_flat, tensor_2_flat, dim=1)

    return cosine_sim.mean()

def cosine_sim_embeddings(tensor_file_1, tensor_file_2):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, 'embeddings')
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, 'embeddings')

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1, tensor_2, dim=1)
    return cosine_sim

if __name__ == '__main__':
    # keys = ['cross_attn_weights', 'self_attn_weights']
    keys = ['hidden_states']
    # keys = ['output_embeddings']
    for _ in keys:
        # similarity = compute_cosine_sim_for_key('/home/venky/CogVLM/logs/2024-03-07/run_d4_dist_2024-03-07_10-40-55/d4_dist_intermediate_representations.h5', '/home/venky/CogVLM/logs/2024-03-07/run_d1_2024-03-07_10-37-04/d1_intermediate_representations.h5', _)
        # similarity = compute_cosine_sim_for_hidden_states('/home/venky/CogVLM/logs/2024-03-13/N8_p3_d5_2024-03-13_14-50-05/d5_intermediate_representations.h5', '/home/venky/CogVLM/logs/2024-03-13/N8_p3_d1_2024-03-13_14-48-56/d1_intermediate_representations.h5', _)
        similarity = cosine_sim_embeddings('/home/venky/CogVLM/baselines/logs/imagenet_d3_2024-03-13_16-09-26/embeddings.h5', '/home/venky/CogVLM/baselines/logs/imagenet_d1_2024-03-13_16-08-59/embeddings.h5')
        print(f"Cosine similarity for {_}:", similarity)
    print("Overall Similarity Score:", similarity)
