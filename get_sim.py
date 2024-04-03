import torch
import pickle
import h5py
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def compute_cosine_sim_for_sam(tensor_file_1, tensor_file_2, key):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, key)
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, key)
    
    # import ipdb; ipdb.set_trace()
    tensor_1 = tensor_1[:, :256, :, :]
    tensor_2 = tensor_2[:, :256, :, :]

    tensor_1_flat = tensor_1.reshape(tensor_1.size(0) * tensor_1.size(1), -1)
    tensor_2_flat = tensor_2.reshape(tensor_2.size(0) * tensor_2.size(1), -1)

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1_flat, tensor_2_flat, dim=1)

    return cosine_sim.mean()

def cosine_sim_embeddings(tensor_file_1, tensor_file_2):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, 'embeddings')
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, 'embeddings')

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1, tensor_2, dim=1)
    return cosine_sim

def sweep_log_summary(sweep_folder, base_rep, keys):
    # sweep_files = os.listdir(sweep_folder)
    # sweep_files = [file for file in sweep_files if os.path.isdir(os.path.join(sweep_folder, file))]
    # sweep_files = [os.path.join(sweep_folder, file, 'hidden_states.h5') for file in sweep_files]

    is_video = True if 'vid' in sweep_folder else False

    sweep_files = os.listdir(sweep_folder)
    sweep_files = [os.path.join(sweep_folder, file, 'hidden_states.h5') for file in sweep_files if os.path.isdir(os.path.join(sweep_folder, file))]

    if is_video:
        sweep_files.sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
    else:
        sweep_files.sort(key=lambda x: int(re.search(r'IMG_(\d+)(?=\.(jpg|png|jpeg|heic))', x, re.IGNORECASE).group(1)))


    # Store the similarity scores
    similarity_scores = {}

    for sweep_file in sweep_files:
        sweep_file_path = os.path.join(sweep_folder, sweep_file)
        for key in keys:
            if key in ['cross_attn_weights', 'self_attn_weights']:
                similarity = compute_cosine_sim_for_key(sweep_file_path, base_rep, key)
            elif key == 'hidden_states':
                similarity = compute_cosine_sim_for_hidden_states(sweep_file_path, base_rep, key)
            elif key == 'embeddings':
                similarity = cosine_sim_embeddings(sweep_file_path, base_rep)
            else:
                raise ValueError(f"Unknown key: {key}")

            # Store the similarity score
            if sweep_file not in similarity_scores:
                similarity_scores[sweep_file] = {}
            similarity_scores[sweep_file][key] = similarity.item()  # Convert from tensor to float

    # Print the summary of similarity scores
    for file_name, scores in similarity_scores.items():
        print(f"\n{file_name}:")
        for key, score in scores.items():
            print(f"  {key}: {score:.4f}")

    # Bar plot of similarity scores
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, key in enumerate(keys):
        x = np.arange(len(sweep_files)) + idx * 0.2
        y = [similarity_scores[file][key] for file in sweep_files]
        ax.bar(x, y, width=0.2, label=key)
    
    ax.set_xticks(np.arange(len(sweep_files)) + 0.2 * (len(keys) - 1) / 2)
    # ax.set_xticklabels(sweep_files, rotation=45, ha='right')
    ax.set_xticklabels([re.search(r'frame_(\d+)', file).group(1) for file in sweep_files], rotation=45, ha='right')
    # scale y limits view to see better between min and max 
    # ax.set_ylim(0.4, 1.0)
    ax.set_title("Cosine Similarity Scores")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_folder, 'similarity_scores.png'))
    plt.show()

# if __name__ == '__main__':
#     # keys = ['cross_attn_weights', 'self_attn_weights']
#     # keys = ['hidden_states']
#     keys = ['image_representation']
#     # keys = ['output_embeddings']

#     sweep_folder = ''
#     base_rep = ''

#     for _ in keys:
#         # similarity_wts = compute_cosine_sim_for_key('/home/venky/CogVLM/logs/2024-03-07/run_d4_dist_2024-03-07_10-40-55/d4_dist_intermediate_representations.h5', '/home/venky/CogVLM/logs/2024-03-07/run_d1_2024-03-07_10-37-04/d1_intermediate_representations.h5', _)
#         similarity = compute_cosine_sim_for_sam('/home/venky/lang-segment-anything/image_representation_20240403142253.h5', '/home/venky/lang-segment-anything/image_representation_20240403140903.h5', _)
#         # similarity = cosine_sim_embeddings('/home/venky/CogVLM/logs/logs_vid_1129/2024-04-01/N33_1129_frame_60.jpg_2024-04-01_16-05-48/hidden_states.h5', '/home/venky/CogVLM/logs/logs_vid_1129/2024-04-01/N33_1129_frame_30.jpg_2024-04-01_16-03-11/hidden_states.h5')
#         print(f"Cosine similarity for {_}:", similarity)
#     print("Overall Similarity Score:", similarity)

if __name__ == '__main__':
    keys = ['hidden_states']
    sweep_folder = '/home/venky/CogVLM/logs/logs_vid_1129/2024-04-01'
    base_rep = '/home/venky/CogVLM/logs/logs_vid_1129/2024-04-01/N33_1129_frame_30.jpg_2024-04-01_16-03-11/hidden_states.h5'

    sweep_log_summary(sweep_folder, base_rep, keys)

