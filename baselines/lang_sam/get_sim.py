import torch
import h5py
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def load_tensor_from_hdf5(tensor_file, key):
    with h5py.File(tensor_file, 'r') as f:
        tensor = torch.tensor(f[key][:])
    return tensor

def compute_cosine_similarity(tensor_file_1, tensor_file_2, key):
    tensor_1 = load_tensor_from_hdf5(tensor_file_1, key)
    tensor_2 = load_tensor_from_hdf5(tensor_file_2, key)

    tensor_1 = tensor_1[:, :256, :]
    tensor_2 = tensor_2[:, :256, :]

    tensor_1_flat = tensor_1.reshape(tensor_1.size(0) * tensor_1.size(1), -1)
    tensor_2_flat = tensor_2.reshape(tensor_2.size(0) * tensor_2.size(1), -1)

    cosine_sim = torch.nn.functional.cosine_similarity(tensor_1_flat, tensor_2_flat, dim=1)
    
    return cosine_sim.mean().item()

def sam_sweep(sweep_folder, base_rep, keys):
    sweep_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(sweep_folder) for f in fn if 'representation.h5' in f]

    # Extract the numeric part after the first four digits for sorting
    # sweep_files.sort(key=lambda x: int(re.search(r'(\d{4})(\d+)', os.path.basename(x)).group(2)))
    sweep_files.sort(key=lambda x: int(os.path.basename(os.path.dirname(x))))

    similarity_scores = {}
    for sweep_file in sweep_files:
        file_key = os.path.basename(os.path.dirname(sweep_file))
        # numeric_key = re.search(r'(\d{4})(\d+)', file_key).group(2)
        numeric_key = file_key
        similarity_scores[numeric_key] = {}
        for key in keys:
            similarity = compute_cosine_similarity(base_rep, sweep_file, key)
            similarity_scores[numeric_key][key] = similarity

    # Sort the scores based on the numeric key
    sorted_keys = sorted(similarity_scores.keys(), key=lambda x: int(x))

    # Print the sorted summary of similarity scores
    for numeric_key in sorted_keys:
        scores = similarity_scores[numeric_key]
        print(f"\n{numeric_key}:")
        for key, score in scores.items():
            print(f"  {key}: {score:.4f}")

    # Bar plot of similarity scores
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sorted_keys))
    for key in keys:
        y = [similarity_scores[num_key][key] for num_key in sorted_keys]
        ax.bar(x, y, width=0.2, label=key)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, rotation=45, ha='right')
    ax.set_title("Cosine Similarity Scores")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_folder, 'similarity_scores.png'))
    plt.show()

if __name__ == '__main__':
    keys = ['image_representation']
    sweep_folder = '/home/venky/lang-segment-anything/sam_logs/20240403_180740'
    base_rep = '/home/venky/lang-segment-anything/sam_logs/20240403_180740/1115/1115_representation.h5'

    sam_sweep(sweep_folder, base_rep, keys)