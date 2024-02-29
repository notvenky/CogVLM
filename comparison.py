import pickle
file_path1 = '/home/venky/CogVLM/pkl_outputs/2024-02-28/representation_18-23-58.pkl'
file_path2 = '/home/venky/CogVLM/pkl_outputs/2024-02-29/representation_14-38-18.pkl'

with open(file_path1, 'rb') as file1:
    tensor1 = pickle.load(file1)

with open(file_path2, 'rb') as file2:
    tensor2 = pickle.load(file2)
are_identical = (tensor1[:, 400:, :] == tensor2[:, 400:, :]).all()

print(are_identical)