import numpy as np
import matplotlib.pyplot as plt

pytorch_embeddings = np.load('./output-dir/path_embeddings.npy', allow_pickle=True).item()
tensorrt_embeddings = np.load('./output-dir/path_embeddings_tensorrt.npy', allow_pickle=True).item()

cosines = []
bad_count = 0
all_count = 0
tolerance = 10e-3

for i, key in enumerate(pytorch_embeddings.keys()):
    cosine = np.dot(pytorch_embeddings[key], tensorrt_embeddings[key]) / (np.linalg.norm(pytorch_embeddings[key]) * np.linalg.norm(tensorrt_embeddings[key]))
    cosines.append(cosine)
    all_count += 1
    if abs(1 - cosine) > tolerance:
        print('Not equal')
        l1 = np.linalg.norm(pytorch_embeddings[key] - tensorrt_embeddings[key], ord=1)
        l2 = np.linalg.norm(pytorch_embeddings[key] - tensorrt_embeddings[key], ord=2)
        print(f'L1: {l1}')
        print(f'L2: {l2}')
        print(f'Cosine: {cosine}')
        torch_emb = pytorch_embeddings[key]
        tensorrt_emb = tensorrt_embeddings[key]
        for i in range(tensorrt_emb.shape[0]):
            print(tensorrt_emb[i], torch_emb[i])
        print("Key: ", key)
        print("Index: ", i)
        bad_count += 1

print("Bad embeddings exceeding tolerance:", bad_count, "making up", bad_count / all_count * 100, "%")

n, bins, patches = plt.hist(x=cosines, bins='auto')
plt.show()