import pickle


with open("data/fast_pretrained_vectors.pkl","rb") as f1:
	vectors, look_up = pickle.load(f1)


print vectors[0]


# with open("data/NODE_MAP.pkl","rb") as f1:
# 	NODE_MAP = pickle.load(f1)

# print NODE_MAP