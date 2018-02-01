import pickle
with open("data/NODE_MAP_GLOVE.pkl") as f:
	node_map = pickle.load(f)

print node_map