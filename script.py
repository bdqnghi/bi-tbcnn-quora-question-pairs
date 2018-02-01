import pickle
import numpy as np
word_list = []
vector_list = []

# with open("data/GoogleNews-vectors.txt") as f:
with open("data/glove.840B.300d.txt") as f:
	lines = f.readlines()

	for line in lines:

		split = line.split(" ")
		# print split[0]
		word_list.append(split[0])

		vector_np = np.array(split[1:len(split)])
		vector_list.append(vector_np)

embeddings_list = np.array(vector_list)

print "Dumping np array of embeddings ......"
with open("data/glove_embeddings_list.pkl","wb") as f1:
	pickle.dump(embeddings_list, f1)


# word_list_np = np.array(word_list)


NODE_MAP = {x: i for (i, x) in enumerate(word_list)}

with open("data/NODE_MAP_GLOVE.pkl","wb") as f1:
	pickle.dump(NODE_MAP, f1)