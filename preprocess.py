import csv
import spacy
import random
from random import randint
import pickle
import codecs
from django.utils.encoding import smart_str, smart_unicode

question1_list = []
question2_list = []
is_duplicate = []
pairs_tuple_list = []

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
   
    root_json = {
        "node": smart_str(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
       
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.children]
        queue.extend(children)
        
        for child in children:
       
            #print child.kind

            child_json = {
                "node": smart_str(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
        
  
    return root_json, num_nodes

def remove_any_random_samples_match_question(samples, question):
	
	no_duplicated_samples = []
	for sample in samples:
		if sample != question:
			no_duplicated_samples.append(sample)
	return no_duplicated_samples

def get_trees(question1,question2):
	
	doc1 = en_nlp(unicode(question1,"utf-8"))
	doc2 = en_nlp(unicode(question2,"utf-8"))
	root1 = None
	root2 = None

	for sent1 in doc1.sents:
		root1 = sent1.root

	for sent2 in doc2.sents:
		root2 = sent2.root
	
	tree_1, _ = _traverse_tree(root1)
	tree_2, _ = _traverse_tree(root2)

	return tree_1,tree_2
	

en_nlp = spacy.load('en') 

with open("data/train.csv","rb") as csvfile:
# with codecs.open("data/train.csv", "r", encoding = "utf-8", errors = 'ignore') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=',')
	for row in reader:
		question1_list.append(row['question1'])
		question2_list.append(row['question2'])

with open("data/train.csv","rb") as csvfile:
# with codecs.open("data/train.csv", "r", encoding = "utf-8", errors = 'ignore') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=',')
	error_count = 0
	count = 0
	for row in reader:
		try:
		
			tree_1, tree_2 = get_trees(smart_str(row['question1']), smart_str(row['question2']))
			# print "----------------"
			# print row['question1']
			# print row['question2']
			pair_tuple = (tree_1,tree_2,smart_str(row['is_duplicate']))
			pairs_tuple_list.append(pair_tuple)
			

			# # This part is for more data
			# samples = random.sample(question1_list,1)
			# samples = remove_any_random_samples_match_question(samples,smart_str(row['question1']))
			# # print "samples : " + str(samples)
			# for sample in samples:

			# 	tree1, tree_2 = get_trees(sample,row['question2'])

			# pair_tuple = (tree_1,tree_2,"0")
			# pairs_tuple_list.append(pair_tuple)
		

			# samples = random.sample(question2_list,1)
			# samples = remove_any_random_samples_match_question(samples,smart_str(row['question2']))
			# for sample in samples:
			# 	tree1, tree_2 = get_trees(sample,row['question1'])

			# pair_tuple = (tree_1,tree_2,"0")
			# pairs_tuple_list.append(pair_tuple)
			
			count += 1
		except Exception as e:
			print "Error :" + str(e)
			error_count+=1

		
print "Errors : " + str(error_count)
print "Total : " + str(count)

with open("data/training_pairs_original.pkl","wb") as f1:
	pickle.dump(pairs_tuple_list, f1)


# for sent in doc.sents:
# 	node = sent.root
# 	print node
# 	print node.children
# 		


  