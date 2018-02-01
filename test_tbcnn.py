import os
import logging
import pickle
import tensorflow as tf
import numpy as np
import network as network
import sampling as sampling
from parameters import LEARN_RATE, EPOCHS, CHECKPOINT_EVERY, TEST_BATCH_SIZE, DROP_OUT
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import json
import csv
from django.utils.encoding import smart_str, smart_unicode
import spacy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


en_nlp = spacy.load('en') 


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
            # print child.text
            child_json = {
                "node": smart_str(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
        
  
    return root_json, num_nodes

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




def test_model(logdir, inputs, embeddings_list_url,node_map_url, epochs=EPOCHS):
    """Train a classifier to label ASTs"""


    n_classess = 2
  
 

   
    print("Loading embedding list.....")
    with open(embeddings_list_url, "rb") as embeddings_list_fh:
        embeddings_list = pickle.load(embeddings_list_fh)


    num_feats = len(embeddings_list[0])
    print("number of features : " + str(num_feats))



    print("Loading node map for looking up.....")
    with open(node_map_url, "rb") as node_map_fh:
        # all_1_pairs, all_0_pairs = pickle.load(fh)
        node_map = pickle.load(node_map_fh)

   


    # build the inputs and outputs of the network
    left_nodes_node, left_children_node, left_pooling_node = network.init_net_for_siamese(
        num_feats
    )

    right_nodes_node, right_children_node, right_pooling_node = network.init_net_for_siamese(
        num_feats
    )

    merge_node = tf.concat([left_pooling_node, right_pooling_node], -1)


    hidden_node = network.hidden_layer(merge_node, 600, 300)
    # hidden_node = tf.layers.dropout(hidden_node, rate=0.2, training=False)

    hidden_node = network.hidden_layer(hidden_node, 300, 100)
    # hidden_node = tf.layers.dropout(hidden_node, rate=0.2, training=False)

    hidden_node = network.hidden_layer(hidden_node, 100, n_classess)


    out_node = network.out_layer(hidden_node)

    labels_node, loss_node = network.loss_layer(hidden_node, n_classess)

    optimizer = tf.train.AdamOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)

    # tf.summary.scalar('loss', loss_node)

    ### init the graph
    sess = tf.Session()#config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise 'Checkpoint not found.'

    checkfile = os.path.join(logdir, 'cnn_tree.ckpt')
    steps = 0


    correct_labels = []
    predictions = []
    print('Computing testing accuracy...')


    with open(inputs,"rb",) as csvfile:
    # with codecs.open("data/test.csv", "r", encoding = "utf-8", errors = 'replace') as csvfile:
        test_data_reader = csv.DictReader(csvfile, delimiter=',')
        for row in test_data_reader:

            print("----------------------")
            print(smart_str(row['test_id']))
            print(smart_str(row['question1']))
            print(smart_str(row['question2']))
            try:
                left_tree, right_tree = get_trees(smart_str(row['question1']), smart_str(row['question2']))
                left_nodes, left_children, right_nodes, right_children = sampling.patch_data(left_tree, right_tree, embeddings_list, node_map)
                # for left_nodes, left_children, right_nodes, right_children in sampling.patch_data(left_tree, right_tree, embeddings_list, node_map):

                    # left_nodes, left_children = left_gen_batch

                    # right_nodes, right_children = right_gen_batch
                   
                output = sess.run([out_node],
                    feed_dict={
                        left_nodes_node: left_nodes,
                        left_children_node: left_children,
                        right_nodes_node: right_nodes,
                        right_children_node: right_children
                    }
                )
                print(output)
                predicted = np.argmax(output[0])
                print(predicted)
                with open("data/predict_proba2.csv","a") as f2:
                    f2.write(row['test_id'] + "," + str(format(output[0][0][1],"f")) + "\n")
            except Exception as e:
                print "Error : " + str(e)
                with open("data/predict_proba2.csv","a") as f2:
                    f2.write(row['test_id'] + "," + "0" + "\n")

   
def main():

     # example params : 
        # argv[1] = ./bi-tbcnn/bi-tbcnn/logs/1
        # argv[2] = ./sample_pickle_data/all_training_pairs.pkl
        # argv[3] = ./sample_pickle_data/python_pretrained_vectors.pkl
        # argv[4] = ./sample_pickle_data/fast_pretrained_vectors.pkl
    # test_model(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])
    test_model("logs/model2/","data/test.csv","data/glove_embeddings_list.pkl", "data/NODE_MAP_GLOVE.pkl")


if __name__ == "__main__":
    main()
