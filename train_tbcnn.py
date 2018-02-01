# This file is just another version to test with 2 different AST tree on each side of the Bi-TBCNN
import os
import logging
import pickle
import tensorflow as tf
import numpy as np
import network as network
import sampling as sampling
from parameters import LEARN_RATE, EPOCHS, CHECKPOINT_EVERY, BATCH_SIZE, DROP_OUT
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import sys
import json
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# device = "/cpu:0"
# device = "/device:GPU:0"
def get_one_hot_similarity_label(batch_labels):
    sim_labels = []
    sim_labels_num = []
    for i in range(0,len(batch_labels)):
        #print left_labels[i] + "," + right_labels[i]
        if batch_labels[i] == "1":
            sim_labels.append([0.0,1.0])
            sim_labels_num.append(1)
        else:
            sim_labels.append([1.0,0.0])
            sim_labels_num.append(0)
    return sim_labels, sim_labels_num


def get_trees_from_pairs(training_pairs):
    

    left_trees = []
    right_trees = []
    labels = []
    for pair in training_pairs:
        left_trees.append(pair[0])
        right_trees.append(pair[1])
        labels.append(pair[2])
    return left_trees, right_trees, labels

def generate_random_batch(iterable,size):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def train_model(logdir, inputs, embeddings_list_url, node_map_url, epochs=EPOCHS, with_drop_out=1,device="0"):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    
    print("Using device : " + device)
    print("Batch size : " + str(BATCH_SIZE))
    if int(with_drop_out) == 1:
        print("Training with drop out rate : " + str(DROP_OUT))
    n_classess = 2
   
    print("Loading training data....")
    # print "Using device : " + device
    with open(inputs, "rb") as fh:
        # all_1_pairs, all_0_pairs = pickle.load(fh)
        all_training_pairs = pickle.load(fh)

    random.shuffle(all_training_pairs)
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
    # with tf.device(device):
    print("Left pooling shape : " + str(tf.shape(left_pooling_node)))
    print("Right pooling shape : " + str(tf.shape(right_pooling_node)))
    merge_node = tf.concat([left_pooling_node, right_pooling_node], -1)
    print(tf.shape(merge_node))
    hidden_node = network.hidden_layer(merge_node, 600, 300)
    if int(with_drop_out) == 1:
        hidden_node = tf.layers.dropout(hidden_node, rate=DROP_OUT, training=True)

    hidden_node = network.hidden_layer(hidden_node, 300, 100)

    if int(with_drop_out) == 1:
        hidden_node = tf.layers.dropout(hidden_node, rate=DROP_OUT, training=True)

    hidden_node = network.hidden_layer(hidden_node, 100, n_classess)

    if int(with_drop_out) == 1:
        hidden_node = tf.layers.dropout(hidden_node, rate=DROP_OUT, training=True)

    out_node = network.out_layer(hidden_node)

    labels_node, loss_node = network.loss_layer(hidden_node, n_classess)

    optimizer = tf.train.AdamOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)

    # tf.summary.scalar('loss', loss_node)

    # correct_prediction = tf.equal(tf.argmax(out_node,1), tf.argmax(labels_node,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### init the graph
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

  
    config = tf.ConfigProto()
    # config.gpu_options.allocator_type ='BFC'
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.98


    sess = tf.Session(config = config)

    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     raise 'Checkpoint not found.'

    checkfile = os.path.join(logdir, 'cnn_tree.ckpt')
    steps = 0   

    using_vector_lookup_left = False
    if os.path.isfile("/input/config.json"):
        file_handler = open(config_file, 'r')
        contents = json.load(file_handler)
        using_vector_lookup_left = contents['using_vector_lookup_left'] == "false"

    print("Begin training....")

    # with tf.device(device):
    for epoch in range(1, epochs+1):
        # sample_1_pairs = random.sample(all_1_pairs,1000)
        # sample_0_pairs = random.sample(all_0_pairs,1000)

        # sample_training_pairs = random.sample(all_training_pairs,6400)

        shuffle_left_trees, shuffle_right_trees, labels = get_trees_from_pairs(all_training_pairs)
        print("Len left:",len(shuffle_left_trees),"Len right:",len(shuffle_right_trees))
        for left_gen_batch, right_gen_batch, labels_batch in sampling.batch_random_samples_2_sides(shuffle_left_trees, shuffle_right_trees, embeddings_list, node_map, labels, BATCH_SIZE):
            print("----------------------------------------------------")
            print("Len of label batch : " + str(labels_batch))
            left_nodes, left_children = left_gen_batch

            right_nodes, right_children = right_gen_batch

            sim_labels, sim_labels_num = get_one_hot_similarity_label(labels_batch)
            # print("sim labels : " + str(sim_labels))

                
            _, err, out, merge, labs = sess.run(
                [train_step, loss_node, out_node, merge_node, labels_node],
                feed_dict={
                    left_nodes_node: left_nodes,
                    left_children_node: left_children,
                    right_nodes_node: right_nodes,
                    right_children_node: right_children,
                    labels_node: sim_labels
                }
            )

            # print "hidden : " + str(loss)
            print('Epoch:', epoch,'Steps:', steps,'Loss:', err, "True Label vs Predicted Label:", zip(labs,out))
         
            # print('Epoch:', epoch,'Steps:', steps,'Loss:', err)
            if steps % CHECKPOINT_EVERY == 0:
                # save state so we can resume later
                saver.save(sess, os.path.join(checkfile), steps)
                print('Checkpoint saved.')

    
            steps+=1
        steps = 0

def main():
        
    # example params : 
        # argv[1] = ./bi-tbcnn/bi-tbcnn/logs/1
        # argv[2] = ./sample_pickle_data/all_training_pairs.pkl
        # argv[3] = ./sample_pickle_data/python_pretrained_vectors.pkl
        # argv[4] = ./sample_pickle_data/fast_pretrained_vectors.pkl
        # argv[5] = 1

    # print "Loading embdded..."
    # vector_list = []
    # with open("data/GoogleNews-vectors.txt") as f:
    # # with open("data/test_vectors.txt") as f:
    #     lines = f.readlines()
    #     index = 0
    #     for line in lines:
    #         if index != 0:

    #             split = line.split(" ")

    #             vector = map(float, split[1:len(split)])
    #             vector_np = np.array(vector)
               
    #             vector_list.append(vector_np)
    #             # print vector_list
    #         index +=1

    # embeddings_list = np.array(vector_list)
    
    # print len(embeddings_list)

    # def train_model(logdir, inputs, embeddings_list_url, node_map_url, epochs=EPOCHS, with_drop_out=1,device="0"):
    # train_model("logs/model2/","data/training_pairs_original.pkl","data/glove_embeddings_list.pkl", "data/NODE_MAP_GLOVE.pkl",EPOCHS, 1, "0")
    train_model(sys.argv[1],sys.argv[2],"data/glove_embeddings_list.pkl", "data/NODE_MAP_GLOVE.pkl",EPOCHS, 1, "0")
    


if __name__ == "__main__":
    main()
