import utils, cv2
import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import Config
from model import GANnomaly
from utils import read_images

config = Config()
config.MODE = "train"

model = GANnomaly(config)

t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    train_D = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_D, var_list=model.vars_D)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='encoder')):
    train_G = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_G, global_step=model.global_step, var_list=model.vars_G)


sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

images, labels = read_images(config.PATH_DATA, "folder")
num_iters = len(images) // config.BATCH_SIZE

test_images, test_labels = read_images(config.PATH_TEST, "folder")
test_num_iters = len(test_images) // config.BATCH_SIZE

cnt = 0
length = 6
best_auc = 0

scores_out = []
labels_out = []

with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_checkpoint_name = config.PATH_CHECKPOINT + "/model.ckpt"
    if config.IS_CONTINUE:
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)
       
    for epoch in range(config.EPOCH):
        cnt = 0
        scores_out = []
        labels_out = []
        images, labels = utils.data_shuffle(images, labels)

        for idx in range(num_iters):
            st = time.time()
            image_batch = images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
            label_batch = labels[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]

            for _ in range(1):
                _, loss_D = sess.run([train_D, model.loss_D], feed_dict={model.image:image_batch})
            _, loss_G = sess.run([train_G, model.loss_G], feed_dict={model.image:image_batch})
            _, loss_G, global_step = sess.run([train_G, model.loss_G, model.global_step], feed_dict={model.image:image_batch})

            cnt = cnt + config.BATCH_SIZE
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss_D = %.4f Current_Loss_G = %.4f Time = %.2f"%(epoch, cnt, loss_D, loss_G, time.time()-st)
                utils.LOG(string_print)
                st = time.time()
        
        print("Performing validation")
        for idx in range(test_num_iters):
            st = time.time()
            image_batch = test_images[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]
            label_batch = test_labels[idx*config.BATCH_SIZE:(idx+1)*config.BATCH_SIZE]

            latent_loss, latent_gen_loss = sess.run([model.encoded_input, model.encoded_sample], feed_dict={model.image:image_batch})
            latent_error = np.mean(abs(latent_loss-latent_gen_loss), axis=-1)
            latent_error = np.reshape(latent_error, [-1])
            scores_out = np.append(scores_out, latent_error)
            labels_out = np.append(labels_out, label_batch)

            #out_str = "---------->%d/%d" % (config.BATCH_SIZE*idx, config.BATCH_SIZE*test_num_iters)
            #print(out_str, end='\r')

            scores_out = np.array(scores_out)
            labels_out = np.array(labels_out)
            scores_out = (scores_out - scores_out.min())/(scores_out.max()-scores_out.min())
        auc_out = utils.roc(labels_out, scores_out)
        print("AUC: %.4f BEST AUC: %.4f" %(auc_out, best_auc))

        if auc_out > best_auc:
            best_auc = auc_out
        #if True:
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("best_checkpoints",epoch)):
                os.makedirs("%s/%04d"%("best_checkpoints",epoch))

            print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
            saver.save(sess, "%s/%04d/model.ckpt"%("best_checkpoints",epoch))

            # Save latest checkpoint to same file name
            print('Saving model with %d epochs to disk' % (epoch))
            saver.save(sess, "best_checkpoints/model.ckpt")

        #if epoch%2==0:
        #    results=None
        #    for idx in range(length//2):
        #        X = sess.run(model.sample, feed_dict={model.image:images[length*idx:length*(idx+1)]})
        #        X = (X+1)/2.0
        #        if results is None:
        #            results = (images[length*idx:length*(idx+1)]+1)/2.0
        #            results = np.vstack((results, X))
        #        else:
        #            results = np.vstack((results, (images[length*idx:length*(idx+1)]+1)/2.0))
        #            results = np.vstack((results, X))
        #    utils.save_plot_generated(results, length, "sample_data/" + str(global_step) + "_" + str(epoch) + "_gene_data.png")
        
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

        print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
        saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints",epoch))

        # Save latest checkpoint to same file name
        print('Saving model with %d epochs to disk' % (epoch))
        saver.save(sess, model_checkpoint_name)

    print("Complete train!")