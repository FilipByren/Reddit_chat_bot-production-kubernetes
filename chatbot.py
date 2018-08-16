
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
from datasets import data
import data_utils

print("Process data")
data.process_data(PATH='datasets/')
# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16 # 5
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3,
                               epochs=10
                               )


# In[8]:
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 256)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)




# In[9]:
#sess = model.restore_last_session()
print("Train model")
sess = model.train(train_batch_gen, val_batch_gen)
print("Export model")
model.export_model()

# export as pb
w2idx = metadata['w2idx']
test_data = ["hi what is your name","lool","do you know me?","daina white"]
encode_test_input = data_utils.encode(test_data,w2idx)
output = model.predict(sess, encode_test_input.T)
replies = []
for ii, oi in zip(encode_test_input, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded.count('unk') == 0:
        #if decoded not in replies:
        print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
        replies.append(decoded)