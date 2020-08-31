import argparse
from util import *
from sasrec import *

parser = argparse.ArgumentParser()
parser.add_argument('--testset', required=True)
args = parser.parse_args()

testset = data_partition('data/{}.txt'.format(args.testset))

dataset = data_partition('./data/data_trv.txt')
user_train, user_valid, user_test, user_num, item_num, item_count, cum_table = dataset

max_len = 200
input_dim=item_num + 1
embedding_dim=50
feed_forward_units=50
head_num=1
block_num=2
dropout_rate=0.2

_, emb = build_model(max_len=max_len,
                         input_dim=item_num + 1,
                         embedding_dim=50,
                         feed_forward_units=50,
                         head_num=1,
                         block_num=2,
                         dropout_rate=0.2)

with open('.model/model.json', "r") as f:
    sasrec_model = json.load(f)

sasrec_model = model_from_json(sasrec_model,custom_objects={
    'PositionEmbedding' : PositionEmbedding(input_dim=max_len,
                                            output_dim=embedding_dim,
                                            mode=PositionEmbedding.MODE_ADD,
                                            mask_zero=True),
    'MultiHeadAttention' :  MultiHeadAttention(head_num=head_num,
                            activation=None,
                            use_bias=False,
                            history_only=True,
                            trainable=True),
    'FeedForward' : FeedForward(units=feed_forward_units,
                                activation='relu',
                                trainable=True)})
sasrec_model.load_weights(".model/model_weights.h5")

train = tesstset[0][0]
valid = testset[1][0]
rated = set(tesetset[0][0])
item_idx = [dataset[2][0][0]]

seq = np.zeros([max_len], dtype=np.int32)
idx = max_len - 1
seq[idx] = valid[0]
for i in reversed(train):
    seq[idx] = i
    idx -= 1
    if idx == -1:
        break
        

test_emb = emb(np.pad(np.array(item_idx).reshape(-1, 1),((0, 0), (max_len-1, 0)), 'constant'))[:, -1]
seq_emb = (-sasrec_model.predict(seq.reshape(1, max_len)))[:, -1]
predictions = tf.matmul(seq_emb, test_emb, transpose_b=True)
pred = predictions.numpy()

print(pred)