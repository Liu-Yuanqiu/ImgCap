import os
import json
import redis
import base64, struct, numpy

pool = redis.ConnectionPool(host='210.30.97.224', password="411303qwer", port=6377, db=0)

def redis_save(key, array):
    shape = array.shape
    dim = len(shape)
    value = struct.pack(''.join(['>I']+['I'*dim]), *((dim,)+shape))
    value = base64.a85encode(value+array.tobytes())
    db = redis.Redis(connection_pool=pool)
    db.set(key, value)

def redis_load(key):
    SIZE = 4
    db = redis.Redis(connection_pool=pool)
    bytes = base64.a85decode(db.get(key))
    db.close()
    dim = struct.unpack('>I', bytes[:1*SIZE])[0]
    shape = struct.unpack('>%s' % ('I'*dim), bytes[1*SIZE:(dim+1)*SIZE])
    ret = numpy.frombuffer(
        bytes,
        offset=(dim+1)*SIZE,
        dtype=numpy.float32
    ).reshape(shape)
    return ret

from data.field import TextField
from data.utils import load_txt
def get_stop_words(text_field, stop_word_path):
    words = load_txt(stop_word_path)
    stop_word_ids = []
    for w in words:
        stop_word_ids.append(text_field.vocab.stoi[w])
    return stop_word_ids
path = '../mscoco/'
text_field = TextField(vocab_path=os.path.join(path, "txt", "coco_vocabulary.txt"), vocab_s_path=os.path.join(path, "txt", "se_labels.txt"))
stop_words = get_stop_words(text_field, os.path.join(path, "txt", "english"))
origin_cap = ['swin_dert_grid', 'up_down_36']
for cap in origin_cap:
    cached_train = "cached_coco_train_"+cap+".json"
    cached_val = "cached_coco_val_"+cap+".json"
    cached_val_test = "cached_coco_val_test_"+cap+".json"
    cached_test_test = "cached_coco_test_test_"+cap+".json"

    train_samples = json.load(open(os.path.join(path, cached_train), "r"))
    val_samples = json.load(open(os.path.join(path, cached_val), "r"))
    val_test_samples = json.load(open(os.path.join(path, cached_val_test), "r"))
    test_test_samples = json.load(open(os.path.join(path, cached_test_test), "r"))
    for samples in [train_samples, val_samples, val_test_samples, test_test_samples]:
        for sample in samples:
            id = sample['id']
            token_gt = sample['token_gt']
            token_kd = sample['token_kd']
            max_len = max( max([len(x) for x in token_gt]), len(token_kd) )
            label = numpy.zeros(len(text_field.vocab), dtype=numpy.float32)
            for i in range(max_len):
                for j in range(len(token_gt)):
                    if i >= len(token_gt[j]):
                        pass
                    else:
                        wid = token_gt[j][i]
                        if wid not in [0, 1, 2, 3] and wid not in stop_words:
                            label[wid] += 1
                        else:
                            pass
            redis_save("label/"+str(id), label)
            test_id = id
            print(test_id)
            break

a = redis_load("label/"+str(test_id))
print(a.shape)