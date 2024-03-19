import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os, time
from model import *
from PIL import Image

in_dir = '/content/data/eval15/low/'
out_dir = '/content/test_results_rdgan/'
rd_dir = '/content/RDGAN_cs131/rd_model/'
fe_dir = '/content/RDGAN_cs131/fe_model/'

def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    # img_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 3])
    # hei = tf.compat.v1.placeholder(tf.int32)
    # wid = tf.compat.v1.placeholder(tf.int32)
    img_holder = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32)
    print("img holder shape: ", img_holder.shape)
    hei = tf.keras.Input(shape=(), dtype=tf.int32)
    wid = tf.keras.Input(shape=(), dtype=tf.int32)
    
    img = tf.expand_dims(img_holder, 0)
    print("img shape: ", img.shape)
    img_v = tf.reduce_max(img, axis=-1, keepdims=True)
    print("img_v shape: ", img.shape)
    img_v = close_op(img_v)
    
    img_i, img_r = rdnet(img_v, img, hei, wid)
    img_i = stretch(img_v, img_i)
    img_crm = CRM(img, img_i)
    
    out = fenet(img_crm, img, img_r, hei, wid)
    out = tf.clip_by_value(out[0], 0, 1)
    
    print('Loading...')
    ckpt = tf.train.latest_checkpoint(rd_dir)
    rd_vars = tl.layers.get_variables_with_name('retinex', printable=False)
    rd_saver = tf.train.Saver(rd_vars)
    rd_saver.restore(sess, ckpt)
    
    ckpt = tf.train.latest_checkpoint(fe_dir)
    fe_vars = tl.layers.get_variables_with_name('fusion', printable=False)
    fe_saver = tf.train.Saver(fe_vars)
    fe_saver.restore(sess, ckpt)
    
    img_files = os.listdir(in_dir)
    img_num = len(img_files)
    img_id = 0
    avg_time = 0
    
    for img_file in img_files:
        img_id += 1
        in_img = Image.open(in_dir+img_file).convert("RGB")
        assert in_img is not None
        w = in_img.size[0]
        h = in_img.size[1]
        in_img = np.array(in_img) / 255
        
        start_time = time.time()
        out_img = sess.run(out, feed_dict={img_holder:in_img, hei:h, wid:w})
        duration = float(time.time() - start_time)
        if img_id > 1:
            avg_time += duration
        
        out_name = img_file.split('.', 1)[0] + '.png'
        out_img = Image.fromarray(np.uint8(out_img * 255))
        out_img.save(out_dir+out_name)
        print('step: %d/%d, time: %.2f sec' % (img_id, img_num, duration))
    
    print('Finish! Avg time: %.2f sec' % (avg_time / (img_num-1)))
    sess.close()

if __name__ == '__main__':
    main()
