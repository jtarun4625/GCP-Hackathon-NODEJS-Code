# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 21:02:07 2018

@author: jtaru
"""




#This will print 60 which is calculated 


from flask import Flask
import tensorflow as tf
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    temp = request.args.get('temp')
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('model.ckpt/my_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model.ckpt/'))
    
    
    # Access saved Variables directly
    # print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved
    
    
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    feed_dict ={w1:temp}
    
    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
    test = str(sess.run(op_to_restore,feed_dict))
    response = jsonify({'state': test})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8081)

