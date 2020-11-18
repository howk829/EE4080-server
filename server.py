from cv2 import cv2
import numpy as np
import pandas as pd
import pickle
import csv
import tensorflow as tf
import os
import json
import http.server
import socketserver
import cgi
from PIL import Image
import io

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

import time

tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, './checkpoints/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

with open('./lat_long_data/result_json.txt', 'r') as out_data:
        image_data = json.load(out_data)


PORT = 1234

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        print("I got a request!!!")
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            fields = cgi.parse_multipart(self.rfile, pdict)
            imgfile = fields.get('image')[0]
            im = Image.open(io.BytesIO(bytes(imgfile)))
            # im.show()

            inim = np.asarray(im)
            inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
            inim = cv2.resize(inim, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            batch = np.expand_dims(inim, axis=0)
            result = sess.run(net_out, feed_dict={image_batch: batch})

            # find the image that has minimum error vector
            min = np.linalg.norm(np.abs(result - image_data['data'][0]['feature_vector']))
            print(min)
            for data in image_data['data']:
                error_vector = np.linalg.norm(np.abs(result - data['feature_vector']))
                if error_vector <= min:
                    #result_lat = data['lat']
                    #result_long = data['long'] 
                    result_id = data['image_id']
                    min = error_vector
                    print(min)

            
            response = {
                # "result_lat": result_lat,
                # "result_long": result_long,
                "result_image_id": result_id,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

            print(json.dumps(response))
            print(result_id)



with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Stopping httpd...\n')