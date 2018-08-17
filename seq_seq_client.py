from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.
import grpc
import numpy
import tensorflow as tf
from absl import flags
from absl import app
import tensorflow.contrib
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
flags.DEFINE_string('data','', 'Word')
FLAGS = flags.FLAGS

import data
import data_utils

# load data from pickle and npy files
metadata= data.load_metadata(PATH='datasets/')


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    encode_test_input = data_utils.encode([f.read()],w2idx)
    request = predict_pb2.PredictRequest()
    request.inputs.CopyFrom(tf.contrib.util.make_tensor_proto(encode_test_input))
    output = stub.Predict(request, 10.0)  # 10 secs timeout
    replies = []
    for ii, oi in zip(encode_test_input, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        if decoded.count('unk') == 0:
            #if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)


if __name__ == '__main__':
  app.run()