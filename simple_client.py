import argparse
import tensorflow as tf

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
from google.protobuf import json_format, text_format

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from absl import flags
from absl import app
flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
flags.DEFINE_string('data','', 'Word')
flags.DEFINE_string('model_name','saved', 'Word')
flags.DEFINE_string('signature_name','serving_default', 'Word')
FLAGS = flags.FLAGS



def main(_):

  channel = implementations.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name
  request.inputs['v1'].CopyFrom(
    tf.contrib.util.make_tensor_proto(
      [2.0, 2.0, 2.0], shape=[3]
    )
  )
  request.inputs['v2'].CopyFrom(
    tf.contrib.util.make_tensor_proto(
      [3.0, 5.0, 8.0], shape=[3]
    )
  )
  print('request:', request)

  result = stub.Predict(request, 60.0)  # 60 secs timeout
  print('result:', result)

if __name__ == '__main__':
  app.run(main)