import argparse
import tensorflow as tf

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
from google.protobuf import json_format, text_format

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def saved_model_client_barebone(host, port, model_name, signature_name):
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name
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

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--host",
      type=str,
      default="127.0.0.1",
      required=False,
      help="The host of tfs model, default 127.0.0.1."
  )

  parser.add_argument(
      "--port",
      type=str,
      default="9000",
      required=False,
      help="The port of tfs model, default 9000."
  )

  parser.add_argument(
      "--model_name",
      type=str,
      default="saved",
      required=False,
      help="The tfs model name, default `saved`."
  )

  parser.add_argument(
      "--signature_name",
      type=str,
      default="serving_default",
      required=False,
      help="The signature name, default `serving_default`."
  )

  args = parser.parse_args()


  saved_model_client_barebone(args.host, args.port, args.model_name, args.signature_name)