import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer=tf.constant_initializer([1.0, 2.0, 3.0]))
v2 = tf.get_variable("v2", shape=[3], initializer=tf.constant_initializer([2.0, 3.0, 5.0]))

v3 = v1 + v2

v4 = 2 * v1 + 3 * v2

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

export_dir = './saved/2'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

with tf.Session() as sess:
  sess.run(init_op)
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
  print("v3 : %s" % v3.eval())
  print("v4 : %s" % v4.eval())

  # Build a SavedModel with multi-tags, multi-meta-graphs, and multi-signature

  default_v3_def = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={
      "v1": v1,
      "v2": v2,
    },
    outputs={
      "v3": v3
    }
  )

  predict_v4_def = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={
      "v1": v1,
      "v2": v2,
    },
    outputs={
      "v4": v4
    }
  )


  builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_v3_def,
      "predict_v4": predict_v4_def
    }
  )

  builder.save(as_text=False)