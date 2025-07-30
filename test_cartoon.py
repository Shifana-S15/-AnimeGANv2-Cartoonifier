import tensorflow as tf
import cv2
import numpy as np

# Load your frozen .pb model
model_path = 'models/Shinkai_53.pb'  # Update if your path differs

with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

# Load image
input_image_path = 'h2.jpg'  # Replace with your image path
image = cv2.imread(input_image_path)
image = cv2.resize(image, (256, 256))  # Required size for model
image = image / 127.5 - 1.0  # Normalize to [-1, 1]
image = np.expand_dims(image, axis=0)

# Start session
with tf.compat.v1.Session(graph=graph) as sess:
    input_tensor = graph.get_tensor_by_name('generator_input:0')
    output_tensor = graph.get_tensor_by_name('generator/G_MODEL/out_layer/Tanh:0')

    output = sess.run(output_tensor, feed_dict={input_tensor: image})
    output = (output[0] + 1) * 127.5  # Rescale to [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)

# Save cartoonified image
cv2.imwrite('output_cartoon.jpg', output)
print("✅ Cartoonified image saved as output_cartoon.jpg")
