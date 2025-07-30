import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# Load model once at startup
@st.cache_resource
def load_model():
    model_path = 'models/Shinkai_53.pb'
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def cartoonize_image(image_np, graph):
    image_resized = cv2.resize(image_np, (256, 256))
    image_norm = image_resized / 127.5 - 1.0
    image_input = np.expand_dims(image_norm, axis=0)

    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('generator_input:0')
        output_tensor = graph.get_tensor_by_name('generator/G_MODEL/out_layer/Tanh:0')
        output = sess.run(output_tensor, feed_dict={input_tensor: image_input})
        output = (output[0] + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Streamlit UI
st.title("🎨 Cartoonify Your Image (Anime Style)")
st.markdown("Upload an image and get a cartoon version using the AnimeGAN model!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption='Original Image', use_container_width=True)

    with st.spinner("🌀 Cartoonifying..."):
        graph = load_model()
        cartoon = cartoonize_image(image_np, graph)

    st.image(cartoon, caption='Cartoonified Image', use_container_width=True)

    # Download button
    cartoon_pil = Image.fromarray(cartoon)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cartoon_pil.save(tmp.name)
        st.download_button("⬇️ Download Cartoon Image", data=open(tmp.name, "rb"), file_name="cartoonified.jpg", mime="image/jpeg")
        #os.unlink(tmp.name)
