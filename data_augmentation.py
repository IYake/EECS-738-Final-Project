import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMAGE_SIZE = 256

def resize_image(img):
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = img[:, :, :3]# Do not read alpha channel.
        resized_img = sess.run(tf_img, feed_dict = {X: img})
    return resized_img


def flip_image(img):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flipped_img = sess.run(tf_img1, feed_dict = {X: img})
    return flipped_img


def crop_image(img):
    crop_img = img[0:227, 0:227]
    return crop_img


def data_aug(imgPath):
    img = mpimg.imread(imgPath)
    plt.subplot(221)
    plt.title("original image")
    plt.axis('off')
    plt.imshow(img)
    
    resized_img = resize_image(img)
    plt.subplot(222)
    plt.title("resized image")
    plt.imshow(resized_img)
    plt.axis('off')
    
    flipped_img = flip_image(resized_img)
    plt.subplot(223)
    plt.title("flipped image")
    plt.axis('off')
    plt.imshow(flipped_img)
    
    cropped_img = crop_image(resized_img)
    plt.subplot(224)
    plt.imshow(cropped_img)
    plt.axis('off')
    plt.title("cropped image")
    plt.show()
    
    imgs =[]
    imgs.append(resized_img)
    imgs.append(flipped_img)
    imgs.append(cropped_img)
    return imgs


data_aug('Presentation/image_dog.png')
