import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = 'C:\\project\\caffe\\'

import caffe

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(50, 3,227, 227)
image = caffe.io.load_image('E:\\happy_dog.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0]
print 'predicted class is:', output_prob.argmax()
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]

top_inds = output_prob.argsort()[::-1][:5]
print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])