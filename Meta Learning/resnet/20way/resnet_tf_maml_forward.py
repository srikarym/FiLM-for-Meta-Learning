import tensorflow as tf
import numpy as np
import resnet_tf_maml_weights

def film_channels_forward(inp,weights,string):
	return tf.add(tf.multiply(inp,weights[string+"_w"]),weights[string+"_b"])

def conv_block_forward(inp, weights, string, stride=1):
	stride_size = [1,stride,stride,1]
	return tf.nn.conv2d(inp, weights[string], stride_size, 'SAME')

def downsample_forward(inp,weights,string,stride):
	out = conv_block_forward(inp, weights, string, stride=stride)
	return tf.layers.batch_normalization(out)

def BasicBlock_forward(inp,weights,string,stride=1,isdownsample=False):
	residual = inp
	out = conv_block_forward(inp, weights, string+str(0), stride=stride)
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_"+string+str(0))
	out = tf.nn.relu(out)

	out = conv_block_forward(out, weights, string+str(1), stride=1)
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_"+string+str(1))
	if isdownsample:
		residual = downsample_forward(inp,weights,string+str(0)+"_down",stride)

	out = out + residual
	out = tf.nn.relu(out)
	return out

def Bottleneck_forward(inp,weights,string,stride=1,isdownsample=False):
	expansion = 4
	residual = inp

	out = conv_block_forward(inp, weights, string+str(0))
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_"+string+str(0))
	out = tf.nn.relu(out)

	out = conv_block_forward(out, weights, string+str(1), stride=stride)
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_"+string+str(1))
	out = tf.nn.relu(out)

	out = conv_block_forward(out, weights, string+str(2))
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_"+string+str(2))
	if isdownsample:
		residual = downsample_forward(inp,weights,string+str(0)+"_down",stride)

	out = out + residual
	out = tf.nn.relu(out)
	return out

# only one of isBasicBlock or isBottleNeck can be true
def make_layer_forward(inp,weights,isBasicBlock,isBottleNeck,inplanes,planes,blocks,string,stride=1):
	if isBasicBlock:
		expansion = 1
		if ((stride!=1) or(inplanes != planes*expansion)):
			out = BasicBlock_forward(inp,weights,string+str(0)+"_",stride,True)
		else:
			out = BasicBlock_forward(inp,weights,string+str(0)+"_",stride,False)
		inplanes = planes*expansion
		for i in range(1,blocks):
			out = BasicBlock_forward(out,weights,string+str(i)+"_")

	if isBottleNeck:
		expansion = 4
		if ((stride!=1) or(inplanes != planes*expansion)):
			out = Bottleneck_forward(inp,weights,string+str(0)+"_",stride,True)
		else:
			out = BasicBlock_forward(inp,weights,string+str(0)+"_",stride,False)
		inplanes = planes*expansion
		for i in range(1,blocks):
			out = Bottleneck_forward(out,weights,string+str(i)+"_")

	return out;

def ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers,num_classes=5):
	inplanes = 64
	if isBasicBlock:
		expansion = 1
	if isBottleNeck:
		expansion = 4
	out = conv_block_forward(inp, weights, "conv1", stride=2)
	out = tf.layers.batch_normalization(out)
	out = film_channels_forward(out,weights,"film_conv1")
	out = tf.nn.relu(out)
	out = tf.nn.max_pool(out, [1,3,3,1], [1,2,2,1],"SAME")

	out = make_layer_forward(out,weights,isBasicBlock,isBottleNeck,inplanes,64,layers[0],"res2_",stride=1)
	inplanes = 64*expansion
	out = make_layer_forward(out,weights,isBasicBlock,isBottleNeck,inplanes,128,layers[1],"res3_",stride=2)
	inplanes = 128*expansion
	out = make_layer_forward(out,weights,isBasicBlock,isBottleNeck,inplanes,256,layers[2],"res4_",stride=2)
	inplanes = 256*expansion
	out = make_layer_forward(out,weights,isBasicBlock,isBottleNeck,inplanes,512,layers[3],"res5_",stride=2)
	inplanes = 512*expansion

	# VERY IMPORTANT UNCOMMENT THE BELOW COMMENTED LINE FOR LARGE INPUT SHAPES
	# out = tf.nn.pool(out,[1,7,7,1],"AVG","SAME",strides=[1,1,1,1])
	out = tf.reshape(out, [-1, np.prod([int(dim) for dim in out.get_shape()[1:]])])
	out = tf.matmul(out,weights["fc_1_w"]) + weights["fc_1_b"]

	return out

def resnet18_forward(inp,weights):
	layers = [2,2,2,2]
	isBasicBlock = True
	isBottleNeck = False
	out = ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers)
	return out

def resnet34_forward(inp,weights):
	layers = [3,4,6,3]
	isBasicBlock = True
	isBottleNeck = False
	out = ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers)
	return out

def resnet50_forward(inp,weights):
	layers = [3,4,6,3]
	isBasicBlock = False
	isBottleNeck = True
	out = ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers)
	return out

def resnet101_forward(inp,weights):
	layers = [3,4,23,3]
	isBasicBlock = False
	isBottleNeck = True
	out = ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers)
	return out

def resnet152_forward(inp,weights):
	layers = [3,4,36,3]
	isBasicBlock = False
	isBottleNeck = True
	out = ResNet_forward(inp,weights,isBasicBlock,isBottleNeck,layers)
	return out
