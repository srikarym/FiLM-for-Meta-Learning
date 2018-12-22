import tensorflow as tf

def film_channels_weights(weights,planes,string):
	weights[string+"_w"] = tf.get_variable(string+"_w",planes,dtype=tf.float32)
	weights[string+"_b"] = tf.get_variable(string+"_b",planes,dtype=tf.float32)
	return weights

def conv_block_weight(weights, inplanes, out_planes, string, kernel=3):
	init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
	weights[string] = tf.get_variable(string, [kernel, kernel, inplanes, out_planes], initializer=init, dtype=tf.float32)
	return weights

def downsample_weights(weights,inplanes,planes,expansion,string):
	init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
	weights[string] = tf.get_variable(string, [1, 1, inplanes, planes*expansion], initializer=init, dtype=tf.float32)
	return weights

def BasicBlock_weights(weights,inplanes,planes,string,isdownsample=False):
	expansion = 1
	weights = conv_block_weight(weights, inplanes, planes, string+str(0), kernel=3)
	weights = film_channels_weights(weights,planes,"film_"+string+str(0))
	weights = conv_block_weight(weights, planes, planes, string+str(1), kernel=3)
	weights = film_channels_weights(weights,planes,"film_"+string+str(1))
	if isdownsample:
		weights = downsample_weights(weights, inplanes, planes, expansion, string+str(0)+"_down")
	return weights

def Bottleneck_weights(weights,inplanes,planes,string,isdownsample=False):
	expansion = 4
	weights = conv_block_weight(weights, inplanes, planes, string+str(0), kernel=1)
	weights = film_channels_weights(weights,planes,"film_"+string+str(0))
	weights = conv_block_weight(weights, planes, planes, string+str(1), kernel=3)
	weights = film_channels_weights(weights,planes,"film_"+string+str(1))
	weights = conv_block_weight(weights, planes, planes*expansion, string+str(2), kernel=1)
	weights = film_channels_weights(weights,planes*expansion,"film_"+string+str(2))
	if isdownsample:
		weights = downsample_weights(weights, inplanes, planes, expansion, string+str(0)+"_down")
	return weights

# only one of isBasicBlock or isBottleNeck can be true
def make_layer_weights(weights,isBasicBlock,isBottleNeck,inplanes,planes,blocks,string,stride=1):
	if isBasicBlock:
		expansion = 1
		if ((stride!=1) or(inplanes != planes*expansion)):
			weights = BasicBlock_weights(weights,inplanes,planes,string+str(0)+"_",True)
		else:
			weights = BasicBlock_weights(weights,inplanes,planes,string+str(0)+"_",False)
		inplanes = planes*expansion
		for i in range(1,blocks):
			weights = BasicBlock_weights(weights,inplanes,planes,string+str(i)+"_")

	if isBottleNeck:
		expansion = 4
		if ((stride!=1) or(inplanes != planes*expansion)):
			weights = Bottleneck_weights(weights,inplanes,planes,string+str(0)+"_",True)
		else:
			weights = Bottleneck_weights(weights,inplanes,planes,string+str(0)+"_",True)
		inplanes = planes*expansion
		for i in range(1,blocks):
			weights = Bottleneck_weights(weights,inplanes,planes,string+str(i)+"_",True)

	return weights

def ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes):
	inplanes = 64
	if isBasicBlock:
		expansion = 1
	if isBottleNeck:
		expansion = 4
	weights = conv_block_weight(weights, 1, 64, "conv1", kernel=7)
	weights = film_channels_weights(weights,64,"film_conv1")

	weights = make_layer_weights(weights,isBasicBlock,isBottleNeck,inplanes,64,layers[0],"res2_",stride=1)
	inplanes = 64*expansion
	weights = make_layer_weights(weights,isBasicBlock,isBottleNeck,inplanes,128,layers[1],"res3_",stride=2)
	inplanes = 128*expansion
	weights = make_layer_weights(weights,isBasicBlock,isBottleNeck,inplanes,256,layers[2],"res4_",stride=2)
	inplanes = 256*expansion
	weights = make_layer_weights(weights,isBasicBlock,isBottleNeck,inplanes,512,layers[3],"res5_",stride=2)
	inplanes = 512*expansion

	string = "fc_1_w"
	init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
	weights[string] = tf.get_variable(string, [512*expansion, num_classes], initializer=init, dtype=tf.float32)

	string = "fc_1_b"
	weights[string] = tf.get_variable(string, [num_classes], initializer=init, dtype=tf.float32)

	return weights

def resnet18_weights(num_classes):
	layers = [2,2,2,2]
	isBasicBlock = True
	isBottleNeck = False
	weights = {}
	weights = ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes)
	return weights

def resnet34_weights(num_classes):
	layers = [3,4,6,3]
	isBasicBlock = True
	isBottleNeck = False
	weights = {}
	weights = ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes)
	return weights

def resnet50_weights(num_classes):
	layers = [3,4,6,3]
	isBasicBlock = False
	isBottleNeck = True
	weights = {}
	weights = ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes)
	return weights

def resnet101_weights(num_classes):
	layers = [3,4,23,3]
	isBasicBlock = False
	isBottleNeck = True
	weights = {}
	weights = ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes)
	return weights

def resnet152_weights(num_classes):
	layers = [3,4,36,3]
	isBasicBlock = False
	isBottleNeck = True
	weights = {}
	weights = ResNet_weights(weights,isBasicBlock,isBottleNeck,layers,num_classes)
	return weights
