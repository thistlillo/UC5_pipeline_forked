import numpy as np
import pyecvl.ecvl as ecvl


mean = [0.48197903, 0.48197903, 0.48197903]
std = [0.26261734, 0.26261734, 0.26261734]

# mean = np.array([0.48197903, 0.48197903, 0.48197903]) * 255
# std = np.array([0.26261734, 0.26261734, 0.26261734]) * 255


train_augs = ecvl.SequentialAugmentationContainer([
                ecvl.AugResizeDim([300, 300]),
                ecvl.AugRandomCrop([224, 224]),  # XXX should be parametric, for resnet 18
                ecvl.AugToFloat32(divisor=255.0),
                ecvl.AugNormalize(mean, std),
            ])

test_augs =  ecvl.SequentialAugmentationContainer([
                ecvl.AugResizeDim([300, 300]),
                ecvl.AugCenterCrop([224, 224]),
                # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
                ecvl.AugToFloat32(divisor=255.0),
                ecvl.AugNormalize(mean, std),
            ])

# train_augs = ecvl.SequentialAugmentationContainer([
#                 ecvl.AugResizeDim([300, 300]),
#                 ecvl.AugRandomCrop([224, 224]),  # XXX should be parametric, for resnet 18
#                 ecvl.AugToFloat32(),
#                 ecvl.AugNormalize(mean, std),
#             ])

# test_augs =  ecvl.SequentialAugmentationContainer([
#                 ecvl.AugResizeDim([224, 224]),
#                 # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
#                 ecvl.AugToFloat32(),
#                 ecvl.AugNormalize(mean, std),
#             ])

test_augs2 =  ecvl.SequentialAugmentationContainer([
                ecvl.AugResizeDim([300, 300]),
                ecvl.AugCenterCrop([224, 224]),
                # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
                ecvl.AugToFloat32(),
                ecvl.AugNormalize(mean, std),
            ])

# train_augs = lambda size: ecvl.SequentialAugmentationContainer([
#                 ecvl.AugResizeDim([300, 300]),
#                 ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
#                 ecvl.AugToFloat32(),
#                 ecvl.AugNormalize(mean, std),
#             ])

# test_augs =  lambda size: ecvl.SequentialAugmentationContainer([
#                 ecvl.AugResizeDim([size, size]),
#                 # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
#                 ecvl.AugToFloat32(),
#                 ecvl.AugNormalize(mean, std),
#             ])

# test_augs2 =  lambda size: ecvl.SequentialAugmentationContainer([
#                 ecvl.AugResizeDim([300, 300]),
#                 ecvl.AugCenterCrop([size, size]),
#                 # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
#                 ecvl.AugToFloat32(),
#                 ecvl.AugNormalize(mean, std),
#             ])