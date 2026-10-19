#!/usr/bin/python3
"""Evaluate a released CrossLocate checkpoint on a test split."""

import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    choices=['segments', 'depth', 'farfield_charles', 'farfield_harbor', 'farfield_charles_void', 'farfield_harbor_void',
                             'farfield_charles_eye32', 'farfield_harbor_eye40', 'farfield_train',
                             'farfield_train_full', 'farfield_graded',
                             'farfield_uniform_charles', 'farfield_uniform_charles_eye32',
                             'farfield_uniform_harbor', 'farfield_uniform_harbor_eye40'])
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--outputPath', required=True)
parser.add_argument('--set', default='test')
parser.add_argument('--inputSize', type=int, default=500)   # released models were trained at 500x500
parser.add_argument('--inputHeight', type=int)   # overrides --inputSize; use with --inputWidth for non-square
parser.add_argument('--inputWidth', type=int)
parser.add_argument('--locationThresholds', default='20,100,1000',
                    type=lambda s: [int(v) for v in s.split(',')])
parser.add_argument('--recallNumbers', default='1,10,100',   # ascending; the last sets n_neighbors
                    type=lambda s: [int(v) for v in s.split(',')])
parser.add_argument('--locationThresholdStep', type=int, default=10)  # every locationThreshold must be a multiple of it
parser.add_argument('--successfulLocalizationThreshold', type=int, default=20)
parser.add_argument('--distances', default='utm', choices=['utm', 'wgs'])   # wgs is Vincenty per query, O(Nq*Ndb)
parser.add_argument('--freshDescriptors', action='store_true', help='ignore cached descriptors')
args = parser.parse_args()

from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)

import numpy as np
import tensorflow as tf

from core.DatasetDescriptions import (datasetOriginalToSegmentsSwiss, datasetOriginalToDepthSwiss,
                                      datasetFarfieldCharlesDepth, datasetFarfieldHarborDepth,
    datasetFarfieldCharlesVoidDepth, datasetFarfieldHarborVoidDepth,
    datasetFarfieldCharlesEye32Depth, datasetFarfieldHarborEye40Depth,
    datasetFarfieldTrainDepth, datasetFarfieldTrainFullDepth, datasetFarfieldGradedDepth,
    datasetFarfieldUniformCharlesDepth, datasetFarfieldUniformCharlesEye32Depth,
    datasetFarfieldUniformHarborDepth, datasetFarfieldUniformHarborEye40Depth)
from core.Dataset import Dataset
from core.TensorAssembler import TensorAssembler, TensorFormat
from core.NetworkBuilder import NetworkBuilder, Branching, Metric
from core.DescriptorExtractor import DescriptorExtractor
from core.Evaluator import Evaluator
from networks.NeuralNetwork import Architecture, Pooling, FeatureNormalization

datasetParams = {'segments': datasetOriginalToSegmentsSwiss,
                 'depth': datasetOriginalToDepthSwiss,
                 'farfield_charles': datasetFarfieldCharlesDepth,
                 'farfield_harbor': datasetFarfieldHarborDepth,
                 'farfield_charles_void': datasetFarfieldCharlesVoidDepth,
                 'farfield_harbor_void': datasetFarfieldHarborVoidDepth,
                 'farfield_charles_eye32': datasetFarfieldCharlesEye32Depth,
                 'farfield_harbor_eye40': datasetFarfieldHarborEye40Depth,
                 'farfield_train': datasetFarfieldTrainDepth,
                 'farfield_train_full': datasetFarfieldTrainFullDepth,
                 'farfield_graded': datasetFarfieldGradedDepth,
                 'farfield_uniform_charles': datasetFarfieldUniformCharlesDepth,
                 'farfield_uniform_charles_eye32': datasetFarfieldUniformCharlesEye32Depth,
                 'farfield_uniform_harbor': datasetFarfieldUniformHarborDepth,
                 'farfield_uniform_harbor_eye40': datasetFarfieldUniformHarborEye40Depth}[args.dataset]

netParams = {'inputDimensions': (args.inputHeight or args.inputSize, args.inputWidth or args.inputSize),
             'branching': Branching.ONE_BRANCH,
             'architecture': Architecture.VGG16,
             'preFeatNorm': FeatureNormalization.L2,
             'pooling': Pooling.MAC,
             'postFeatNorm': FeatureNormalization.L2}

_bad = [t for t in args.locationThresholds if (t % args.locationThresholdStep != 0)]
if _bad:
    sys.exit('locationThresholds %s are not multiples of --locationThresholdStep %d' % (_bad, args.locationThresholdStep))

evalParams = {'recallNumbers': args.recallNumbers,
              'locationThresholds': args.locationThresholds,
              'locationThresholdStep': args.locationThresholdStep,
              'successfulLocalizationThreshold': args.successfulLocalizationThreshold,
              'yawThreshold': 30}

descriptorsPath = os.path.join(args.outputPath, 'descriptors')
evaluationsPath = os.path.join(args.outputPath, 'evaluations')
for p in (descriptorsPath, evaluationsPath):
    os.makedirs(p, exist_ok=True)

datasetsPath = '../datasets'
fullDatasetPath = os.path.join(datasetsPath, datasetParams['name'])
databasePath = os.path.join(fullDatasetPath, datasetParams['database'])
queriesPath = os.path.join(fullDatasetPath, datasetParams['queries'])

print('----- Dataset -----')
dataset = Dataset(datasetParams['name'], datasetParams['meta'], args.set, databasePath, queriesPath,
                  positiveLocationThreshold=20, positiveYawThreshold=15, negativeLocationMinThreshold=2000)
dataset.loadFromMetaStructs()
if args.distances == 'wgs':
    dataset.precomputeLocationDistancesWGS()
else:
    dataset.precomputeLocationDistancesUTM()

tensorAssembler = TensorAssembler(datasetParams['meta'], databasePath, queriesPath,
                                  netParams['inputDimensions'], 3,
                                  TensorFormat.INPUT_RGB, TensorFormat.INPUT_RGB)

print('----- Network -----')
networkBuilder = NetworkBuilder(netParams['inputDimensions'], 3, 5, Metric.EUCLIDEAN, 'Adam')
graph, model, _, _, _, initializer, _ = networkBuilder.build(
    netParams['branching'], netParams['architecture'],
    netParams['preFeatNorm'], netParams['pooling'], netParams['postFeatNorm'], 0.1)

extractor = DescriptorExtractor(dataset, tensorAssembler, descriptorsPath, netParams['inputDimensions'])
evaluator = Evaluator(dataset, evalParams['recallNumbers'], evalParams['locationThresholds'],
                      evalParams['locationThresholdStep'], yawThreshold=evalParams['yawThreshold'],
                      successfulLocalizationThreshold=evalParams['successfulLocalizationThreshold'],
                      outputPath=evaluationsPath)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(graph=graph, config=config) as sess:
    sess.run(initializer)
    with graph.as_default():
        saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver.restore(sess, args.checkpoint)
    print('Restored', args.checkpoint)

    if args.freshDescriptors:
        print('[evaluate] --freshDescriptors: ignoring any cached descriptors in', descriptorsPath)
        dbDescriptors, qDescriptors = None, None
    else:
        dbDescriptors, qDescriptors = extractor.loadDatasetDescriptors()
    if (dbDescriptors is not None) and (len(dbDescriptors) != len(dataset.getDatabaseImageFns())):
        sys.exit('cached db descriptors have %d rows, dataset has %d — delete %s'
                 % (len(dbDescriptors), len(dataset.getDatabaseImageFns()), descriptorsPath))
    if (dbDescriptors is not None) and (qDescriptors is None):
        # db already in hand (sets sharing one db copy it across); extract queries only
        qDescriptors = extractor.extractQueryDescriptors(sess, qModel=model)
    elif (dbDescriptors is None) or (qDescriptors is None):
        dbDescriptors, qDescriptors = extractor.extractDatasetDescriptors(sess, dbModel=model, qModel=model)

print('----- Recalls -----')
evaluator.setDescriptors(dbDescriptors, qDescriptors)
posDistRecalls, posOrientDistRecalls, posRecalls, posOrientRecalls, successes, fails = \
    evaluator.computeRecallsAtNs(withPrint=True)

results = {
    'dataset': args.dataset,
    'datasetParams': datasetParams,
    'set': args.set,
    'distances': args.distances,
    'locationThresholds': evalParams['locationThresholds'],
    'checkpoint': args.checkpoint,
    'inputDimensions': list(netParams['inputDimensions']),
    'numQueries': len(dataset.getQueryImageFns()),
    'numDatabase': len(dataset.getDatabaseImageFns()),
    'recallNumbers': evalParams['recallNumbers'],
    'positionRecalls': {str(k): list(map(float, v)) for k, v in posRecalls.items()},
    'positionAndOrientationRecalls': {str(k): list(map(float, v)) for k, v in posOrientRecalls.items()},
    'numSuccesses': len(successes),
    'numFails': len(fails),
}
with open(os.path.join(args.outputPath, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
np.save(os.path.join(evaluationsPath, 'positionDistanceRecalls.npy'), posDistRecalls)
np.save(os.path.join(evaluationsPath, 'positionAndOrientationDistanceRecalls.npy'), posOrientDistRecalls)
print('Wrote', os.path.join(args.outputPath, 'results.json'))
