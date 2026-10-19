#!/usr/bin/python3
"""
	CrossLocate: Cross-Modal Large-Scale Visual Geo-Localization in Natural Environments Using Rendered Modalities
	WACV 2022

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2022, Jan Tomesek"


# Properties
# CL_EXPERIMENT / CL_PRETRAINED let run 1 and run 2 differ without editing this file; CL_PRETRAINED
# may be 'IMAGENET', an output/<name> dir, or a checkpoint prefix path (CROSSLOCATE_PLAN.md sec 9).
import os as _os
experimentName = _os.environ.get('CL_EXPERIMENT', 'CrossLocate')
tag = 'train'


# Parameters
from core.DatasetDescriptions import datasetRenderV1, datasetFCN8sV1, datasetDeeplabV1, datasetOriginalV1, datasetDepthV1, datasetSilhouettesV1, datasetAustriaQ1
from core.DatasetDescriptions import datasetSegments, datasetSilhouettes, datasetDepth, datasetOriginalToSegments, datasetOriginalToSilhouettes, datasetOriginalToDepth
from core.DatasetDescriptions import datasetUniformSegments2000, datasetUniformSegments500, datasetUniformOriginalToSegments500
from core.DatasetDescriptions import datasetSegmentsSwiss, datasetOriginalToSegmentsSwiss
from core.DatasetDescriptions import datasetOriginalToSilhouettesSwiss, datasetOriginalToDepthSwiss
from core.DatasetDescriptions import datasetOriginalToSegmentsSwissResolution
from core.DatasetDescriptions import datasetAlpsGP3KSegmentsCompact, datasetAlpsGP3KOriginalToSegmentsCompact, datasetAlpsPhotosToSegmentsCompact
from core.DatasetDescriptions import datasetAlpsPhotosToSegmentsResolutionCompact
from core.DatasetDescriptions import datasetAlpsPhotosToSilhouettesCompact
from core.DatasetDescriptions import datasetAlpsPhotosToDepthCompact, datasetAlpsPhotosToDepthResolutionCompact
from core.DatasetDescriptions import datasetAlpsCH1ToSegments, datasetAlpsCH1ToSegmentsResolution
from core.DatasetDescriptions import datasetAlpsCH1ToSilhouettes
from core.DatasetDescriptions import datasetAlpsCH1ToDepth, datasetAlpsCH1ToDepthResolution
from core.DatasetDescriptions import datasetAlpsCH2ToSegments, datasetAlpsCH2ToSegmentsResolution, datasetAlpsCH2ToSegmentsExt, datasetAlpsCH2ToSegmentsExtResolution
from core.DatasetDescriptions import datasetAlpsCH2ToSilhouettes, datasetAlpsCH2ToSilhouettesExt
from core.DatasetDescriptions import datasetAlpsCH2ToDepth, datasetAlpsCH2ToDepthResolution, datasetAlpsCH2ToDepthExt, datasetAlpsCH2ToDepthExtResolution
from core.DatasetDescriptions import datasetFarfieldTrainDepth, datasetFarfieldTrainSparseDepth, datasetFarfieldGradedDepth
from core.NetworkBuilder import Branching, Metric
from networks.NeuralNetwork import Architecture, Pooling, FeatureNormalization
from core.TensorAssembler import TensorFormat
from core.TripletGenerator import MiningStrategy
from core.Plotter import RecallType

# 'released' reproduces the shipped defaults verbatim; 'farfield' trains on Charles/Harbor
# data with the sec 7 yaw-aware mining fix (CROSSLOCATE_PLAN.md sec 7-8).
trainingConfig = 'farfield'

if (trainingConfig == 'released'):
	netParams = { 	'inputDimensions': (224, 224),
				 	'branching': Branching.ONE_BRANCH,
				 	'architecture': Architecture.VGG16,
				 	'preFeatNorm': FeatureNormalization.L2,
					'pooling': Pooling.MAC,
				 	'postFeatNorm': FeatureNormalization.L2,
				 	'pretrained': 'IMAGENET' }

	trainParams = { 'startEpoch': 1,
					'endEpoch': 20,
					'batchSize': 4,
					'margin': 0.1,
					'queryExtractFeatsInterval': 250,
					'queryTensorFormat': TensorFormat.INPUT_RGB,
					'databaseTensorFormat': TensorFormat.INPUT_RGB,
					'metric': Metric.EUCLIDEAN }

	mineParams = {  'positiveLocationThreshold': 20,
	                'positiveYawThreshold': 15,
	                'negativeLocationMinThreshold': 2000,
	                'margin': 0.2,
	                'numberOfNegatives': 10,
	                'negativeSampleAttempts': 2,
	                'numberOfSampledNegatives': 4000,
	                'reusingNegatives': True,
	                'trainMiningStrategy': MiningStrategy.SEMIHARD_AND_HARD_NEGATIVE,
					'numberOfTrainTripletReusages': 1 }

	# 'released' config keeps the original scalar-margin, descriptor-mining behavior untouched.
	bankParams = { 'adaptiveMargin': False, 'useBands': False }

	evalParams = {	'recallNumbers' : [1, 10, 100],
				  	'locationThresholds': [20, 100, 1000],
				  	'locationThresholdStep': 20,
				  	'successfulLocalizationThreshold': 20,
				  	'yawThreshold': 30 }

	datasetParams = datasetSegments

elif (trainingConfig == 'farfield'):
	# inputDimensions per sec 1c depth hyperparameters (VGG16/MAC/L2-L2/ONE_BRANCH already match released)
	# (height, width). Non-square is fine: VGG16 is all SAME padding and MAC pools over both
	# spatial axes, so only the input placeholder and the resize target change.
	netParams = { 	'inputDimensions': (int(_os.environ.get('CL_INPUT_H', 500)),
									    int(_os.environ.get('CL_INPUT_W', 500))),
				 	'branching': Branching.ONE_BRANCH,
				 	'architecture': Architecture.VGG16,
				 	'preFeatNorm': FeatureNormalization.L2,
					'pooling': Pooling.MAC,
				 	'postFeatNorm': FeatureNormalization.L2,
				 	'pretrained': _os.environ.get('CL_PRETRAINED', 'IMAGENET') }

	# batchSize/endEpoch/margin per sec 1c depth hyperparameters
	trainParams = { 'startEpoch': int(_os.environ.get('CL_START_EPOCH', 1)),
					# CL_END_EPOCH lets a run be extended later without editing this file; resuming
					# with CL_START_EPOCH restores the latest checkpoint and the history.
					'endEpoch': int(_os.environ.get('CL_END_EPOCH', 30)),
					'batchSize': int(_os.environ.get('CL_BATCH', 3)),
					'margin': 0.1,
					# Supplementary specifies this PER TRIPLET: 250 (Sparse) / 1000 (Uniform). Our query
					# set is 6.8x theirs, so a faithful 250 costs 39 db passes/epoch (4.2 h). 1000 is
					# the paper's own Uniform value and 1.75x fresher than the 1750 first used here.
					'queryExtractFeatsInterval': int(_os.environ.get('CL_EXTRACT_INTERVAL', 1000)),
					'queryTensorFormat': TensorFormat.INPUT_RGB,
					'databaseTensorFormat': TensorFormat.INPUT_RGB,
					'metric': Metric.EUCLIDEAN }

	# Paper's Sparse values (supplementary): positives <20 m and within 15 deg, negatives >2000 m
	# with NO yaw constraint. The earlier retune assumed a 1.7 km basin; the combined
	# Charles+Harbor set spans 18 km, where 20 m gives 5.0 positives/query (0 starved) and
	# 2000 m leaves 67.6% of the db as negatives — measured, so no retune is warranted.
	mineParams = {  'positiveLocationThreshold': int(_os.environ.get('CL_POS_LOC', 20)),
	                'positiveYawThreshold': int(_os.environ.get('CL_POS_YAW', 15)),
	                'negativeLocationMinThreshold': int(_os.environ.get('CL_NEG_LOC', 2000)),
	                'negativeYawThreshold': (int(_os.environ['CL_NEG_YAW']) if _os.environ.get('CL_NEG_YAW') else None),
	                'margin': 0.2,
	                # 8 = 1 positive + band budget 2/4/2 (bankParams['negativesPerBand']).
	                'numberOfNegatives': 8,
	                'negativeSampleAttempts': 2,
	                'numberOfSampledNegatives': 4000,
	                'reusingNegatives': True,
	                'trainMiningStrategy': (MiningStrategy.SIMILARITY_BANDS
	                                        if (_os.environ.get('CL_ADAPTIVE_MARGIN', '1') == '1')
	                                        else MiningStrategy.SEMIHARD_AND_HARD_NEGATIVE),
				# queryExtractFeatsInterval is spec'd per triplet but only divided by batchSize, so a
				# reusage of 3 refreshes 3x more often unless CL_EXTRACT_INTERVAL is scaled with it.
				'numberOfTrainTripletReusages': int(_os.environ.get('CL_REUSE', 1)) }

	# useBands defaults with adaptiveMargin: bank-supplied negatives make descriptor-based mining moot.
	bankParams = { 'adaptiveMargin': _os.environ.get('CL_ADAPTIVE_MARGIN', '1') == '1',
				   # 0.27 puts the median bank margin (0.368 at alpha 1) on trainParams margin 0.1, so the
				   # first graded run differs from the fixed-margin baseline in shape only, not strength.
				   'alpha': float(_os.environ.get('CL_MARGIN_ALPHA', 0.27)),
				   'bankPath': _os.environ.get('CL_BANK_PATH', '/home/aneesa/results/render_matching/2026-09-11_similarity_labels_bank'),
				   'positiveFloor': float(_os.environ.get('CL_POS_FLOOR', 0.50)),
				   'bandEdges': [0.35, 0.18],
				   'negativesPerBand': [2, 4, 2],
				   # margins now come from warp cells (see TripletGenerator), so the bank path needs them loaded
				   'loadWarps': _os.environ.get('CL_BANK_LOAD_WARPS', '1') == '1' }
	bankParams['useBands'] = bankParams['adaptiveMargin']

	# yawThreshold replaces the bare 30 literal at the Evaluator calls below
	evalParams = {	'recallNumbers' : [1, 10, 100],
				  	'locationThresholds': [20, 50, 100, 200, 1000],
				  	# Evaluator indexes locationThresholds into range(step, max+1, step), so every
				  	# threshold must be a multiple of the step -- 50 is not a multiple of 20.
				  	'locationThresholdStep': 10,
				  	'successfulLocalizationThreshold': 50,
				  	'yawThreshold': 30 }

	# The graded set is the bank's own grid, so band selection needs it whenever useBands is on.
	# Otherwise uses the sparse dataset, which is only useful for rapid training and not for anything you
	# want to run real evaluation experiments on.
	datasetParams = (datasetFarfieldGradedDepth if (bankParams['useBands'])
					 else datasetFarfieldTrainSparseDepth)

else:
	raise ValueError('Unknown trainingConfig: {}'.format(trainingConfig))

_bad = [t for t in evalParams['locationThresholds'] if (t % evalParams['locationThresholdStep'] != 0)]
if (_bad):
	raise ValueError('locationThresholds {} are not multiples of locationThresholdStep {} — '
					 'Evaluator.__getValueAtThreshold will raise "not in list" after the first '
					 'descriptor extraction'.format(_bad, evalParams['locationThresholdStep']))

learnParams = {	'optimizer': 'Adam',
				'learningRate': 0.00001,
			   	'learningRateDownscaleEpochs': [],
			   	'learningRateDownscaleFactors': [] }

augmentParams = {   'augmentation': True }

plotParams = { 'distanceRecallsXTicks': [0, 20, 100, 250, 500, 750, 1000] }

# batchVisualLimit: the released code visualises EVERY batch of these epochs — 3207 batches x 21
# images cost 3.0 GB and ~2.7 s/batch, 6x the 0.43 s training step itself. The first few batches
# carry all the sanity-check value.
# The val triplet loss only feeds plotValEpochLoss - nothing else reads valEpochLossHistory - and a
# per-split val db cannot supply the 4000 sampled negatives the loss is defined over.
valLossEnabled = _os.environ.get('CL_VAL_LOSS', '0') == '1'

visualParams = { 'batchVisualLimit': 20,
				 'batchVisualEpochs': [trainParams['startEpoch'], trainParams['endEpoch']],
				 'successesVisualEpochs': [trainParams['endEpoch']],
				 'failsVisualEpochs': [trainParams['endEpoch']] }

techParams = { 'moveToSSD': False }

outputDirectories = {	'charts': 'charts',
						'descriptors': 'descriptors',
					 	'evaluations': 'evaluations',
						'history': 'history',
						'models': 'models',
					 	'visuals': 'visuals' }


# Reproducibility
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
# still possible randomness from CuDNN


# Imports
import tensorflow as tf
import os
import numpy as np
from math import floor
from shutil import rmtree


# Output paths and directories
fullOutputPath = os.path.join('output', experimentName)
chartsPath = os.path.join(fullOutputPath, outputDirectories['charts'])
descriptorsPath = os.path.join(fullOutputPath, outputDirectories['descriptors'])
evaluationsPath = os.path.join(fullOutputPath, outputDirectories['evaluations'])
historyPath = os.path.join(fullOutputPath, outputDirectories['history'])
modelsPath = os.path.join(fullOutputPath, outputDirectories['models'])
visualsPath = os.path.join(fullOutputPath, outputDirectories['visuals'])

# exist_ok so a CL_START_EPOCH resume does not die on directories its first run created
os.makedirs(chartsPath, exist_ok=True)
os.makedirs(descriptorsPath, exist_ok=True)
os.makedirs(evaluationsPath, exist_ok=True)
os.makedirs(historyPath, exist_ok=True)
os.makedirs(modelsPath, exist_ok=True)
os.makedirs(visualsPath, exist_ok=True)


# Nothing else records what a run was: every knob is an env var and train.py is uncommitted, so a
# checkpoint on its own cannot say which alpha, bank or dataset produced it. One file per launch.
def _snapshotParams(outputPath):
	import json, sys, subprocess
	from datetime import datetime

	def _git(repo):
		try:
			sha = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
			status = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], stderr=subprocess.STDOUT)
			return { 'commit': sha.decode().strip(), 'dirty': bool(status.decode().strip()) }
		except Exception as error:
			return { 'error': str(error) }

	# isinstance(dict) also keeps this function, whose own name ends in Params, out of its dump
	snapshot = dict((name, value) for (name, value) in globals().items()
					if name.endswith('Params') and isinstance(value, dict))
	snapshot['experimentName'] = experimentName
	snapshot['trainingConfig'] = trainingConfig
	snapshot['startedAt'] = datetime.now().isoformat()
	snapshot['argv'] = sys.argv
	# only CL_* is ours; the rest of the environment is noise and may hold credentials
	snapshot['env'] = dict((k, v) for (k, v) in _os.environ.items() if k.startswith('CL_'))
	snapshot['git'] = { 'crosslocate': _git(os.path.dirname(os.path.abspath(__file__))),
						'farfield': _git('/home/aneesa/src/farfield') }

	stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	fn = os.path.join(outputPath, 'params_{}.json'.format(stamp))
	with open(fn, 'w') as f:
		# default=str keeps enums (Pooling.MAC) and tuples from killing the dump
		json.dump(snapshot, f, indent=2, sort_keys=True, default=str)
	print('[{}] wrote run parameters to {}'.format(tag, fn))

_snapshotParams(fullOutputPath)


# GPU selection
from core.Environment import Environment
print('----- GPU -----')

# seizeGPU() demands a GPU with NO processes at all, which a workstation running Xorg on the
# card can never satisfy (Xorg is a type-G process, not a compute one). CL_GPU names the device.
if (_os.environ.get('CL_GPU') is not None):
	gpuID = _os.environ['CL_GPU']
	_os.environ['CUDA_VISIBLE_DEVICES'] = gpuID
else:
	gpuID = Environment.seizeGPU()
	if (gpuID == -1):
		print('[{}] No GPU, exit'.format(tag))
		exit(1)
print('[{}] Seized GPU: {}'.format(tag, gpuID))

print('')


# Environment info
print('----- Environment -----')

Environment.showInfo()

print('')


# Configuration
print('----- Config -----')
# Do not allocate all of the memory

config = tf.ConfigProto()
# allow_growth and per_process_gpu_memory_fraction are mutually exclusive in TF1: growth
# wins and the fraction is ignored, so BFC grabs the whole card and starves a second run.
config.gpu_options.allow_growth = (float(_os.environ.get('CL_GPU_FRACTION', 1.0)) >= 1.0)

# CL_GPU_FRACTION caps this process so two runs share one card (the model needs 8.6 GB fp32 /
# 6.8 GB AMP, measured, so 24 GB holds two).
gpu_memory_usage = float(_os.environ.get('CL_GPU_FRACTION', 1.0))
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage	# TODO: check

print('')


# Local experiment directory
if (techParams['moveToSSD']):
	userSSDPath = '/mnt/ssd/' + 'CrossLocate'
	try:
		os.mkdir(userSSDPath, mode=0o700)
	except FileExistsError:
		pass

	experimentSSDPath = os.path.join(userSSDPath, experimentName)
	os.mkdir(experimentSSDPath)


# Datasets
from core.Dataset import Dataset
print('----- Datasets -----')

datasetsPath = '../datasets'

fullDatasetPath = os.path.join(datasetsPath, datasetParams['name'])
databasePath = os.path.join(fullDatasetPath, datasetParams['database'])
queriesPath = os.path.join(fullDatasetPath, datasetParams['queries'])

# Move dataset to local experiment directory
if (techParams['moveToSSD']):
	print('Moving dataset to SSD ...')
	databasePath, queriesPath = Environment.moveDataset([databasePath, queriesPath], experimentSSDPath)
	print('')

trainDataset = Dataset(datasetParams['name'], datasetParams['meta'], 'train', databasePath, queriesPath, positiveLocationThreshold=mineParams['positiveLocationThreshold'], positiveYawThreshold=mineParams['positiveYawThreshold'], negativeLocationMinThreshold=mineParams['negativeLocationMinThreshold'], negativeYawThreshold=mineParams.get('negativeYawThreshold'))
trainDataset.loadFromMetaStructs()
# UTM for farfield (small basin, WGS great-circle overkill); released path keeps its WGS behaviour
trainDataset.precomputeLocationDistancesUTM() if (trainingConfig == 'farfield') else trainDataset.precomputeLocationDistancesWGS()
trainDataset.findPositiveDbIDs()
trainDataset.findNonNegativeDbIDs()

print('')

valDataset = Dataset(datasetParams['name'], datasetParams['meta'], 'val', databasePath, queriesPath, positiveLocationThreshold=mineParams['positiveLocationThreshold'], positiveYawThreshold=mineParams['positiveYawThreshold'], negativeLocationMinThreshold=mineParams['negativeLocationMinThreshold'], negativeYawThreshold=mineParams.get('negativeYawThreshold'))
valDataset.loadFromMetaStructs()
valDataset.precomputeLocationDistancesUTM() if (trainingConfig == 'farfield') else valDataset.precomputeLocationDistancesWGS()
valDataset.findPositiveDbIDs()
valDataset.findNonNegativeDbIDs()

print('')


# Tensor assembler
from core.TensorAssembler import TensorAssembler
print('----- Tensor assembler -----')

tensorAssembler = TensorAssembler(datasetParams['meta'], databasePath, queriesPath, netParams['inputDimensions'], 3, trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'])

fullDepthDatasetPath = os.path.join(datasetsPath, datasetDepth['name'])
fullSilhouetteDatasetPath = os.path.join(datasetsPath, datasetSilhouettes['name'])

if (trainParams['databaseTensorFormat'] is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
	depthDatabasePath = os.path.join(fullDepthDatasetPath, datasetDepth['database'])
	silhouetteDatabasePath = os.path.join(fullSilhouetteDatasetPath, datasetSilhouettes['database'])

	if (techParams['moveToSSD']):
		depthDatabasePath, silhouetteDatabasePath = Environment.moveDataset([depthDatabasePath, silhouetteDatabasePath], experimentSSDPath)

	tensorAssembler.setDepthDatabasePath(depthDatabasePath)
	tensorAssembler.setSilhouetteDatabasePath(silhouetteDatabasePath)

if (trainParams['queryTensorFormat'] is TensorFormat.INPUT_GRAY_DEPTH_1CH_SILH_1CH):
	depthQueriesPath = os.path.join(fullDepthDatasetPath, datasetDepth['queries'])
	silhouetteQueriesPath = os.path.join(fullSilhouetteDatasetPath, datasetSilhouettes['queries'])

	if (techParams['moveToSSD']):
		depthQueriesPath, silhouetteQueriesPath = Environment.moveDataset([depthQueriesPath, silhouetteQueriesPath], experimentSSDPath)

	tensorAssembler.setDepthQueriesPath(depthQueriesPath)
	tensorAssembler.setSilhouetteQueriesPath(silhouetteQueriesPath)

# Set path to original queries
if (trainParams['queryTensorFormat'] is TensorFormat.ORIG_GRAY_INPUT_GRAY_INPUT_GRAY):
	fullOriginalDatasetPath = os.path.join(datasetsPath, datasetOriginalV1['name'])
	originalQueriesPath = os.path.join(fullOriginalDatasetPath, datasetOriginalV1['queries'])

	tensorAssembler.setOriginalQueriesPath(originalQueriesPath)

print('')


# Network building
from core.NetworkBuilder import NetworkBuilder
print('----- Network -----')

networkBuilder = NetworkBuilder(netParams['inputDimensions'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['metric'], learnParams['optimizer'])

graph, model, trainLoss, learningRate, optimizer, initializer, marginInput = networkBuilder.build(netParams['branching'], netParams['architecture'], netParams['preFeatNorm'], netParams['pooling'], netParams['postFeatNorm'], trainParams['margin'], adaptiveMargin=bankParams['adaptiveMargin'])

if (netParams['branching'] is Branching.ONE_BRANCH):
	dbModel = model
	qModel = model
elif (netParams['branching'] is Branching.TWO_BRANCH):
	dbModel = model[1]
	qModel = model[0]

print('')


# Descriptor extractors
from core.DescriptorExtractor import DescriptorExtractor
print('----- Descriptor extractors -----')

trainingDescriptorExtractor = DescriptorExtractor(trainDataset, tensorAssembler, descriptorsPath, netParams['inputDimensions'])
validationDescriptorExtractor = DescriptorExtractor(valDataset, tensorAssembler, descriptorsPath, netParams['inputDimensions'])

print('')


# Augmentator
from core.Augmentator import Augmentator
print('----- Augmentator -----')

dbAugmentator = Augmentator()
qAugmentator = Augmentator()

if (augmentParams['augmentation']):
	dbAugmentator.activateShiftAugmentation(maxHorizontalPercentage=0.05, maxVerticalPercentage=0.025)
	dbAugmentator.activateRotationAugmentation(maxAngle=5)
	dbAugmentator.activateFlipAugmentation()

	qAugmentator.activateShiftAugmentation(maxHorizontalPercentage=0.10, maxVerticalPercentage=0.050)
	qAugmentator.activateRotationAugmentation(maxAngle=5)
	qAugmentator.activateFlipAugmentation()
	qAugmentator.activateBrightnessAugmentation(maxMagnitude=40)

	qAugmentator.activateHueAugmentation(maxMagnitude=10)
	qAugmentator.activateSaturationAugmentation(maxMagnitude=50)
	qAugmentator.activateContrastAugmentation(maxPercentage=0.4)

	qAugmentator.activateBlurAugmentation(maxSigmaPercentage=0.004)
	qAugmentator.activateNoiseAugmentation(std=20)

print('')


# Triplet generators
from core.TripletGenerator import TripletGenerator
print('----- Triplet generators -----')

generatorParams = {	'branching': netParams['branching'],
				   	'dimensions': netParams['inputDimensions'],
		            'batchSize': trainParams['batchSize'],
		            'channels': 3,
		            'margin': mineParams['margin'],
				   	'numberOfNegatives': mineParams['numberOfNegatives'],
				   	'negativeSampleAttempts': mineParams['negativeSampleAttempts'],
				   	'numberOfSampledNegatives': mineParams['numberOfSampledNegatives'],
                    'reusingNegatives': mineParams['reusingNegatives']}

# One bank per Dataset: the c/q resolution is against that split's own dbIDs/qIDs.
if (bankParams['useBands']):
	from core.SimilarityBank import SimilarityBank
	trainBank = SimilarityBank(bankParams['bankPath'], trainDataset.dbIDs, trainDataset.qIDs,
							   bankParams['positiveFloor'], bankParams['bandEdges'],
							   loadWarps=bankParams['loadWarps'], tag='SimilarityBank (train)')
	# The val generator only runs under CL_VAL_LOSS, but its margins must come from the same
	# bands or the adaptive graph would be fed a fixed-margin batch.
	valBank = SimilarityBank(bankParams['bankPath'], valDataset.dbIDs, valDataset.qIDs,
							 bankParams['positiveFloor'], bankParams['bandEdges'],
							 loadWarps=bankParams['loadWarps'], tag='SimilarityBank (val)') if valLossEnabled else None
else:
	trainBank = valBank = None

valMiningStrategy = MiningStrategy.SIMILARITY_BANDS if (valBank is not None) else MiningStrategy.NONE

trainingGenerator = TripletGenerator(trainDataset, tensorAssembler, dbAugmentator, qAugmentator, **generatorParams, numberOfTripletReusages=mineParams['numberOfTrainTripletReusages'], miningStrategy=mineParams['trainMiningStrategy'], shuffle=True, similarityBank=trainBank, bankParams=bankParams)
validationGenerator = TripletGenerator(valDataset, tensorAssembler, None, None, **generatorParams, numberOfTripletReusages=1, miningStrategy=valMiningStrategy, shuffle=False, similarityBank=valBank, bankParams=bankParams)

print('')


# Objects
from core.Evaluator import Evaluator
from core.Historian import Historian
from core.Plotter import Plotter
from core.TripletVisualizer import TripletVisualizer

from core.TripletVisualizer import VisualMode

# Evaluators
trainEvaluator = Evaluator(trainDataset, evalParams['recallNumbers'], evalParams['locationThresholds'], evalParams['locationThresholdStep'], yawThreshold=evalParams['yawThreshold'], successfulLocalizationThreshold=evalParams['successfulLocalizationThreshold'], outputPath=evaluationsPath)
valEvaluator = Evaluator(valDataset, evalParams['recallNumbers'], evalParams['locationThresholds'], evalParams['locationThresholdStep'], yawThreshold=evalParams['yawThreshold'], successfulLocalizationThreshold=evalParams['successfulLocalizationThreshold'], outputPath=evaluationsPath)

# Historian
historian = Historian(evalParams['recallNumbers'], evalParams['locationThresholds'], historyPath)

# Plotter
plotter = Plotter(evalParams['recallNumbers'], plotParams['distanceRecallsXTicks'], chartsPath)

# Batch viewer
batchVisualizer = TripletVisualizer(VisualMode.BATCH, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)

trainLocVisualizer = TripletVisualizer(VisualMode.LOC, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)
trainLocVisualizer.setupForQueryLocalizationsVisualization(trainDataset, tensorAssembler, netParams['inputDimensions'], 3, evalParams['locationThresholds'])

valLocVisualizer = TripletVisualizer(VisualMode.LOC, datasetParams['meta'], trainParams['batchSize'], mineParams['numberOfNegatives'], trainParams['databaseTensorFormat'], trainParams['queryTensorFormat'], visualsPath)
valLocVisualizer.setupForQueryLocalizationsVisualization(valDataset, tensorAssembler, netParams['inputDimensions'], 3, evalParams['locationThresholds'])

print('')




# Preparation
print('----- Session -----')
with tf.Session(graph=graph, config=config) as sess:	# TODO: check
	print('')

	sess.run(initializer)								# TODO: check (but network seems to be pretrained)

	# 100 epochs of val recall are only useful if the epoch that peaked is still on disk
	saver = tf.train.Saver(max_to_keep=trainParams['endEpoch'] if (trainingConfig == 'farfield') else 2)

	if (trainParams['startEpoch'] == 1):
		if (netParams['pretrained'] == 'IMAGENET'):
			if (netParams['architecture'] is Architecture.ALEXNET_SIMPLE):
				raise Exception('ImageNet weights cannot be used for {} architecture!'.format(netParams['architecture'].name))
			elif (netParams['architecture'] is Architecture.ALEXNET_SPLIT):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					model.loadWeights('models/bvlc_alexnet.npy', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					dbModel.loadWeights('models/bvlc_alexnet.npy', sess)
					qModel.loadWeights('models/bvlc_alexnet.npy', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))

			# Load weights
			if (netParams['architecture'] is Architecture.VGG16):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					model.loadWeights('models/vgg16_weights.npz', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					dbModel.loadWeights('models/vgg16_weights.npz', sess)
					qModel.loadWeights('models/vgg16_weights.npz', sess)
					print('Loaded ImageNet weights for {} {}'.format(netParams['branching'].name, netParams['architecture'].name))

		elif (netParams['pretrained']):
			if (os.sep in netParams['pretrained']):
				# Explicit checkpoint prefix: trainables only, since released checkpoints have no Adam slots
				with graph.as_default():
					warmSaver = tf.train.Saver(var_list=tf.trainable_variables())
				warmSaver.restore(sess, netParams['pretrained'])
				print('Warm-started from checkpoint', netParams['pretrained'])
			else:
				pretrainedModelsPath = os.path.join('output', netParams['pretrained'], outputDirectories['models'])
				saver.restore(sess, tf.train.latest_checkpoint(pretrainedModelsPath))
				print('Loaded latest model from', pretrainedModelsPath)

	else:
		# Load graph and latest model
		#modelFn = experimentName+'-model-'+str(trainParams['startEpoch']-1)
		#saver = tf.train.import_meta_graph(os.path.join(modelsPath, modelFn+'.meta'))
		saver.restore(sess, tf.train.latest_checkpoint(modelsPath))
		print('Restored latest model from', modelsPath)

		# An epoch that crashed before historian.write() leaves nothing to load; the weights still
		# resume, the recall history just restarts at startEpoch.
		if (os.path.exists(os.path.join(historyPath, 'trainEpochLossHistory.npy'))):
			historian.load()
		else:
			print('No history to load - resuming with an empty history')

	fileWriter = tf.summary.FileWriter(os.path.join('tensorboard', experimentName), sess.graph)	# TODO: check

	print('')


	# Initial descriptor extraction
	print('----- Initial descriptor extraction -----')
	print('')

	# Defined here, not at the epoch loop: the initial extraction below has to honour it too, or
	# a shared database is walked twice (274k renders, ~20 min) before epoch 1 for nothing.
	sharedDatabase = (list(trainDataset.getDatabaseImageFns()) == list(valDataset.getDatabaseImageFns()))
	if (sharedDatabase):
		print('[{}] train and val share one database — extracting its descriptors once per pass'.format(tag))

	trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.loadDatasetDescriptors()
	if (trainDbDescriptors is None) or (trainQDescriptors is None):
		print('')
		print('Precomputing training descriptors ...')
		# A seeded db cache (same weights, same renders) saves a ~38 min pass; the query side is
		# under a minute, so extract only the side that is actually missing.
		if (trainDbDescriptors is not None):
			trainQDescriptors = trainingDescriptorExtractor.extractQueryDescriptors(sess, qModel=qModel)
		else:
			trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

	print('')

	valDbDescriptors, valQDescriptors = validationDescriptorExtractor.loadDatasetDescriptors()
	if (valDbDescriptors is None) or (valQDescriptors is None):
		print('')
		print('Precomputing validation descriptors ...')
		if (sharedDatabase):
			valQDescriptors = validationDescriptorExtractor.extractQueryDescriptors(sess, qModel=qModel)
			valDbDescriptors = trainDbDescriptors
		else:
			valDbDescriptors, valQDescriptors = validationDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

	trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)
	validationGenerator.setDescriptors(valDbDescriptors, valQDescriptors)

	print('')


	# Initial evaluation
	print('----- Initial recalls -----')
	print('')

	trainEvaluator.setDescriptors(trainDbDescriptors, trainQDescriptors)
	valEvaluator.setDescriptors(valDbDescriptors, valQDescriptors)

	print('-- Chance recalls --')
	print('')
	trainChancePositionDistanceRecalls, trainChancePositionAndOrientationDistanceRecalls, trainChancePositionRecalls, trainChancePositionAndOrientationRecalls, _, _ = trainEvaluator.computeRecallsAtNs(chanceSolution=True, withPrint=True)
	print('')
	valChancePositionDistanceRecalls, valChancePositionAndOrientationDistanceRecalls, valChancePositionRecalls, valChancePositionAndOrientationRecalls, _, _ = valEvaluator.computeRecallsAtNs(chanceSolution=True, withPrint=True)
	print('')

	print('-- Real recalls --')
	print('')
	trainPositionDistanceRecalls, trainPositionAndOrientationDistanceRecalls, trainPositionRecalls, trainPositionAndOrientationRecalls, _, _ = trainEvaluator.computeRecallsAtNs(withPrint=True)
	print('')
	valPositionDistanceRecalls, valPositionAndOrientationDistanceRecalls, valPositionRecalls, valPositionAndOrientationRecalls, _, _ = valEvaluator.computeRecallsAtNs(withPrint=True)
	print('')

	# Only the per-epoch recall series must not be appended twice on a resume; the chance and
	# distance recalls overwrite dataset-level constants, so they are stored either way.
	if (trainParams['startEpoch'] == 1):
		# Recalls
		historian.addTrainPositionRecalls(trainPositionRecalls)
		historian.addValPositionRecalls(valPositionRecalls)
		historian.addTrainPositionAndOrientationRecalls(trainPositionAndOrientationRecalls)
		historian.addValPositionAndOrientationRecalls(valPositionAndOrientationRecalls)

	# Chance recalls
	historian.storeTrainChancePositionRecalls(trainChancePositionRecalls)
	historian.storeValChancePositionRecalls(valChancePositionRecalls)
	historian.storeTrainChancePositionAndOrientationRecalls(trainChancePositionAndOrientationRecalls)
	historian.storeValChancePositionAndOrientationRecalls(valChancePositionAndOrientationRecalls)

	# Distance recalls
	historian.storeTrainPositionDistanceRecalls(trainPositionDistanceRecalls)
	historian.storeValPositionDistanceRecalls(valPositionDistanceRecalls)
	historian.storeTrainPositionAndOrientationDistanceRecalls(trainPositionAndOrientationDistanceRecalls)
	historian.storeValPositionAndOrientationDistanceRecalls(valPositionAndOrientationDistanceRecalls)

	# Chance distance recalls
	historian.storeTrainChancePositionDistanceRecalls(trainChancePositionDistanceRecalls)
	historian.storeValChancePositionDistanceRecalls(valChancePositionDistanceRecalls)
	historian.storeTrainChancePositionAndOrientationDistanceRecalls(trainChancePositionAndOrientationDistanceRecalls)
	historian.storeValChancePositionAndOrientationDistanceRecalls(valChancePositionAndOrientationDistanceRecalls)

	# Resuming a run whose first epoch died before historian.write() leaves nothing to plot yet.
	trainRecallsHistory = historian.getTrainPositionRecallsHistory()
	if (any(len(trainRecallsHistory[threshold]) for threshold in trainRecallsHistory)):
		plotter.plotRecalls(RecallType.POS, historian.getTrainPositionRecallsHistory(True), historian.getValPositionRecallsHistory(True), historian.getTrainChancePositionRecalls(), historian.getValChancePositionRecalls())
		plotter.plotRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationRecallsHistory(True), historian.getValPositionAndOrientationRecallsHistory(True), historian.getTrainChancePositionAndOrientationRecalls(), historian.getValChancePositionAndOrientationRecalls())

		plotter.plotDistanceRecalls(RecallType.POS, historian.getTrainPositionDistanceRecalls(), historian.getValPositionDistanceRecalls(), historian.getTrainChancePositionDistanceRecalls(), historian.getValChancePositionDistanceRecalls())
		plotter.plotDistanceRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationDistanceRecalls(), historian.getValPositionAndOrientationDistanceRecalls(), historian.getTrainChancePositionAndOrientationDistanceRecalls(), historian.getValChancePositionAndOrientationDistanceRecalls())




	# Model training
	print('')
	print('-------------------- Training model --------------------')

	print('Train on {} samples, validate on {} samples (queries)'.format(len(trainDataset), len(valDataset)))
	print('')

	numberOfTrainBatches = len(trainingGenerator)
	numberOfValBatches = len(validationGenerator)

	currentLearningRate = learnParams['learningRate']
	batchExtractFeatsInterval = floor(trainParams['queryExtractFeatsInterval']/trainParams['batchSize'])

	for epoch in range(trainParams['startEpoch'], trainParams['endEpoch']+1):

		print('')
		print('--------------------')
		print('Starting epoch {}'.format(epoch))
		print('--------------------')

		historian.onEpochStart(epoch)
		print('')

		# Learning rate adjustment
		if (epoch in learnParams['learningRateDownscaleEpochs']):
			print('Changing learning rate')
			index = learnParams['learningRateDownscaleEpochs'].index(epoch)
			downscaleFactor = learnParams['learningRateDownscaleFactors'][index]
			currentLearningRate = currentLearningRate / downscaleFactor
		print('Learning rate {} ({:.20f})'.format(currentLearningRate, currentLearningRate).rstrip('0'))
		print('')

		# Prepare directories for batch visualization
		if (epoch in visualParams['batchVisualEpochs']):
			trainBatchVisualDirName = 'train_batch_epoch{}'.format(epoch)
			valBatchVisualDirName = 'val_batch_epoch{}'.format(epoch)
			os.makedirs(os.path.join(visualsPath, trainBatchVisualDirName), exist_ok=True)
			os.makedirs(os.path.join(visualsPath, valBatchVisualDirName), exist_ok=True)




		# Training phase
		print('')
		print('----- Training phase -----')
		print('')

		trainSummary = tf.summary.scalar('train_batch_loss_epoch{}'.format(epoch), trainLoss)	# TODO: check

		trainLossAcc = 0

		queryRefreshWindowEnd = None

		for batch in range(1, numberOfTrainBatches+1):
			print('Processing training batch {} out of {}'.format(batch, numberOfTrainBatches))

			tripletBatch, marginBatch = trainingGenerator.getBatch()		# TODO: batch index

			if (tripletBatch is not None):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					feedDict = {model.input: tripletBatch, learningRate: currentLearningRate}
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					qBatch, dbBatch = tripletBatch
					feedDict = {qModel.input: qBatch, dbModel.input: dbBatch, learningRate: currentLearningRate}

				if (marginInput is not None):
					if (marginBatch is None):
						raise RuntimeError('adaptiveMargin is set but trainingGenerator.getBatch() returned no marginBatch')
					feedDict[marginInput] = marginBatch

				_, trainBatchLoss, summary = sess.run([optimizer, trainLoss, trainSummary], feed_dict=feedDict)	# TODO: check
				print('Train (batch) loss:', trainBatchLoss)
				trainLossAcc += trainBatchLoss
				historian.addTrainBatchLoss(trainBatchLoss)
				fileWriter.add_summary(summary, batch)						# TODO: check, change counter
			else:
				print('')
				print('Premature end of training epoch')
				print('')
				break

			if (epoch in visualParams['batchVisualEpochs']) and (batch <= visualParams['batchVisualLimit']):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					imageBatch = tripletBatch
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					imageBatch = np.concatenate(tripletBatch)
				tripletFnsBatch = trainingGenerator.getFnsBatch()
				tripletLocDistancesBatch = trainingGenerator.getLocDistancesBatch()
				tripletYawDistancesBatch = trainingGenerator.getYawDistancesBatch()
				batchVisualizer.visualizeBatch(imageBatch, tripletFnsBatch, tripletLocDistancesBatch, tripletYawDistancesBatch, trainBatchVisualDirName, batch)

			# Descriptor-based mining is unused when the bank supplies negatives (band mining); skip the refresh.
			if (not bankParams['useBands']) and (batch % batchExtractFeatsInterval == 0):
				print('')
				print('----- Intermediate descriptor extraction -----')
				print('')

				# Mining reads one query descriptor per triplet, so only the queries drawn before the
				# next refresh are ever used; 3x nominal covers the triplets mining skips.
				if (queryRefreshWindowEnd is not None) and (trainingGenerator.tripletIndex > queryRefreshWindowEnd):
					print('[WARN] query refresh window exceeded by {} triplets - stale descriptors were mined'.format(
						trainingGenerator.tripletIndex - queryRefreshWindowEnd))

				queryRefreshWindow = 3 * batchExtractFeatsInterval * trainParams['batchSize']
				upcomingQueryIDs = trainingGenerator.upcomingQueryIDs(queryRefreshWindow)
				queryRefreshWindowEnd = trainingGenerator.tripletIndex + queryRefreshWindow

				trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptorsPartialQueries(
					sess, dbModel=dbModel, qModel=qModel, queryIDs=upcomingQueryIDs, qDescriptors=trainQDescriptors)

				trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)

			print('')

		trainEpochLoss = trainLossAcc / batch		# TODO: check
		trainEpochLossSummary = tf.Summary(value=[ tf.Summary.Value(tag='train_epoch_loss', simple_value=trainEpochLoss) ])
		historian.addTrainEpochLoss(trainEpochLoss)
		fileWriter.add_summary(trainEpochLossSummary, epoch)

		trainingGenerator.onEpochEnd()


		# Save network
		saver.save(sess, os.path.join(modelsPath, experimentName+'-model'), global_step=epoch, write_meta_graph=True)
		print('Saved network for epoch', epoch)
		print('')




		# Epoch descriptor extraction
		print('----- Epoch descriptor extraction -----')
		print('')

		trainDbDescriptors, trainQDescriptors = None, None
		valDbDescriptors, valQDescriptors = None, None

		trainDbDescriptors, trainQDescriptors = trainingDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)
		if (sharedDatabase):
			# Only when db_per_split is off; otherwise val has its own db and takes the else branch.
			valQDescriptors = validationDescriptorExtractor.extractQueryDescriptors(sess, qModel=qModel)
			valDbDescriptors = trainDbDescriptors
		else:
			valDbDescriptors, valQDescriptors = validationDescriptorExtractor.extractDatasetDescriptors(sess, dbModel=dbModel, qModel=qModel)

		trainingGenerator.setDescriptors(trainDbDescriptors, trainQDescriptors)
		validationGenerator.setDescriptors(valDbDescriptors, valQDescriptors)

		print('')




		# Validation phase
		print('')
		print('---- Validation phase ----')
		print('')

		valSummary = tf.summary.scalar('val_batch_loss_epoch{}'.format(epoch), trainLoss)

		valLossAcc = 0
		batch = 0

		for batch in range(1, (numberOfValBatches+1) if valLossEnabled else 1):
			print('Processing validation batch {} out of {}'.format(batch, numberOfValBatches))

			tripletBatch, marginBatch = validationGenerator.getBatch()		# TODO: batch index

			if (tripletBatch is not None):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					feedDict = {model.input: tripletBatch}
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					qBatch, dbBatch = tripletBatch
					feedDict = {qModel.input: qBatch, dbModel.input: dbBatch}

				if (marginInput is not None):
					if (marginBatch is None):
						raise RuntimeError('adaptiveMargin is set but validationGenerator.getBatch() returned no marginBatch')
					feedDict[marginInput] = marginBatch

				valBatchLoss, summary = sess.run([trainLoss, valSummary], feed_dict=feedDict)
				print('Val (batch) loss', valBatchLoss)
				valLossAcc += valBatchLoss
				historian.addValBatchLoss(valBatchLoss)
				fileWriter.add_summary(summary, batch)
			else:
				print('')
				print('Premature end of validation epoch')
				print('')
				break

			if (epoch in visualParams['batchVisualEpochs']) and (batch <= visualParams['batchVisualLimit']):
				if (netParams['branching'] is Branching.ONE_BRANCH):
					imageBatch = tripletBatch
				elif (netParams['branching'] is Branching.TWO_BRANCH):
					imageBatch = np.concatenate(tripletBatch)
				tripletFnsBatch = validationGenerator.getFnsBatch()
				tripletLocDistancesBatch = validationGenerator.getLocDistancesBatch()
				tripletYawDistancesBatch = validationGenerator.getYawDistancesBatch()
				batchVisualizer.visualizeBatch(imageBatch, tripletFnsBatch, tripletLocDistancesBatch, tripletYawDistancesBatch, valBatchVisualDirName, batch)

			print('')

		valEpochLoss = (valLossAcc / batch) if batch else float('nan')
		valEpochLossSummary = tf.Summary(value=[ tf.Summary.Value(tag='val_epoch_loss', simple_value=valEpochLoss) ])
		historian.addValEpochLoss(valEpochLoss)
		fileWriter.add_summary(valEpochLossSummary, epoch)

		validationGenerator.onEpochEnd()




		# Epoch evaluation
		print('----- Epoch {} recalls -----'.format(epoch))
		print('')

		trainEvaluator.setDescriptors(trainDbDescriptors, trainQDescriptors)
		valEvaluator.setDescriptors(valDbDescriptors, valQDescriptors)

		trainPositionDistanceRecalls, trainPositionAndOrientationDistanceRecalls, trainPositionRecalls, trainPositionAndOrientationRecalls, trainQuerySuccesses, trainQueryFails = trainEvaluator.computeRecallsAtNs(withPrint=True)
		print('')
		valPositionDistanceRecalls, valPositionAndOrientationDistanceRecalls, valPositionRecalls, valPositionAndOrientationRecalls, valQuerySuccesses, valQueryFails = valEvaluator.computeRecallsAtNs(withPrint=True)
		print('')

		# Recalls
		historian.addTrainPositionRecalls(trainPositionRecalls)
		historian.addValPositionRecalls(valPositionRecalls)
		historian.addTrainPositionAndOrientationRecalls(trainPositionAndOrientationRecalls)
		historian.addValPositionAndOrientationRecalls(valPositionAndOrientationRecalls)

		# Distance recalls
		historian.storeTrainPositionDistanceRecalls(trainPositionDistanceRecalls)
		historian.storeValPositionDistanceRecalls(valPositionDistanceRecalls)
		historian.storeTrainPositionAndOrientationDistanceRecalls(trainPositionAndOrientationDistanceRecalls)
		historian.storeValPositionAndOrientationDistanceRecalls(valPositionAndOrientationDistanceRecalls)


		# Plots
		plotter.plotTrainBatchLoss(historian.getTrainBatchLossHistory(), epoch)
		plotter.plotValBatchLoss(historian.getValBatchLossHistory(), epoch)

		plotter.plotTrainEpochLoss(historian.getTrainEpochLossHistory())
		plotter.plotValEpochLoss(historian.getValEpochLossHistory())

		plotter.plotEpochLossComparison([historian.getTrainEpochLossHistory(), historian.getValEpochLossHistory()])

		plotter.plotRecalls(RecallType.POS, historian.getTrainPositionRecallsHistory(True), historian.getValPositionRecallsHistory(True), historian.getTrainChancePositionRecalls(), historian.getValChancePositionRecalls())
		plotter.plotRecalls(RecallType.POS_AND_ORIENT, historian.getTrainPositionAndOrientationRecallsHistory(True), historian.getValPositionAndOrientationRecallsHistory(True), historian.getTrainChancePositionAndOrientationRecalls(), historian.getValChancePositionAndOrientationRecalls())

		plotter.plotDistanceRecalls(RecallType.POS, historian.getTrainPositionDistanceRecalls(), historian.getValPositionDistanceRecalls(), historian.getTrainChancePositionDistanceRecalls(), historian.getValChancePositionDistanceRecalls())
		plotter.plotDistanceRecalls(RecallType.POS_AND_ORIENT,historian.getTrainPositionAndOrientationDistanceRecalls(), historian.getValPositionAndOrientationDistanceRecalls(), historian.getTrainChancePositionAndOrientationDistanceRecalls(), historian.getValChancePositionAndOrientationDistanceRecalls())


		if (epoch in visualParams['successesVisualEpochs']):
			trainSuccessesVisualDirName = 'train_successes_epoch{}'.format(epoch)
			valSuccessesVisualDirName = 'val_successes_epoch{}'.format(epoch)
			os.mkdir(os.path.join(visualsPath, trainSuccessesVisualDirName))
			os.mkdir(os.path.join(visualsPath, valSuccessesVisualDirName))

			trainLocVisualizer.visualizeQueryLocalizations(trainQuerySuccesses, trainSuccessesVisualDirName)
			valLocVisualizer.visualizeQueryLocalizations(valQuerySuccesses, valSuccessesVisualDirName)

		if (epoch in visualParams['failsVisualEpochs']):
			trainFailsVisualDirName = 'train_fails_epoch{}'.format(epoch)
			valFailsVisualDirName = 'val_fails_epoch{}'.format(epoch)
			os.mkdir(os.path.join(visualsPath, trainFailsVisualDirName))
			os.mkdir(os.path.join(visualsPath, valFailsVisualDirName))

			trainLocVisualizer.visualizeQueryLocalizations(trainQueryFails, trainFailsVisualDirName)
			valLocVisualizer.visualizeQueryLocalizations(valQueryFails, valFailsVisualDirName)


		# Epoch finish
		print('----- Finishing epoch {} -----'.format(epoch))
		print('')

		historian.onEpochEnd(epoch)
		historian.write()
		print('')

		# On-command end
		if (os.path.exists(os.path.join(fullOutputPath, 'end.command'))):
			print('Premature end of training with epoch', epoch)
			print('')
			break

	# Show training summary
	historian.showSummary()

	fileWriter.close()

	# Remove local experiment directory
	if (techParams['moveToSSD']):
		rmtree(experimentSSDPath)

	print('')
	print('Training successfully finished')
