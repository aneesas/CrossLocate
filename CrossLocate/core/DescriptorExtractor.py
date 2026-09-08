#!/usr/bin/python3
"""
	Descriptor Extractor

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import os
import numpy as np
from math import ceil, log10
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class DescriptorExtractor():
	"""
		Loads and extracts database and query descriptors from dataset images.

		Assumptions:
		Dataset images are in RGB format.
	"""

	def __init__(self, dataset, tensorAssembler, descriptorsPath, inputDimensions=None):
		self.tag = 'DescriptorExtractor ({})'.format(dataset.getSet())

		self.dataset = dataset
		self.tensorAssembler = tensorAssembler
		self.descriptorsPath = descriptorsPath

		self.inputDimensions = inputDimensions

		self.dbDescriptorsFn = dataset.getSet() + 'DbDescriptors'
		self.qDescriptorsFn = dataset.getSet() + 'QDescriptors'

		self.extractBatchSize = 10
		# Image loading runs ~7x faster than the VGG16 forward pass, so prefetching batches off
		# the GPU thread is what takes extraction from ~25 to ~100 img/s.
		self.prefetchWorkers = 6
		self.prefetchDepth = 8

	def loadDatasetDescriptors(self):
		dbDescriptors = None
		qDescriptors = None

		print('[{}] Loading existing database descriptors ...'.format(self.tag))
		dbDescriptorsPath = os.path.join(self.descriptorsPath, self.dbDescriptorsFn+'.npy')
		if (os.path.exists(dbDescriptorsPath)):
			dbDescriptors = np.load(dbDescriptorsPath)
			print('[{}] Database descriptors successfully loaded from {}'.format(self.tag, dbDescriptorsPath))
		else:
			print('[{}] Could not load database descriptors'.format(self.tag))

		print('')

		print('[{}] Loading existing query descriptors ...'.format(self.tag))
		qDescriptorsPath = os.path.join(self.descriptorsPath, self.qDescriptorsFn+'.npy')
		if (os.path.exists(qDescriptorsPath)):
			qDescriptors = np.load(qDescriptorsPath)
			print('[{}] Query descriptors successfully loaded from {}'.format(self.tag, qDescriptorsPath))
		else:
			print('[{}] Could not load query descriptors'.format(self.tag))

		return dbDescriptors, qDescriptors

	def extractDatasetDescriptors(self, session, dbModel, qModel):
		if (self.inputDimensions is None):
			raise Exception('Input dimensions required!')

		startExtractTime = time.time()

		print('[{}] Extracting database descriptors ...'.format(self.tag))
		dbDescriptors = self.__extractDescriptors('db', session, dbModel, self.dataset.getDatabaseImageFns())
		np.save(os.path.join(self.descriptorsPath, self.dbDescriptorsFn), dbDescriptors)

		print('')

		print('[{}] Extracting query descriptors ...'.format(self.tag))
		qDescriptors = self.__extractDescriptors('q', session, qModel, self.dataset.getQueryImageFns())
		np.save(os.path.join(self.descriptorsPath, self.qDescriptorsFn), qDescriptors)

		print('')

		endExtractTime = time.time()
		self.__printExtractionTime(endExtractTime - startExtractTime)

		return dbDescriptors, qDescriptors

	def extractQueryDescriptors(self, session, qModel):
		"""Query side only, for when the db descriptors are already in hand."""
		if (self.inputDimensions is None):
			raise Exception('Input dimensions required!')

		print('[{}] Extracting query descriptors ...'.format(self.tag))
		qDescriptors = self.__extractDescriptors('q', session, qModel, self.dataset.getQueryImageFns())
		np.save(os.path.join(self.descriptorsPath, self.qDescriptorsFn), qDescriptors)

		return qDescriptors

	def extractDatasetDescriptorsPartialQueries(self, session, dbModel, qModel, queryIDs, qDescriptors):
		"""Full db, but only queryIDs on the query side, merged into qDescriptors in place."""
		if (self.inputDimensions is None):
			raise Exception('Input dimensions required!')

		startExtractTime = time.time()

		print('[{}] Extracting database descriptors ...'.format(self.tag))
		dbDescriptors = self.__extractDescriptors('db', session, dbModel, self.dataset.getDatabaseImageFns())
		np.save(os.path.join(self.descriptorsPath, self.dbDescriptorsFn), dbDescriptors)

		print('')

		allQueryFns = self.dataset.getQueryImageFns()
		print('[{}] Extracting {} of {} query descriptors ...'.format(self.tag, len(queryIDs), len(allQueryFns)))
		if (len(queryIDs) > 0):										# __extractDescriptors takes log10 of the count
			qDescriptors[queryIDs] = self.__extractDescriptors('q', session, qModel, [allQueryFns[i] for i in queryIDs])
			np.save(os.path.join(self.descriptorsPath, self.qDescriptorsFn), qDescriptors)

		print('')

		endExtractTime = time.time()
		self.__printExtractionTime(endExtractTime - startExtractTime)

		return dbDescriptors, qDescriptors

	def __extractDescriptors(self, type, session, model, imageFns):
		numberOfDescriptors = len(imageFns)
		descriptors = np.zeros(shape=(numberOfDescriptors, model.output.shape[1]), dtype=np.float32)

		numberOfExtractBatches = ceil(numberOfDescriptors / self.extractBatchSize)		# ceil to assure processing of all samples
		orderOfNumberOfDescriptors = int(log10(numberOfDescriptors))
		sampleProgressInterval = min(max(10**orderOfNumberOfDescriptors, 100), 10000)
		batchProgressInterval = ceil(sampleProgressInterval / self.extractBatchSize)

		with ThreadPoolExecutor(self.prefetchWorkers) as pool:
			pending = deque()

			def submit(batch):
				startBatchIndex = batch*self.extractBatchSize
				endBatchIndex = (batch+1)*self.extractBatchSize
				batchImageFns = imageFns[startBatchIndex : endBatchIndex]				# correctly takes only remaining samples in the final batch
				pending.append((startBatchIndex, pool.submit(self.__getImages, type, batchImageFns)))

			for batch in range(min(self.prefetchDepth, numberOfExtractBatches)):
				submit(batch)

			for batch in range(numberOfExtractBatches):
				if (batch % batchProgressInterval == 0):
					print('[{}] {} / {}'.format(self.tag, batch*self.extractBatchSize, numberOfDescriptors))

				nextBatch = batch + self.prefetchDepth
				if (nextBatch < numberOfExtractBatches):
					submit(nextBatch)

				startBatchIndex, future = pending.popleft()
				batchDescriptors = session.run(model.output, feed_dict={model.input: future.result()})

				descriptors[startBatchIndex : startBatchIndex+len(batchDescriptors)] = batchDescriptors

		print('[{}] Total: {}'.	format(self.tag, startBatchIndex+len(batchDescriptors)))

		return descriptors

	def __getImages(self, type, imageFns):
		if (type == 'db'):
			return [self.tensorAssembler.getDatabaseTensor(imageFn) for imageFn in imageFns]
		elif (type == 'q'):
			return [self.tensorAssembler.getQueryTensor(imageFn) for imageFn in imageFns]

	def __getDescriptors(self, type, session, model, imageFns):
		if (type == 'db'):
			images = [self.tensorAssembler.getDatabaseTensor(imageFn) for imageFn in imageFns]
		elif (type == 'q'):
			images = [self.tensorAssembler.getQueryTensor(imageFn) for imageFn in imageFns]

		descriptors = session.run(model.output, feed_dict={model.input: images})

		return descriptors

	def __printExtractionTime(self, extractionTime):
		hours = round(extractionTime // 3600)
		remaining = extractionTime - (hours * 3600)
		minutes = round(remaining // 60)
		print('[{}] Extraction took {}h {}min'.format(self.tag, hours, minutes))
