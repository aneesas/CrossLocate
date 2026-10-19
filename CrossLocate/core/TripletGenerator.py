#!/usr/bin/python3
"""
	Triplet Generator

	Created by Jan Tomesek on 24.10.2018.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2018, Jan Tomesek"


import numpy as np
from sklearn.neighbors import NearestNeighbors

from core.NetworkBuilder import Branching


from enum import Enum

class MiningStrategy(Enum):
	NONE = 0
	SEMIHARD_AND_HARD_NEGATIVE = 1
	SEMIHARD_NEGATIVE = 2
	SIMILARITY_BANDS = 3


class TripletGenerator():
	"""
		Generates triplet batches of query, positive and negative samples.
		Based on geographic and descriptor distances.
		Each sample is assembled based on specified tensor format.

		Assumptions:
		Descriptor distances (from NN) are already sorted (small to big).
	"""

	def __init__(self, dataset, tensorAssembler, dbAugmentator, qAugmentator, branching, batchSize, dimensions, channels, margin, numberOfNegatives,
				 negativeSampleAttempts, numberOfSampledNegatives, reusingNegatives, numberOfTripletReusages, miningStrategy,
				 shuffle, similarityBank=None, bankParams=None):
		self.tag = 'TripletGenerator ({})'.format(dataset.getSet())

		self.dataset = dataset
		self.tensorAssembler = tensorAssembler
		self.dbAugmentator = dbAugmentator
		self.qAugmentator = qAugmentator

		self.databasePath = self.dataset.getDatabasePath()
		self.queriesPath = self.dataset.getQueriesPath()

		self.branching = branching

		self.batchSize = batchSize
		self.dimensions = dimensions
		self.channels = channels

		self.margin = margin

		self.numberOfNegatives = numberOfNegatives

		self.negativeSampleAttempts = negativeSampleAttempts
		self.numberOfSampledNegatives = numberOfSampledNegatives

		self.reusingNegatives = reusingNegatives
		self.numberOfTripletReusages = numberOfTripletReusages
		self.miningStrategy = miningStrategy

		self.similarityBank = similarityBank
		self.bankParams = bankParams
		if (self.miningStrategy is MiningStrategy.SIMILARITY_BANDS):
			if (self.similarityBank is None) or (self.bankParams is None):
				raise ValueError('MiningStrategy.SIMILARITY_BANDS requires both similarityBank and bankParams')

			self.bankAlpha = self.bankParams['alpha']
			self.bankNegativesPerBand = self.bankParams['negativesPerBand']
			self.bankPositiveFloor = self.bankParams['positiveFloor']
			self.bankWarps = self.similarityBank.warps()	# cached: query/candidate warp lists + draw weights

			if (sum(self.bankNegativesPerBand) != numberOfNegatives):
				raise ValueError('bankParams.negativesPerBand sums to {} but numberOfNegatives is {}'.format(
					sum(self.bankNegativesPerBand), numberOfNegatives))

			self.bankSkipQueryIDs = set(self.similarityBank.queriesWithoutBand([1] + list(self.bankNegativesPerBand)))
			print('[{}] {} queries skipped (cannot fill positive/band counts)'.format(self.tag, len(self.bankSkipQueryIDs)))

		self.shuffle = shuffle  # shuffle sample order (even within batches)

		self.numberOfTriplets = len(self.dataset)
		self.tripletIndex = 0
		self.queryIndexes = np.arange(self.numberOfTriplets)
		self.currentTripletReusage = 0

		if (self.shuffle):
			np.random.shuffle(self.queryIndexes)

		self.numberOfPositivesNotFound = 0
		self.numberOfNegativesNotFound = 0
		self.numberOfBandShortfalls = [0, 0, 0]
		self.numberOfMarginClamps = 0
		self.numberOfBankSkips = 0
		self.numberOfWarpFallbacks = 0

		self.nnAlgorithm = 'ball_tree'
		self.nnMetric = 'euclidean'

		self.lastNegativeDbIDs = [[]] * len(self.dataset)

		#self.on_epoch_end()

	def onEpochEnd(self):
		self.tripletIndex = 0
		self.queryIndexes = np.arange(self.numberOfTriplets)
		self.currentTripletReusage = 0

		if (self.shuffle):
			np.random.shuffle(self.queryIndexes)

		self.numberOfPositivesNotFound = 0
		self.numberOfNegativesNotFound = 0
		self.numberOfBandShortfalls = [0, 0, 0]
		self.numberOfMarginClamps = 0
		self.numberOfBankSkips = 0
		self.numberOfWarpFallbacks = 0

	# Returns the number of batches (per epoch)
	# Keras method (equivalent to steps_per_epoch)
	def __len__(self):
		return int(np.floor(len(self.dataset)*self.numberOfTripletReusages / float(self.batchSize)))		# floor and float not necessary (but explicit)

	def setDescriptors(self, dbDescriptors, qDescriptors):
		self.dbDescriptors = dbDescriptors
		self.qDescriptors = qDescriptors

	def __updateLastNegativeDbIDs(self, queryID, negativeDbIDs):
		if (len(negativeDbIDs) == self.numberOfNegatives):
			# Replace last negative IDs with new negative IDs
			self.lastNegativeDbIDs[queryID] = negativeDbIDs
		else:
			# From last negative IDs keep only those IDs that are not in new negative IDs (duplicate negative IDs from new negative IDs have priority)
			uniqueLastNegativeDbIDs = [negDbID for negDbID in self.lastNegativeDbIDs[queryID] if negDbID not in negativeDbIDs]
			# Prepend new negative IDs to last negative IDs
			self.lastNegativeDbIDs[queryID] = negativeDbIDs + uniqueLastNegativeDbIDs
			# Keep only (newest) numberOfNegatives negativeIDs
			self.lastNegativeDbIDs[queryID] = self.lastNegativeDbIDs[queryID][0:self.numberOfNegatives]

	def __getLastNegativeDbIDs(self, queryID):
		return self.lastNegativeDbIDs[queryID]

	def upcomingQueryIDs(self, count):
		"""Query IDs the next count draws will mine; an upper bound, reusage consumes fewer."""
		return self.queryIndexes[self.tripletIndex : self.tripletIndex + count]

	def getBatch(self):
		inputBatch, targetBatch = self.__generateTripletBatch()
		return inputBatch, targetBatch

	def getFnsBatch(self):
		return self.tripletFnsBatch

	def getLocDistancesBatch(self):
		return self.tripletLocDistancesBatch

	def getYawDistancesBatch(self):
		return self.tripletYawDistancesBatch

	# TODO: create tensors possibly before WHILE to avoid allocation
	def __generateTripletBatch(self):
		queryBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
		positiveBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)
		negativeBatch = np.ndarray(shape=(0, *self.dimensions, self.channels), dtype=np.float32)

		queryFnsBatch = []
		positiveFnsBatch = []
		negativeFnsBatch = []

		queryLocDistancesBatch = []
		positiveLocDistancesBatch = []
		negativeLocDistancesBatch = []

		queryYawDistancesBatch = []
		positiveYawDistancesBatch = []
		negativeYawDistancesBatch = []

		marginBatch = np.ndarray(shape=(0, self.numberOfNegatives), dtype=np.float32)

		currentBatchSize = 0
		while (currentBatchSize < self.batchSize):

			if (self.tripletIndex == self.numberOfTriplets):
				print('')
				print('[{}] All {} possible triplets processed'.format(self.tag, self.numberOfTriplets))
				print('[{}] {} triplets skipped because of positive samples'.format(self.tag, self.numberOfPositivesNotFound))
				print('[{}] {} triplets skipped because of negative samples'.format(self.tag, self.numberOfNegativesNotFound))
				print('[{}] {} triplets skipped because query has no bank data'.format(self.tag, self.numberOfBankSkips))
				print('[{}] band shortfalls (top/mid/low): {}'.format(self.tag, self.numberOfBandShortfalls))
				print('[{}] {} margins clamped to 0'.format(self.tag, self.numberOfMarginClamps))
				print('[{}] {} warp cells fell back (invalid cell or missing pair)'.format(self.tag, self.numberOfWarpFallbacks))
				return None, None

			# Get query ID for current triplet
			queryID = self.queryIndexes[self.tripletIndex]

			if (self.currentTripletReusage == 0):
				# Get positive and negative database IDs for current triplet (selection only; bank
				# path's margin is None here, computed below per augmentation pass instead)
				positiveDbID, negativeDbIDs, _ = self.__getTripletDbIDs(queryID)

				# Go through triplets until successful
				if (positiveDbID is None) or (negativeDbIDs is None):
					print('[{}] Skipping triplet {} (ID {})'.format(self.tag, self.tripletIndex, queryID))
					self.tripletIndex += 1
					continue

				self.reusePositiveDbID = positiveDbID
				self.reuseNegativeDbIDs = negativeDbIDs
			else:
				positiveDbID = self.reusePositiveDbID
				negativeDbIDs = self.reuseNegativeDbIDs

			# Bank path: warp indices and the margin are redrawn every augmentation pass (including
			# reusages), since a margin fixed at selection time would describe the wrong warp on
			# reusage 2..n. queryWarp/positiveWarp/negativeWarps stay None otherwise -> augmentImage
			# falls back to its normal random augmentation, byte-identical to before this change.
			negativeMargins = None
			queryWarp = None
			positiveWarp = None
			negativeWarps = None
			if (self.miningStrategy is MiningStrategy.SIMILARITY_BANDS):
				negativeMargins, queryWarp, positiveWarp, negativeWarps = self.__drawWarpsAndMargins(
					queryID, positiveDbID, negativeDbIDs)

			# Create tensors for current triplet
			queryTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
			positiveTensor = np.ndarray(shape=(1, *self.dimensions, self.channels), dtype=np.float32)
			negativeTensor = np.ndarray(shape=(self.numberOfNegatives, *self.dimensions, self.channels), dtype=np.float32)

			# Fill query tensor
			queryFn = self.dataset.getQueryImageFn(queryID)
			if (self.currentTripletReusage == 0):
				queryTensor[0] = self.tensorAssembler.getQueryTensor(queryFn)
				# .copy(): the row is augmented in place below, so a view would make reusages compound
				self.reuseQueryTensor = queryTensor[0].copy()
			else:
				queryTensor[0] = self.reuseQueryTensor
			if (self.qAugmentator):
				queryTensor[0] = self.qAugmentator.augmentImage(queryTensor[0], warp=queryWarp)
				isFlipped = self.qAugmentator.wasFlipped()

			queryFnsBatch.append(queryFn)
			queryLocDistancesBatch.append(None)
			queryYawDistancesBatch.append(None)

			# Fill positive tensor
			positiveFn = self.dataset.getDatabaseImageFn(positiveDbID[0])
			if (self.currentTripletReusage == 0):
				positiveTensor[0] = self.tensorAssembler.getDatabaseTensor(positiveFn)
				self.reusePositiveTensor = positiveTensor[0].copy()
			else:
				positiveTensor[0] = self.reusePositiveTensor
			if (self.dbAugmentator):
				positiveTensor[0] = self.dbAugmentator.augmentImage(positiveTensor[0], applyFlip=isFlipped, disableFlip=(not isFlipped), warp=positiveWarp)

			positiveFnsBatch.append(positiveFn)
			positiveLocDistancesBatch += self.dataset.computeLocationDistances(queryID, positiveDbID, squared=False).tolist()
			positiveYawDistancesBatch += self.dataset.computeYawDistances(queryID, positiveDbID, inDegrees=True)

			# Fill negative tensor
			for negIndex, negativeDbID in enumerate(negativeDbIDs):
				negativeFn = self.dataset.getDatabaseImageFn(negativeDbID)
				if (self.currentTripletReusage == 0):
					negativeTensor[negIndex] = self.tensorAssembler.getDatabaseTensor(negativeFn)

				negativeFnsBatch.append(negativeFn)
			negativeLocDistancesBatch += self.dataset.computeLocationDistances(queryID, negativeDbIDs, squared=False).tolist()
			negativeYawDistancesBatch += self.dataset.computeYawDistances(queryID, negativeDbIDs, inDegrees=True)

			if (self.currentTripletReusage == 0):
				self.reuseNegativeTensor = negativeTensor.copy()
			else:
				negativeTensor[:] = self.reuseNegativeTensor

			# Build this triplet's margin row (bank path only), padding to match negativeTensor duplication below
			if (negativeMargins is not None):
				marginRow = np.zeros(shape=(self.numberOfNegatives,), dtype=np.float32)
				marginRow[0:len(negativeMargins)] = negativeMargins

			# Add remaining negatives by duplicating existing ones
			if (len(negativeDbIDs) < self.numberOfNegatives):
				remainingStartIndex = len(negativeDbIDs)
				for sampleIndex, remainingIndex in enumerate(range(remainingStartIndex, self.numberOfNegatives)):
					moduloSampleIndex = sampleIndex % remainingStartIndex
					negativeTensor[remainingIndex] = negativeTensor[moduloSampleIndex]
					if (negativeMargins is not None):
						marginRow[remainingIndex] = marginRow[moduloSampleIndex]
					if (negativeWarps is not None):
						# same warp as the copied image, so augmentImage reproduces it and the
						# duplicated margin (from the same source index, above) stays correct
						negativeWarps.append(negativeWarps[moduloSampleIndex])

					negativeFnsBatch.append('repeat')
					negativeLocDistancesBatch += [None]
					negativeYawDistancesBatch += [None]

			if (negativeMargins is not None):
				marginBatch = np.concatenate((marginBatch, marginRow.reshape(1, -1)))

			if (self.dbAugmentator):
				for negIndex in range(len(negativeTensor)):
					thisWarp = negativeWarps[negIndex] if (negativeWarps is not None) else None
					negativeTensor[negIndex] = self.dbAugmentator.augmentImage(negativeTensor[negIndex], applyFlip=isFlipped, disableFlip=(not isFlipped), warp=thisWarp)

			# Add current tensors to (sub)batches
			queryBatch = np.concatenate((queryBatch, queryTensor))
			positiveBatch = np.concatenate((positiveBatch, positiveTensor))
			negativeBatch = np.concatenate((negativeBatch, negativeTensor))

			self.currentTripletReusage += 1

			if (self.currentTripletReusage == self.numberOfTripletReusages):
				self.currentTripletReusage = 0
				self.tripletIndex += 1
			currentBatchSize += 1

		# Assemble triplet batch
		if (self.branching is Branching.ONE_BRANCH):
			tripletBatch = np.concatenate((queryBatch, positiveBatch, negativeBatch))
		elif (self.branching is Branching.TWO_BRANCH):
			tripletBatch = (queryBatch, np.concatenate((positiveBatch, negativeBatch)))

		self.tripletFnsBatch = queryFnsBatch + positiveFnsBatch + negativeFnsBatch
		self.tripletLocDistancesBatch = queryLocDistancesBatch + positiveLocDistancesBatch + negativeLocDistancesBatch
		self.tripletYawDistancesBatch = queryYawDistancesBatch + positiveYawDistancesBatch + negativeYawDistancesBatch

		if (self.miningStrategy is MiningStrategy.SIMILARITY_BANDS):
			return tripletBatch, marginBatch
		return tripletBatch, None

	def __getTripletDbIDs(self, queryID):
		if (self.miningStrategy is MiningStrategy.SIMILARITY_BANDS):
			return self.__getTripletDbIDsFromBank(queryID)

		# Get positive database IDs (by location and yaw distance)
		potentialPositiveDbIDs = self.dataset.getPositiveDbIDs(queryID=queryID)

		if not potentialPositiveDbIDs:
			print('[{}] No potential positive database IDs (return)'.format(self.tag))
			self.numberOfPositivesNotFound += 1
			return None, None, None

		# Collect violating negative database IDs across multiple sample attempts
		totalViolatingNegativeDbIDs = []
		negativeSampleAttempt = 0
		while ((len(totalViolatingNegativeDbIDs) < self.numberOfNegatives) and (negativeSampleAttempt < self.negativeSampleAttempts)):

			# Get negative database IDs (by location distance)
			negativeDbIDs = self.dataset.sampleNegativeDbIDsByLocation(queryID=queryID, n=self.numberOfSampledNegatives)

			# On the last attempt enrich current negative database IDs by negative database IDs from last epoch
			if ((self.reusingNegatives) and ((negativeSampleAttempt+1) == self.negativeSampleAttempts)):
				lastNegativeDbIDs = self.__getLastNegativeDbIDs(queryID)
				negativeDbIDs += lastNegativeDbIDs				# could introduce duplicate IDs (removed later in addUnique - removes duplicates even in new IDs)

			# Get descriptors corresponding to IDs
			queryDesc = self.qDescriptors[ [queryID] ]
			potentialPositiveDbDescs = self.dbDescriptors[ potentialPositiveDbIDs ]
			negativeDbDescs = self.dbDescriptors[ negativeDbIDs ]

			# Keep only positive and negative database IDs with smallest descriptor distances
			positiveDbID, posDistance = self.__filterPotentialPositiveDbIDs(potentialPositiveDbIDs, queryDesc, potentialPositiveDbDescs)
			negativeDbIDs, negDistances = self.__filterNegativeDbIDs(negativeDbIDs, queryDesc, negativeDbDescs)

			# Square the Euclidean descriptor distances
			posDistance = np.square(posDistance)
			negDistances = np.square(negDistances)

			if (self.miningStrategy is MiningStrategy.SEMIHARD_AND_HARD_NEGATIVE):
				# Keep only negative database IDs with descriptor distances less than positive distance + margin
				violatingNegativeDbIDs = [negativeDbID for negativeDbID, negativeDist in zip(negativeDbIDs, negDistances)
								if (negativeDist < posDistance[0]+self.margin)]
			elif (self.miningStrategy is MiningStrategy.SEMIHARD_NEGATIVE):
				violatingNegativeDbIDs = [negativeDbID for negativeDbID, negativeDist in zip(negativeDbIDs, negDistances)
										  if ((negativeDist < posDistance[0]+self.margin) and (negativeDist >= posDistance[0]))]
			elif (self.miningStrategy is MiningStrategy.NONE):
				violatingNegativeDbIDs = negativeDbIDs

			self.dataset._Dataset__addUnique(totalViolatingNegativeDbIDs, violatingNegativeDbIDs)
			negativeSampleAttempt += 1

		violatingNegativeDbIDs = totalViolatingNegativeDbIDs

		numberOfViolatingNegatives = len(violatingNegativeDbIDs)

		if (numberOfViolatingNegatives == 0):
			print('[{}] No violating negative database IDs (return)'.format(self.tag))
			self.numberOfNegativesNotFound += 1
			return None, None, None

		# Keep max numberOfNegatives violating negative database IDs
		if (numberOfViolatingNegatives > self.numberOfNegatives):
			violatingNegativeDbIDs = violatingNegativeDbIDs[0:self.numberOfNegatives]

		self.__updateLastNegativeDbIDs(queryID, violatingNegativeDbIDs)

		return positiveDbID, violatingNegativeDbIDs, None

	def __getTripletDbIDsFromBank(self, queryID):
		if (queryID in self.bankSkipQueryIDs):
			self.numberOfBankSkips += 1
			return None, None, None

		posDbIDs, posScores = self.similarityBank.positivePool(queryID)
		sampleIndex = np.random.randint(len(posDbIDs))
		positiveDbID = np.array([posDbIDs[sampleIndex]])
		simPos = posScores[sampleIndex]

		# pools[b] = (remaining dbIDs, remaining scores) for band b, shrinking as bands are drawn from
		pools = {b: list(self.similarityBank.negativeBand(queryID, b)[0]) for b in (0, 1, 2)}
		poolScores = {b: list(self.similarityBank.negativeBand(queryID, b)[1]) for b in (0, 1, 2)}

		def drawFrom(band, count):
			available = len(pools[band])
			take = min(count, available)
			pickIndexes = np.random.choice(available, size=take, replace=False) if take else np.array([], dtype=int)
			pickIndexes = sorted(pickIndexes, reverse=True)		# pop highest index first
			dbIDs, scores = [], []
			for i in pickIndexes:
				dbIDs.append(pools[band].pop(i))
				scores.append(poolScores[band].pop(i))
			return dbIDs, scores

		# No cross-band borrowing: a donor would silently change the score range the config asked
		# for. A short band stays short and the shortfall counter is the alarm.
		negativeDbIDs = []
		negativeSimNeg = []
		for band in (0, 1, 2):
			wanted = self.bankNegativesPerBand[band]
			dbIDs, scores = drawFrom(band, wanted)
			if (len(dbIDs) < wanted):
				self.numberOfBandShortfalls[band] += 1
			negativeDbIDs += dbIDs
			negativeSimNeg += scores

		# Unwarped fallback scores, kept for __drawWarpsAndMargins (run per augmentation pass, not here)
		self.reuseSimPos = simPos
		self.reuseNegativeSimNeg = np.array(negativeSimNeg, dtype=np.float32)

		return positiveDbID, negativeDbIDs, None

	# Draws this pass's warp indices and looks up the matching bank score for the query and every
	# database image, so the applied warp and the margin it grades are always the same bank cell.
	def __drawWarpsAndMargins(self, queryID, positiveDbID, negativeDbIDs):
		canWarp = (self.dbAugmentator is not None)		# no augmentator -> nothing to warp, use identity

		if (canWarp):
			qWarpIdx = np.random.choice(len(self.bankWarps['query_weight']), p=self.bankWarps['query_weight'])
		else:
			qWarpIdx = 0
		queryWarp = tuple(self.bankWarps['query'][qWarpIdx])

		def drawCandidateWarpIdx():
			if (canWarp):
				return np.random.choice(len(self.bankWarps['candidate_weight']), p=self.bankWarps['candidate_weight'])
			return 0

		cWarpIdxPos = drawCandidateWarpIdx()
		warpedPosScore, positiveWarp = self.__warpAndScore(queryID, int(positiveDbID[0]), qWarpIdx, cWarpIdxPos, self.reuseSimPos)

		negativeWarps = []
		warpedNegScores = []
		for negIndex, negativeDbID in enumerate(negativeDbIDs):
			cWarpIdx = drawCandidateWarpIdx()
			warpedNegScore, negativeWarp = self.__warpAndScore(queryID, int(negativeDbID), qWarpIdx, cWarpIdx, self.reuseNegativeSimNeg[negIndex])
			negativeWarps.append(negativeWarp)
			warpedNegScores.append(warpedNegScore)

		rawMargins = self.bankAlpha * (warpedPosScore - np.array(warpedNegScores, dtype=np.float32))
		self.numberOfMarginClamps += int(np.sum(rawMargins < 0))
		negativeMargins = np.maximum(0.0, rawMargins).astype(np.float32)

		return negativeMargins, queryWarp, positiveWarp, negativeWarps

	# Same qWarpIdx row for both sides of a margin, per the whole point of this class: warp = score.
	def __warpAndScore(self, queryID, dbID, qWarpIdx, cWarpIdx, unwarpedScore):
		identityWarp = tuple(self.bankWarps['candidate'][0])		# bank convention: index 0 is [0,0,0]

		cells = self.similarityBank.warpCells(queryID, dbID)
		if (cells is not None):
			scores, valid = cells
			if valid[qWarpIdx, cWarpIdx]:
				return float(scores[qWarpIdx, cWarpIdx]), tuple(self.bankWarps['candidate'][cWarpIdx])
			if valid[qWarpIdx, 0]:
				self.numberOfWarpFallbacks += 1
				return float(scores[qWarpIdx, 0]), identityWarp

		self.numberOfWarpFallbacks += 1
		return float(unwarpedScore), identityWarp

	# TODO: unite both filter function
	def __filterPotentialPositiveDbIDs(self, potentialPositiveDbIDs, queryDesc, potentialPositiveDbDescs):
		potentialPositive1NN = NearestNeighbors(n_neighbors=1, algorithm=self.nnAlgorithm, metric=self.nnMetric)
		potentialPositive1NN.fit(potentialPositiveDbDescs)

		posDistance, posIndex = potentialPositive1NN.kneighbors(queryDesc)	# [[distance]], [[index]]

		potentialPositiveDbIDs = np.array(potentialPositiveDbIDs)
		positiveDbID = potentialPositiveDbIDs[ posIndex[0] ]

		return positiveDbID, posDistance[0]									# [ID], [distance]

	def __filterNegativeDbIDs(self, negativeDbIDs, queryDesc, negativeDbDescs):
		negativeKNN = NearestNeighbors(n_neighbors=min(self.numberOfNegatives*10,len(negativeDbIDs)), algorithm=self.nnAlgorithm, metric=self.nnMetric)
		negativeKNN.fit(negativeDbDescs)

		negDistances, negIndices = negativeKNN.kneighbors(queryDesc)		# [[distances]], [[indices]]

		negativeDbIDs = np.array(negativeDbIDs)
		negativeDbIDs = negativeDbIDs[ negIndices[0] ]

		return negativeDbIDs, negDistances[0]								# [IDs], [distances]
