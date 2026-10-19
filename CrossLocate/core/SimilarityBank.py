#!/usr/bin/python3
"""
	SimilarityBank

	Wraps a precomputed render-vs-render overlap ("harmonic score") bank so it can be
	queried by CrossLocate's integer dbID/queryID convention instead of the bank's own
	string IDs.
"""

import os
import json
import glob

import numpy as np


N_BANDS = 3			# band 0 (top) .. band 2 (low), on top of the positive pool


class SimilarityBank():
	"""
		Positive pool + 3 negative bands per query, cut from the `harmonic` score at
		positiveFloor > bandEdges[0] > bandEdges[1]. Built once per Dataset.
	"""

	def __init__(self, bankPath, dbIDs, qIDs, positiveFloor, bandEdges, negativesPerBand=None, loadWarps=False, tag='SimilarityBank'):
		self.tag = tag

		if not (positiveFloor > bandEdges[0] > bandEdges[1] >= 0):
			raise ValueError('Invalid thresholds: require positiveFloor > bandEdges[0] > bandEdges[1] >= 0 (got {}, {})'.format(positiveFloor, bandEdges))

		self.bankPath = bankPath
		self.positiveFloor = positiveFloor
		self.bandEdges = list(bandEdges)
		self.negativesPerBand = negativesPerBand		# reserved for caller default budgets, unused here
		self.loadWarps = loadWarps

		self.dbIDs = np.asarray(dbIDs)
		self.qIDs = np.asarray(qIDs)
		self.numDb = len(self.dbIDs)
		self.numQueries = len(self.qIDs)

		self.__loadWarpMeta()
		cToDbIndex = self.__buildCToDbIndex()
		bankQToQueryIndex = self.__buildBankQToQueryIndex()

		self.__streamShards(cToDbIndex, bankQToQueryIndex)
		self.__buildBands()
		if self.loadWarps:
			self.__buildWarpIndex()

		self.__printSummary()

	# Construction helpers

	def __loadWarpMeta(self):
		bankJsonPath = os.path.join(self.bankPath, 'bank.json')
		with open(bankJsonPath, 'r') as f:
			self._bankJson = json.load(f)

		queriesJsonPath = os.path.join(self.bankPath, 'queries.json')
		if not os.path.exists(queriesJsonPath):
			raise FileNotFoundError('[{}] queries.json not found at {} - required to resolve bank query indices'.format(self.tag, queriesJsonPath))
		with open(queriesJsonPath, 'r') as f:
			self._bankQIDs = json.load(f)['qIDs']

		gridPath = os.path.join(self.bankPath, 'grid.npz')
		grid = np.load(gridPath)
		self.headings = grid['headings']
		self.nHead = len(self.headings)
		self.nPositions = grid['p_ll'].shape[0]

	# c = pi * nHead + hi (encode/decode are inverses), so this table is a full c -> dbID string map
	def __buildCToDbIndex(self):
		dbIndexOf = dict(zip(self.dbIDs.tolist(), range(self.numDb)))
		sizeC = self.nPositions * self.nHead
		table = np.full(sizeC, -1, dtype=np.int32)
		for c in range(sizeC):
			pi, hi = divmod(c, self.nHead)
			dbID = '{:07d}_{:03d}'.format(pi, int(round(self.headings[hi])))
			idx = dbIndexOf.get(dbID, -1)
			if idx >= 0:
				table[c] = idx
		return table

	def __buildBankQToQueryIndex(self):
		queryIndexOf = dict(zip(self.qIDs.tolist(), range(self.numQueries)))
		table = np.full(len(self._bankQIDs), -1, dtype=np.int32)
		for bankQ, qID in enumerate(self._bankQIDs):
			table[bankQ] = queryIndexOf.get(qID, -1)
		return table

	def __streamShards(self, cToDbIndex, bankQToQueryIndex):
		shardPaths = sorted(glob.glob(os.path.join(self.bankPath, 'labels', 'shard_*.npz')))

		qChunks, dbChunks, scoreChunks = [], [], []
		bankChunks, bankValidChunks = [], []

		self.rowsRead = 0
		self.droppedInvalid = 0
		self.droppedUnknownDb = 0
		self.droppedOutOfSplit = 0

		for shardPath in shardPaths:
			rows = np.load(shardPath)['rows']
			self.rowsRead += len(rows)

			valid = rows['valid']
			self.droppedInvalid += int((~valid).sum())
			rows = rows[valid]
			if len(rows) == 0:
				continue

			dbIdx = cToDbIndex[rows['c']]
			knownDb = dbIdx >= 0
			self.droppedUnknownDb += int((~knownDb).sum())
			rows = rows[knownDb]
			dbIdx = dbIdx[knownDb]
			if len(rows) == 0:
				continue

			qIdx = bankQToQueryIndex[rows['q']]
			inSplit = qIdx >= 0
			self.droppedOutOfSplit += int((~inSplit).sum())
			rows = rows[inSplit]
			qIdx = qIdx[inSplit]
			dbIdx = dbIdx[inSplit]
			if len(rows) == 0:
				continue

			qChunks.append(qIdx)
			dbChunks.append(dbIdx)
			scoreChunks.append(rows['harmonic'].astype(np.float32))
			if self.loadWarps:
				bankChunks.append(rows['bank'])
				bankValidChunks.append(rows['bank_valid'])

		self._qIdx = np.concatenate(qChunks) if qChunks else np.zeros(0, dtype=np.int32)
		self._dbIdx = np.concatenate(dbChunks) if dbChunks else np.zeros(0, dtype=np.int32)
		self._score = np.concatenate(scoreChunks) if scoreChunks else np.zeros(0, dtype=np.float32)
		if self.loadWarps:
			self._bank = np.concatenate(bankChunks) if bankChunks else np.zeros((0, 8, 8), dtype=np.float16)
			self._bankValid = np.concatenate(bankValidChunks) if bankValidChunks else np.zeros((0, 8, 8), dtype=bool)

	def __buildBands(self):
		masks = [
			self._score >= self.positiveFloor,
			(self._score >= self.bandEdges[0]) & (self._score < self.positiveFloor),
			(self._score >= self.bandEdges[1]) & (self._score < self.bandEdges[0]),
			self._score < self.bandEdges[1],
		]

		self._offsets = []		# [4] of (numQueries+1,) int64
		self._catDbIdx = []		# [4] of int32
		self._catScore = []		# [4] of float32
		self._counts = np.zeros((4, self.numQueries), dtype=np.int32)

		for cat, mask in enumerate(masks):
			q = self._qIdx[mask]
			db = self._dbIdx[mask]
			sc = self._score[mask]

			# primary key q (ascending), secondary -sc (i.e. sc descending)
			order = np.lexsort((-sc, q))
			q, db, sc = q[order], db[order], sc[order]

			counts = np.bincount(q, minlength=self.numQueries).astype(np.int32)
			offsets = np.zeros(self.numQueries + 1, dtype=np.int64)
			np.cumsum(counts, out=offsets[1:])

			self._offsets.append(offsets)
			self._catDbIdx.append(db)
			self._catScore.append(sc)
			self._counts[cat] = counts

		self._resolvedQueryMask = self._counts.sum(axis=0) > 0

	def __buildWarpIndex(self):
		key = self._qIdx.astype(np.int64) * self.numDb + self._dbIdx.astype(np.int64)
		order = np.argsort(key, kind='stable')
		self._warpKeys = key[order]
		self._warpBank = self._bank[order]
		self._warpBankValid = self._bankValid[order]
		del self._bank, self._bankValid

	def __printSummary(self):
		resolvedQueries = int(self._resolvedQueryMask.sum())
		unresolvedQueries = self.numQueries - resolvedQueries

		print('[{}] rows read {}, dropped invalid {}, dropped unknown dbID {}, dropped out-of-split query {}'.format(
			self.tag, self.rowsRead, self.droppedInvalid, self.droppedUnknownDb, self.droppedOutOfSplit))
		print('[{}] resolved queries {}/{}'.format(self.tag, resolvedQueries, self.numQueries))
		if unresolvedQueries > 0:
			print('[{}] WARNING: {} dataset queries have NO bank rows at all'.format(self.tag, unresolvedQueries))

		labels = ['positive', 'band0', 'band1', 'band2']
		for cat, label in enumerate(labels):
			p10 = np.percentile(self._counts[cat], 10)
			median = np.percentile(self._counts[cat], 50)
			print('[{}] {} count per query: p10={:.1f} median={:.1f}'.format(self.tag, label, p10, median))

	# Public surface

	def positivePool(self, queryID):
		return self.__slice(0, queryID)

	def negativeBand(self, queryID, band):
		if band not in (0, 1, 2):
			raise ValueError('band must be 0, 1 or 2, got {}'.format(band))
		return self.__slice(band + 1, queryID)

	def __slice(self, cat, queryID):
		lo, hi = self._offsets[cat][queryID], self._offsets[cat][queryID + 1]
		return self._catDbIdx[cat][lo:hi], self._catScore[cat][lo:hi]

	def warpedScore(self, queryID, dbID, qWarpIdx, cWarpIdx):
		if not self.loadWarps:
			raise RuntimeError('[{}] warpedScore requires construction with loadWarps=True'.format(self.tag))

		key = np.int64(queryID) * self.numDb + np.int64(dbID)
		idx = np.searchsorted(self._warpKeys, key)
		if idx >= len(self._warpKeys) or self._warpKeys[idx] != key:
			return None
		if not self._warpBankValid[idx, qWarpIdx, cWarpIdx]:
			return None
		return float(self._warpBank[idx, qWarpIdx, cWarpIdx])

	def warpCells(self, queryID, dbID):
		"""(scores, valid) as (8,8) float32/bool for this pair, or None if the pair is absent."""
		if not self.loadWarps:
			raise RuntimeError('[{}] warpCells requires construction with loadWarps=True'.format(self.tag))

		key = np.int64(queryID) * self.numDb + np.int64(dbID)
		idx = np.searchsorted(self._warpKeys, key)
		if idx >= len(self._warpKeys) or self._warpKeys[idx] != key:
			return None
		return self._warpBank[idx].astype(np.float32), self._warpBankValid[idx]

	def warps(self):
		return {
			'query': np.asarray(self._bankJson['query']),
			'query_weight': np.asarray(self._bankJson['query_weight']),
			'candidate': np.asarray(self._bankJson['candidate']),
			'candidate_weight': np.asarray(self._bankJson['candidate_weight']),
		}

	def queriesWithoutBand(self, counts):
		counts = np.asarray(counts).reshape(4, 1)
		deficient = np.any(self._counts < counts, axis=0)
		return np.nonzero(deficient)[0].astype(np.int32)
