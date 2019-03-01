import numpy as np

SEG_DURATION = 2000.0
FRAG_DURATION = 1000.0
CHUNK_DURATION = 500.0
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES

# CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)		# 4
# CHUNK_IN_FRAG = int(FRAG_DURATION/FRAG_DURATION)	# 2
# FRAG_IN_SEG = int(SEG_DURATION/FRAG_DURATION)		# 2 or 5
# ADD_DELAY = 3000.0

MS_IN_S = 1000.0
KB_IN_MB = 1000.0
# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]	# 5 actions

BITRATE_LOW_NOISE = 0.7
BITRATE_HIGH_NOISE = 1.3
RATIO_LOW = 2.0
RATIO_HIGH = 10.0
EST_LOW_NOISE = 0.98
EST_HIGH_NOISE = 1.02


class Live_Server(object):
	def __init__(self, seg_duration, frag_duration, chunk_duration, start_up_th):
		self.seg_duration = seg_duration
		self.frag_duration = frag_duration
		self.chunk_duration = chunk_duration
		self.frag_in_seg = seg_duration/frag_duration
		self.chunk_in_frag = frag_duration/chunk_duration
		self.chunk_in_seg = seg_duration/chunk_duration

		self.time = start_up_th + np.random.randint(1,frag_duration)		# start from 2000ms	to 3000.0
		self.start_up_th = start_up_th
		self.current_seg_idx = -1	# For initial
		self.current_chunk_idx = 0
		self.chunks = []	# 1 for initial chunk, 0 for following chunks
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		# self.delay_tol = start_up_th + add_delay
		self.encoding_update(0.0, self.time)

	def encoding_update(self, starting_time, end_time):
		temp_time = starting_time
		while True:
			next_time = (int(temp_time/self.chunk_duration) + 1) * self.chunk_duration
			if next_time > end_time:
				break
			# Generate chunks and insert to encoding buffer
			temp_time = next_time
			if next_time%self.seg_duration == self.chunk_duration:
				self.current_seg_idx += 1
				self.current_chunk_idx = 0
				self.generate_chunk_size()
				self.chunks.append([self.current_seg_idx, self.current_chunk_idx, \
									[chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size],\
									[np.sum(chunk_size) for chunk_size in self.current_seg_size]])	# for 2s segment
			else:
				self.current_chunk_idx += 1
				# print(self.current_chunk_idx, self.current_seg_size)
				self.chunks.append([self.current_seg_idx, self.current_chunk_idx, [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size]])

	def update(self, downloadig_time):
		# update time and encoding buffer
		# Has nothing to do with sync, migrate to player side
		# sync = 0	# sync play
		# missing_count = 0
		new_heading_time = 0.0
		pre_time = self.time
		self.time += downloadig_time
		self.encoding_update(pre_time, self.time)

		# # Check delay threshold
		# # A: Triggered by server sice, not reasonable
		# if len(self.chunks) > 1:
		# 	if self.time - playing_time > self.delay_tol:
		# 		new_heading_time, missing_count = self.sync_encoding_buffer()
		# 		sync = 1

		# # B: Receive time_out from client, and then resync
		# if time_out:
		# 	assert len(self.chunks) > 1
		# 	new_heading_time, missing_count = self.sync_encoding_buffer()
		# 	sync = 1
		# return sync, new_heading_time, missing_count
		return self.time

	def sync_encoding_buffer(self):
		target_encoding_len = 0
		new_heading_time = 0.0
		missing_count = 0
		# Modified for both 200 and 500 ms
		num_chunks = int((self.time%self.frag_duration)/self.chunk_duration)
		target_encoding_len = self.start_up_th/self.chunk_duration + num_chunks
		# Old, for 500
		# if self.time%self.frag_duration >= CHUNK_DURATION:
		# 	target_encoding_len = self.start_up_th/CHUNK_DURATION + 1
		# else:
		# 	target_encoding_len = self.start_up_th/CHUNK_DURATION
		# print(len(self.chunks))
		while not len(self.chunks) == target_encoding_len:
			self.chunks.pop(0)
			missing_count += 1
		new_heading_time = self.chunks[0][0] * self.seg_duration + self.chunks[0][1] * self.chunk_duration
		assert self.chunks[0][1]%self.chunk_in_frag == 0
		return new_heading_time, missing_count

	# chunk size for next/current segment
	def generate_chunk_size(self):
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		for i in range(int(self.frag_in_seg)):
			# Initial coef, all bitrate share the same coef 
			encoding_coef = np.random.uniform(BITRATE_LOW_NOISE, BITRATE_HIGH_NOISE)
			estimate_seg_size = [x * encoding_coef for x in BITRATE]
			# There is still noise for prediction, all bitrate cannot share the same coef exactly same
			seg_size = [np.random.uniform(EST_LOW_NOISE*x, EST_HIGH_NOISE*x) for x in estimate_seg_size]

			if self.chunk_in_frag == 2:
			# Distribute size for chunks, currently, it should depend on chunk duration (200 or 500)
				ratio = np.random.uniform(RATIO_LOW, RATIO_HIGH)
				seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
				for i in range(len(seg_ratio)):
					temp_ratio = seg_ratio[i]
					temp_aux_chunk_size = seg_size[i]/(1+temp_ratio)
					temp_ini_chunk_size = seg_size[i] - temp_aux_chunk_size
					self.current_seg_size[i].extend((temp_ini_chunk_size, temp_aux_chunk_size))
			# if 200ms, needs to be modified, not working
			else:
				assert 1 == 0
				ratio = np.random.uniform(RATIO_LOW, RATIO_HIGH)
				seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
				for i in range(len(seg_ratio)):
					temp_ratio = seg_ratio[i]
					temp_aux_chunk_size = seg_size[i]/(1+temp_ratio)
					temp_ini_chunk_size = seg_size[i] - temp_aux_chunk_size
					self.current_seg_size[i].extend((temp_ini_chunk_size, temp_aux_chunk_size))


	def wait(self):
		next_available_time = (int(self.time/self.chunk_duration) + 1) * self.chunk_duration
		self.encoding_update(self.time, next_available_time)
		assert len(self.chunks) == 1
		time_interval = next_available_time - self.time
		self.time = next_available_time
		return time_interval 

	def test_reset(self, start_up_th):
		self.time = start_up_th + np.random.randint(1,self.frag_duration)		# start from 2000ms	
		self.start_up_th = start_up_th
		self.current_seg_idx = -1
		self.current_chunk_idx = 0
		self.chunks = []	# 1 for initial chunk, 0 for following chunks
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		self.encoding_update(0.0, self.time)
		# self.delay_tol = start_up_th

def main():
	server = Live_Server(seg_duration=SEG_DURATION, frag_duration=FRAG_DURATION, chunk_duration=CHUNK_DURATION, start_up_th=SERVER_START_UP_TH)
	print(server.chunks, server.time)


if __name__ == '__main__':
	main()