import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import live_player_testing as live_player
import live_server_testing as live_server
import static_a3c_chunk as a3c
import load

S_INFO = 8
S_LEN = 12
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001


TEST_DURATION = 100				# Number of testing <===================== Change length here
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]  # 5 actions
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]

RANDOM_SEED = 13
RAND_RANGE = 1000
BITRATE_LOW_NOISE = 0.95
BITRATE_HIGH_NOISE = 1.05

MS_IN_S = 1000.0
KB_IN_MB = 1000.0   # in ms

SEG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION

# Initial buffer length on server side
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound


DEFAULT_ACTION = 0			# lowest bitrate
ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
REBUF_PENALTY = 10.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0 * CHUNK_SEG_RATIO 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 2.0			# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1


RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others

TEST_TRACE_NUM = 1

DATA_DIR = '../bw_traces/'
TRACE_NAME = '70ms_loss0.5_m5.txt'

LOG_FILE_DIR = './test_results'
LOG_FILE = LOG_FILE_DIR + '/rlchunk'

END_EPOCH =  63800
TEST_TRACES = '../test_traces/'
NN_MODEL = './models/nn_model_s_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_ep_' + str(END_EPOCH) + '.ckpt'
# NN_MODEL = './models_new/nn_model_s_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_ep_' + str(END_EPOCH) + '.ckpt'
# NN_MODEL = sys.argv[1]

def ReLU(x):
	return x * (x > 0)

def record_tp(tp_trace, starting_time_idx, duration):
	tp_record = []
	offset = 0
	num_record = int(np.ceil(duration/SEG_DURATION))
	for i in range(num_record):
		if starting_time_idx + i + offset >= len(tp_trace):
			offset = -len(tp_trace)
		tp_record.append(tp_trace[starting_time_idx + i + offset])
	return tp_record

def main():

	np.random.seed(RANDOM_SEED)
	if not os.path.isdir(LOG_FILE_DIR):
		os.makedirs(LOG_FILE_DIR)

	assert len(BITRATE) == A_DIM

	cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)

	player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)
	print server.get_time()

	log_path = LOG_FILE + '_' + TRACE_NAME
	log_file = open(log_path, 'wb')

	with tf.Session() as sess:
		actor = a3c.ActorNetwork(sess,
								 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
								 learning_rate=ACTOR_LR_RATE)
		critic = a3c.CriticNetwork(sess,
								   state_dim=[S_INFO, S_LEN],
								   learning_rate=CRITIC_LR_RATE)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()  # save neural net parameters

		# restore neural net parameters
		if NN_MODEL:  
			# NN_MODEL is the path to file
			print NN_MODEL
			saver.restore(sess, NN_MODEL)
			print("Testing model restored.")
		else: assert 0 == 1

		action_num = DEFAULT_ACTION	# 0
		last_bit_rate = action_num
		bit_rate = action_num

		state = np.zeros((S_INFO, S_LEN))
		r_batch = []
		action_reward = 0.0		# Total reward is for all chunks within on segment
		take_action = 1
		latency = 0.0
		starting_time = server.get_time()
		starting_time_idx = player.get_time_idx()
		init = 1
		for i in range(TEST_DURATION):
			print "Current index: ", i
			if init: 
				if CHUNK_IN_SEG == 5:
					ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
				else:
					ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
				
				server.set_ratio(ratio)
				server.init_encoding()
				init = 0

			action_reward = 0.0	
			take_action = 1

			while True:  # serve video forever
				download_chunk_info = server.get_next_delivery()
				# print "chunk info is " + str(download_chunk_info)
				download_seg_idx = download_chunk_info[0]
				download_chunk_idx = download_chunk_info[1]
				download_chunk_end_idx = download_chunk_info[2]
				download_chunk_size = download_chunk_info[3][bit_rate]		# Might be several chunks
				chunk_number = download_chunk_end_idx - download_chunk_idx + 1
				server_wait_time = 0.0
				sync = 0
				missing_count = 0
				real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																		download_seg_idx, download_chunk_idx, take_action, chunk_number)
				take_action = 0
				buffer_length = player.get_buffer_length()
				server_time = server.update(download_duration)

				if not time_out:
					# server.chunks.pop(0)
					server.clean_next_delivery()
					sync = player.check_resync(server_time)
				else:
					assert player.get_state() == 0
					assert np.round(player.get_buffer_length(), 3) == 0.0
					# Pay attention here, how time out influence next reward, the smoothness
					# Bit_rate will recalculated later, this is for reward calculation
					bit_rate = 0
					sync = 1
				# Disable sync for current situation
				if sync:
					print "Should not happen"
					assert 0 == 1
					# To sync player, enter start up phase, buffer becomes zero
					sync_time, missing_count = server.sync_encoding_buffer()
					player.sync_playing(sync_time)
					buffer_length = player.get_buffer_length()

				latency = server.get_time() - player.get_playing_time()
				# print "latency is: ", latency/MS_IN_S
				player_state = player.get_state()

				log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
				log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
				last_bit_rate = bit_rate
				# print(log_bit_rate, log_last_bit_rate)
				reward = ACTION_REWARD * log_bit_rate * chunk_number \
						- REBUF_PENALTY * freezing / MS_IN_S \
						- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
						- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_number\
						# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
						# - MISSING_PENALTY * missing_count
				# print(reward)
				action_reward += reward

				# chech whether need to wait, using number of available segs
				if server.check_chunks_empty():
					# print "Enter wait"
					server_wait_time = server.wait()
					# print " Has to wait: ", server_wait_time
					assert server_wait_time > 0.0
					assert server_wait_time < CHUNK_DURATION
					# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
					player.wait(server_wait_time)
					# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
					buffer_length = player.get_buffer_length()

				# Establish state for next iteration
				state = np.roll(state, -1, axis=1)
				state[0, -1] = real_chunk_size / KB_IN_MB 		# chunk size
				state[1, -1] = download_duration / MS_IN_S		# downloading time
				state[2, -1] = buffer_length / MS_IN_S			# buffer length
				state[3, -1] = BITRATE[bit_rate] / BITRATE[0]	# video bitrate
				# state[4, -1] = latency / MS_IN_S				# accu latency from start up
				state[4, -1] = sync 							# whether there is resync
				state[5, -1] = player_state						# state of player
				state[6, -1] = server_wait_time / MS_IN_S		# time of waiting for server
				state[7, -1] = freezing / MS_IN_S				# current freezing time
				# generate next set of seg size
				# if add this, this will return to environment
				# next_chunk_size_info = server.chunks[0][2]	# not useful
				# state[7, :A_DIM] = next_chunk_size_info		# not useful
				# print(state)
				if CHUNK_IN_SEG == 5:
					ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
				else:
					ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
				server.set_ratio(ratio)
				server.generate_next_delivery()
				next_chunk_idx = server.get_next_delivery()[1]

				if next_chunk_idx == 0 or sync:
					if sync:
						# Process sync
						print "Should not happen!"
						assert 0 == 1
					else:
						take_action = 1
						# print(action_reward)
						r_batch.append(action_reward)
						# If sync, might go to medium of segment, and there is no estimated chunk size
						action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
						# # Using random
						action_cumsum = np.cumsum(action_prob)
						# print(action_prob)
						action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

						# Use action with largest prob
						# action_num = action_prob.argmax()
						bit_rate = action_num

						# if action_num >= len(BITRATE):
						# 	playing_speed = FAST_PLAYING
						# else:
						# 	playing_speed = NORMAL_PLAYING

						log_file.write(	str(server.time) + '\t' +
									    str(BITRATE[last_bit_rate]) + '\t' +
										str(buffer_length) + '\t' +
										str(freezing) + '\t' +
										str(time_out) + '\t' +
										str(server_wait_time) + '\t' +
									    str(sync) + '\t' +
									    str(latency) + '\t' +
									    str(player.get_state()) + '\t' +
									    str(int(action_num/len(BITRATE))) + '\t' +						    
										str(action_reward) + '\n')
						log_file.flush()
						action_reward = 0.0
					break

		# need to modify
		time_duration = server.get_time() - starting_time
		tp_record = record_tp(player.get_throughput_trace(), starting_time_idx, time_duration) 
		print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
		log_file.write('\t'.join(str(tp) for tp in tp_record))
		log_file.write('\n' + str(starting_time))
		log_file.write('\n')
		log_file.close()		
					

if __name__ == '__main__':
	main()