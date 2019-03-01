import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import live_player
import live_server
import a3c
import load

S_INFO = 10
S_LEN = 12
A_DIM = 12
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

MAX_LIVE_LEN = 200
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]  # 5 actions
BITRATE = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0]

BITRATE_LOW_NOISE = 0.95
BITRATE_HIGH_NOISE = 1.05

MS_IN_S = 1000.0
KB_IN_MB = 1000.0   # in ms

SEG_DURATION = 2000.0
FRAG_DURATION = 1000.0
CHUNK_DURATION = 500.0
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_FRAG_RATIO = CHUNK_DURATION/FRAG_DURATION

SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 1000.0
USER_FREEZING_TOL = 3000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * FRAG_DURATION
USER_LATENCY_TOL = TARGET_LATENCY + 3000.0


DEFAULT_ACTION = 0	# lowest bitrate of ET
SMOOTH_PENALTY = 1.0 * CHUNK_SEG_RATIO
REBUF_PENALTY = 10.0	# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0 * CHUNK_SEG_RATIO 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 2.0	# not included
UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
FAST_PLAYING = 1.1		# For 1
NORMAL_PLAYING = 1.0	# For 0
SLOW_PLAYING = 0.9		# For -1

RANDOM_SEED = 11
RAND_RANGE = 1000

TEST_TRACE_NUM = 4

LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = '../test_traces/'
NN_MODEL = './models/nn_model_ep_99900.ckpt'
# NN_MODEL = sys.argv[1]

def ReLU(x):
	return x * (x > 0)

def record_tp(tp_trace, starting_time_idx, duration):
	tp_record = []
	offset = 0
	num_record = int(np.ceil(duration/FRAG_DURATION))
	for i in range(num_record):
		if starting_time_idx + i + offset >= len(tp_trace):
			offset = -len(tp_trace)
		tp_record.append(tp_trace[starting_time_idx + i + offset])
	return tp_record

def main():

	np.random.seed(RANDOM_SEED)

	assert len(BITRATE) == A_DIM

	all_cooked_time, all_cooked_bw, all_file_names = load.loadBandwidth(TEST_TRACES)

	player = live_player.Live_Player(time_traces=all_cooked_time, throughput_traces=all_cooked_bw, 
										seg_duration=SEG_DURATION, frag_duration=FRAG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)	
	server = live_server.Live_Server(seg_duration=SEG_DURATION, frag_duration=FRAG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH)

	log_path = LOG_FILE + '_' + all_file_names[player.trace_idx]
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
		if NN_MODEL is not None:  # NN_MODEL is the path to file
			saver.restore(sess, NN_MODEL)
			print("Testing model restored.")

		action_num = DEFAULT_ACTION	# 0
		last_bit_rate = DEFAULT_ACTION%len(BITRATE)
		bit_rate = DEFAULT_ACTION%len(BITRATE)
		playing_speed = NORMAL_PLAYING
		action_vec = np.zeros(A_DIM)
		action_vec[action_num] = 1
		take_action = 1
		latency = 0.0

		s_batch = [np.zeros((S_INFO, S_LEN))]
		state = np.array(s_batch[-1], copy=True)		
		a_batch = [action_vec]
		r_batch = []
		action_reward = 0.0		# Total reward is for all chunks within on segment

		video_count = 0
		starting_time = server.time
		starting_time_idx = player.time_idx

		while True:  # serve video forever
			assert len(server.chunks) >= 1
			download_chunk_info = server.chunks[0]
			download_chunk_size = download_chunk_info[2]
			download_chunk_idx = download_chunk_info[1]
			download_seg_idx = download_chunk_info[0]
			server_wait_time = 0.0
			sync = 0
			real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(bit_rate, download_chunk_size, 
																								download_seg_idx, download_chunk_idx, take_action, playing_speed)
			# print(freezing, time_out)
			take_action = 0
			past_time = download_duration
			buffer_length = player.buffer
			server_time = server.update(past_time)
			if not time_out:
				server.chunks.pop(0)
				sync = player.check_resync(server_time)
			else:
				assert player.state == 0
				assert np.round(player.buffer, 3) == 0.0
				# Pay attention here, how time out influence next reward, the smoothness
				# Bit_rate will recalculated later, this is for reward calculation
				bit_rate = 0
				sync = 1
			if sync:
				# To sync player, enter start up phase, buffer becomes zero
				sync_time, missing_count = server.sync_encoding_buffer()
				player.sync_playing(sync_time)
				buffer_length = player.buffer

			latency = server.time - player.playing_time
			player_state = player.state

			log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
			log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
			last_bit_rate = bit_rate
			# print(log_bit_rate, log_last_bit_rate)
			reward = ACTION_REWARD * log_bit_rate \
					- REBUF_PENALTY * freezing / MS_IN_S \
					- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
					- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) \
					- UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
					# - MISSING_PENALTY * missing_count
			# print(reward)
			action_reward += reward

			# chech whether need to wait, using number of available segs
			if len(server.chunks) == 0:
				server_wait_time = server.wait()
				assert server_wait_time > 0.0
				assert server_wait_time < CHUNK_DURATION
				player.wait(server_wait_time)
				buffer_length = player.buffer

			# Establish state for next iteration
			state = np.roll(state, -1, axis=1)
			state[0, -1] = BITRATE[bit_rate] / BITRATE[0]	# video bitrate
			state[1, -1] = real_chunk_size / KB_IN_MB 		# chunk size
			state[2, -1] = download_duration / MS_IN_S		# downloading time
			state[3, -1] = freezing / MS_IN_S				# current freezing time
			state[4, -1] = latency / MS_IN_S				# accu latency from start up
			state[5, -1] = sync 							# whether there is resync
			state[6, -1] = player_state						# state of player
			state[7, -1] = server_wait_time / MS_IN_S		# time of waiting for server
			state[8, -1] = buffer_length / MS_IN_S			# buffer length
			# generate next set of seg size
			# if add this, this will return to environment
			# next_chunk_size_info = server.chunks[0][2]	# not useful
			# state[7, :A_DIM] = next_chunk_size_info		# not useful
			# print(state)

			next_chunk_idx = server.chunks[0][1]
			if next_chunk_idx == 0 or sync:
				take_action = 1
				# print(action_reward)
				r_batch.append(action_reward)
				action_reward = 0.0
				# If sync, might go to medium of segment, and there is no estimated chunk size
				next_seg_size_info = []
				if sync and not next_chunk_idx == 0:
					next_seg_size_info = [2 * np.sum(x) / KB_IN_MB for x in server.chunks[0][2]] 
				else:
					next_seg_size_info = [x/KB_IN_MB for x in server.chunks[0][3]]

				state[9, :A_DIM] = next_seg_size_info
				action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
				action_cumsum = np.cumsum(action_prob)
				# print(action_prob)
				action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
				bit_rate = action_num%len(BITRATE)
				if action_num >= len(BITRATE):
					playing_speed = FAST_PLAYING
				else:
					playing_speed = NORMAL_PLAYING

			log_file.write(	str(server.time) + '\t' +
						    str(BITRATE[last_bit_rate]) + '\t' +
							str(buffer_length) + '\t' +
							str(freezing) + '\t' +
							str(time_out) + '\t' +
							str(server_wait_time) + '\t' +
						    str(sync) + '\t' +
						    str(missing_count) + '\t' +
						    str(player.state) + '\t' +
						    str(int(action_num/len(BITRATE))) + '\t' +						    
							str(reward) + '\n')
			log_file.flush()


			if len(r_batch) >= MAX_LIVE_LEN:
				# need to modify
				time_duration = server.time - starting_time
				tp_record = record_tp(player.throughput_trace, starting_time_idx, time_duration) 
				print(starting_time_idx, all_file_names[player.trace_idx], len(player.throughput_trace), player.time_idx, len(tp_record), np.sum(r_batch))
				log_file.write('\t'.join(str(tp) for tp in tp_record))
				log_file.write('\n' + str(starting_time))
				log_file.write('\n')
				log_file.close()

				action_num = DEFAULT_ACTION	# 0
				last_bit_rate = DEFAULT_ACTION%len(BITRATE)
				bit_rate = DEFAULT_ACTION%len(BITRATE)
				playing_speed = NORMAL_PLAYING

				del s_batch[:]
				del a_batch[:]
				del r_batch[:]

				action_vec = np.zeros(A_DIM)
				action_vec[action_num] = 1

				s_batch.append(np.zeros((S_INFO, S_LEN)))
				a_batch.append(action_vec)

				video_count += 1

				if video_count >= TEST_TRACE_NUM:
					break

				player.test_reset(start_up_th=USER_START_UP_TH)
				server.test_reset(start_up_th=SERVER_START_UP_TH)
				# Do not need to append state to s_batch as there is no iteration
				starting_time = server.time
				starting_time_idx = player.time_idx
				log_path = LOG_FILE + '_' + all_file_names[player.trace_idx]
				log_file = open(log_path, 'wb')
				take_action = 1
		
			else:	
				if next_chunk_idx == 0 or sync:			
					s_batch.append(state)
					state = np.array(s_batch[-1], copy=True)
					action_vec = np.zeros(A_DIM)
					action_vec[bit_rate] = 1
					a_batch.append(action_vec)

if __name__ == '__main__':
	main()
