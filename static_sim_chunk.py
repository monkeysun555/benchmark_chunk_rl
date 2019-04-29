import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import live_player
import live_server
import static_a3c_chunk as a3c
import load

IF_NEW = 1
DEBUGGING = 0
if not IF_NEW:
	S_INFO = 8
	S_LEN = 12
else:
	S_INFO = 7
	S_LEN = 15
A_DIM = 6	
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 8

TRAIN_SEQ_LEN = 100
MODEL_SAVE_INTERVAL = 500

# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 12000.0]	# 5 actions

RANDOM_SEED = 13
RAND_RANGE = 1000
MS_IN_S = 1000.0
KB_IN_MB = 1000.0	# in ms
SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
# Initial buffer length on server side
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0									# Single time freezing time upper bound
USER_LATENCY_TOL = TARGET_LATENCY + USER_FREEZING_TOL		# Accumulate latency upperbound

STARTING_EPOCH = 0
NN_MODEL = None
# STARTING_EPOCH = 30000
# NN_MODEL = './results/nn_model_s_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_ep_' + str(STARTING_EPOCH) + '.ckpt'
TERMINAL_EPOCH = 20000

DEFAULT_ACTION = 0			# lowest bitrate
ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
REBUF_PENALTY = 3.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 5.0 * CHUNK_SEG_RATIO 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 3.0 * CHUNK_SEG_RATIO	# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1
RTT_LOW = 30.0

NOR_BW = 10.0
NOR_BUFFER = USER_LATENCY_TOL / MS_IN_S
NOR_CHUNK = CHUNK_IN_SEG
NOR_FREEZING = USER_FREEZING_TOL / MS_IN_S
NOR_RATE = np.log(BITRATE[-1]/BITRATE[0])
NOR_WAIT = CHUNK_DURATION / MS_IN_S
NOR_STATE = 2.0	# 0, 1, 2

if not IF_NEW:
	DATA_DIR = '../bw_traces/'
else:
	DATA_DIR = '../new_traces/train_sim_traces/'
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# TEST_LOG_FOLDER = './test_results/'
# TRAIN_TRACES = './traces/bandwidth/'


def ReLU(x):
	return x * (x > 0)

def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):

	# Initial server and player
	player = live_player.Live_Player(time_traces=all_cooked_time, throughput_traces=all_cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=agent_id)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH)

	with tf.Session() as sess, open(LOG_FILE + '_' + str(int(SERVER_START_UP_TH/MS_IN_S)) +'_agent_' + str(agent_id), 'wb') as log_file:
		actor = a3c.ActorNetwork(sess,
								 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
								 learning_rate=ACTOR_LR_RATE)
		critic = a3c.CriticNetwork(sess,
								   state_dim=[S_INFO, S_LEN],
								   learning_rate=CRITIC_LR_RATE)

		# initial synchronization of the network parameters from the coordinator
		actor_net_params, critic_net_params = net_params_queue.get()
		actor.set_network_params(actor_net_params)
		critic.set_network_params(critic_net_params)

		# FOr new trainning, using initial latency as the target
		t_latency = server.get_time() - player.get_display_time()
		action_num = DEFAULT_ACTION
		last_bit_rate = action_num
		bit_rate = action_num
		# last_bit_rate = DEFAULT_ACTION%len(BITRATE)
		# bit_rate = DEFAULT_ACTION%len(BITRATE)
		# playing_speed = NORMAL_PLAYING
		video_terminate = 0
		action_vec = np.zeros(A_DIM)
		action_vec[action_num] = 1

		s_batch = [np.zeros((S_INFO, S_LEN))]
		state = np.array(s_batch[-1], copy=True)		
		a_batch = [action_vec]
		r_batch = []
		entropy_record = []
		action_reward = 0.0		# Total reward is for all chunks within on segment
		take_action = 1
		latency = 0.0

		while True:
			# get download chunk info
			# Have to modify here, as we can get several chunks at the same time
			# And the recording are jointly calculated. 
			# assert len(server.chunks) >= 1

			# Here, should get server next_delivery. Might be several chunks or a whole segment
			# download_chunk_info = server.chunks[0]		
			# download_chunk_size = download_chunk_info[2]
			# download_chunk_idx = download_chunk_info[1]
			# download_seg_idx = download_chunk_info[0]
			download_chunk_info = server.get_next_delivery()
			download_seg_idx = download_chunk_info[0]
			download_chunk_idx = download_chunk_info[1]
			download_chunk_end_idx = download_chunk_info[2]
			download_chunk_size = download_chunk_info[3]
			chunk_number = download_chunk_end_idx - download_chunk_idx + 1
			assert chunk_number == 1
			if DEBUGGING:
				print("Segment id:", download_seg_idx)
				print("Chunk number:", chunk_number)
				print("Bitrate is:", bit_rate)
			server_wait_time = 0.0
			sync = 0
			missing_count = 0
			real_chunk_size, download_duration, freezing, time_out, player_state, rtt = player.fetch(bit_rate, download_chunk_size, 
																		download_seg_idx, download_chunk_idx, take_action, chunk_number)
			if DEBUGGING:
				print("After downloading, chunk size:", real_chunk_size)
				print("Duration:", download_duration)
				print("Freezing:", freezing)
				print("Is it timeout?", time_out)
				print("Player playing time:", player.get_display_time())
				print("Server time:", server.get_time())
			take_action = 0
			buffer_length = player.get_buffer()
			# print(download_duration, len(server.chunks), server.next_delivery)
			server_time = server.update(download_duration)

			if not time_out:
				# server.chunks.pop(0)
				server.clean_next_delivery()
				sync = player.check_resync(server_time)
			else:
				assert player.get_state() == 0
				assert np.round(player.get_buffer(), 3) == 0.0
				# Pay attention here, how time out influence next reward, the smoothness
				# Bit_rate will recalculated later, this is for reward calculation
				bit_rate = 0
				sync = 1

			if sync:
				if not IF_NEW:
					video_terminate = 1
				if DEBUGGING:
					print("Sync happens, latnecy will change")
				# To sync player, enter start up phase, buffer becomes zero
				sync_time, missing_count = server.sync_encoding_buffer()
				player.sync_playing(sync_time)
				buffer_length = player.get_buffer()

			latency = server.get_time() - player.get_display_time()
			player_state = player.get_state()

			log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
			log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
			last_bit_rate = bit_rate
			# print(log_bit_rate, log_last_bit_rate)
			reward = ACTION_REWARD * log_bit_rate * chunk_number \
					- REBUF_PENALTY * freezing / MS_IN_S \
					- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
					- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency - t_latency)/ MS_IN_S)-1) * chunk_number \
					- MISSING_PENALTY * missing_count
					# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
			if DEBUGGING:
				print("reward is made up of:", ACTION_REWARD * log_bit_rate * chunk_number, - REBUF_PENALTY * freezing / MS_IN_S, - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate),\
									 - LONG_DELAY_PENALTY*((LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S))-1) * chunk_number, - MISSING_PENALTY * missing_count)
			action_reward += reward
			# chech whether need to wait, using number of available segs
			if server.check_chunks_empty():
				server_wait_time = server.wait()
				assert server_wait_time > 0.0
				assert server_wait_time < CHUNK_DURATION
				# print("Before wait, player time:", player.get_display_time())
				player.wait(server_wait_time)
				# print("After wait, player time:", player.get_display_time())
				buffer_length = player.get_buffer()
			server.generate_next_delivery()
			# print("After waiting, latency is:", server.get_time() - player.get_display_time())
			# print("<===============================>")
			# print(bit_rate, download_duration, server_wait_time, player.buffer, \
			# 	server.time, player.playing_time, freezing, reward, action_reward)

			# Establish state for next iteration
			state = np.roll(state, -1, axis=1)
			if IF_NEW:
				state[0, -1] = real_chunk_size / (download_duration - rtt) / NOR_BW		# chunk size
				# state[1, -1] = download_duration / MS_IN_S				# downloading time
				state[1, -1] = buffer_length / MS_IN_S / NOR_BUFFER			# buffer length
				# state[2, -1] = chunk_number / NOR_CHUNK						# number of chunk sent
				state[2, -1] = log_bit_rate	/ NOR_RATE						# video bitrate
				# state[4, -1] = latency / MS_IN_S							# accu latency from start up
				state[3, -1] = sync 										# whether there is resync
				state[4, -1] = player_state	/ NOR_STATE						# state of player
				state[5, -1] = server_wait_time / MS_IN_S/ NOR_WAIT			# time of waiting for server
				state[6, -1] = freezing / MS_IN_S / NOR_FREEZING			# current freezing time
				if DEBUGGING:
					print(state)
			else:
				state[0, -1] = real_chunk_size / KB_IN_MB 		# chunk size
				state[1, -1] = download_duration / MS_IN_S		# downloading time
				state[2, -1] = buffer_length / MS_IN_S			# buffer length
				state[3, -1] = chunk_number						# number of chunk sent
				state[4, -1] = log_bit_rate						# video bitrate
				# state[4, -1] = latency / MS_IN_S				# accu latency from start up
				state[5, -1] = sync 							# whether there is resync
				# state[6, -1] = player_state					# state of player
				state[6, -1] = server_wait_time / MS_IN_S		# time of waiting for server
				state[7, -1] = freezing / MS_IN_S				# current freezing time
			# print state
			# generate next set of seg size
			# if add this, this will return to environment
			# next_chunk_size_info = server.chunks[0][2]	# not useful
			# state[7, :A_DIM] = next_chunk_size_info		# not useful
			# print(state)

			# Get next chunk/chunks information
			# Should not directly get the next one, but might be several chunks togethers
			# next_chunk_idx = server.chunks[0][1]

			# print state
			next_chunk_idx = server.get_next_delivery()[1]
			if next_chunk_idx == 0 or sync:
				# print(action_reward)
				take_action = 1
				r_batch.append(action_reward)
				action_reward = 0.0
				# If sync, might go to medium of segment, and there is no estimated chunk size
				'''
				next_seg_size_info = []
				if sync and not next_chunk_idx == 0:
					next_seg_size_info = [2 * np.sum(x) / KB_IN_MB for x in server.chunks[0][2]] 
				else:
					next_seg_size_info = [x/KB_IN_MB for x in server.chunks[0][3]]

				state[8, :A_DIM] = next_seg_size_info
				'''
				action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
				action_cumsum = np.cumsum(action_prob)
				# print(action_prob)
				# Selection action
				action_num = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

				bit_rate = action_num
				# if action_num >= len(BITRATE):
				# 	playing_speed = FAST_PLAYING
				# else:
				# 	playing_speed = NORMAL_PLAYING
				entropy_record.append(a3c.compute_entropy(action_prob[0]))
				log_file.write(str(buffer_length) + '\t' +
								str(freezing) + '\t' +
								str(time_out) + '\t' +
								# str(buffer_length) + '\t' +
								str(server_wait_time) + '\t' +
							    str(action_prob) + '\t' +
								str(reward) + '\n')
				log_file.flush()
			else:
				log_file.write(str(buffer_length) + '\t' +
								str(freezing) + '\t' +
								str(time_out) + '\t' +
								# str(buffer_length) + '\t' +
								str(server_wait_time) + '\t' +
								str(reward) + '\n')
				log_file.flush()

			if len(r_batch) >= TRAIN_SEQ_LEN or video_terminate:
				# print(r_batch)
				video_terminate = 1
				if len(s_batch) > 1:
					# if initial:
					exp_queue.put([s_batch[1:],  # ignore the first chuck
									a_batch[1:],  # since we don't have the
									r_batch[1:],  # control over it
									# terminal,
									{'entropy': entropy_record}])
						# initial = 0
					# else:
					# 	exp_queue.put([s_batch[:],  # ignore the first chuck
					# 					a_batch[:],  # since we don't have the
					# 					r_batch[:],  # control over it
					# 					# terminal,
					# 					{'entropy': entropy_record}])

					actor_net_params, critic_net_params = net_params_queue.get()
					actor.set_network_params(actor_net_params)
					critic.set_network_params(critic_net_params)

					del s_batch[:]
					del a_batch[:]
					del r_batch[:]
					del entropy_record[:]
					log_file.write('\n')  # so that in the log we know where video ends

				else:
					print("length of s batch is too short: ", len(s_batch))
					assert 0 == 1
					
			# This is infinit seq
			if next_chunk_idx == 0 or sync:
				if video_terminate:
					last_bit_rate = DEFAULT_ACTION
					bit_rate = DEFAULT_ACTION
					action_vec = np.zeros(A_DIM)
					action_vec[bit_rate] = 1
					s_batch.append(np.zeros((S_INFO, S_LEN)))
					a_batch.append(action_vec)
					video_terminate = 0

					# Reset player and server
					player.reset(USER_START_UP_TH)
					server.test_reset(SERVER_START_UP_TH)
					t_latency = server.get_time() - player.get_display_time()
				else:
					s_batch.append(state)
					state = np.array(s_batch[-1], copy=True)
					action_vec = np.zeros(A_DIM)
					action_vec[action_num] = 1
					a_batch.append(action_vec)


def central_agent(net_params_queues, exp_queues):
	assert len(net_params_queues) == NUM_AGENTS
	assert len(exp_queues) == NUM_AGENTS

	logging.basicConfig(filename=LOG_FILE + '_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_central',
						filemode='w',
						level=logging.INFO)

	# with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:
	with tf.Session() as sess:
		actor = a3c.ActorNetwork(sess,
									state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
									learning_rate=ACTOR_LR_RATE)
		critic = a3c.CriticNetwork(sess,
									state_dim=[S_INFO, S_LEN],
									learning_rate=CRITIC_LR_RATE)

		summary_ops, summary_vars = a3c.build_summaries()

		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
		saver = tf.train.Saver()  # save neural net parameters

		# restore neural net parameters
		nn_model = NN_MODEL
		if nn_model is not None:  # nn_model is the path to file
			saver.restore(sess, nn_model)
			print("Model restored.")

		epoch = STARTING_EPOCH

		while epoch < TERMINAL_EPOCH:
			# synchronize the network parameters of work agent
			actor_net_params = actor.get_network_params()
			critic_net_params = critic.get_network_params()
			for i in xrange(NUM_AGENTS):
				net_params_queues[i].put([actor_net_params, critic_net_params])

			total_batch_len = 0.0
			total_reward = 0.0
			total_td_loss = 0.0
			total_entropy = 0.0
			total_agents = 0.0 

			actor_gradient_batch = []
			critic_gradient_batch = []

			for i in xrange(NUM_AGENTS):
				s_batch, a_batch, r_batch, info = exp_queues[i].get()
				if len(s_batch) == 0:
					continue
				actor_gradient, critic_gradient, td_batch = \
					a3c.compute_gradients(
						s_batch=np.stack(s_batch, axis=0),
						a_batch=np.vstack(a_batch),
						r_batch=np.vstack(r_batch),
						# terminal=terminal, actor=actor, critic=critic)
						actor=actor, critic=critic)

				actor_gradient_batch.append(actor_gradient)
				critic_gradient_batch.append(critic_gradient)

				total_reward += np.sum(r_batch)
				total_td_loss += np.sum(td_batch)
				total_batch_len += len(r_batch)
				total_agents += 1.0
				total_entropy += np.sum(info['entropy'])

			# compute aggregated gradient
			assert NUM_AGENTS == len(actor_gradient_batch)
			assert len(actor_gradient_batch) == len(critic_gradient_batch)
			# assembled_actor_gradient = actor_gradient_batch[0]
			# assembled_critic_gradient = critic_gradient_batch[0]
			# for i in xrange(len(actor_gradient_batch) - 1):
			#     for j in xrange(len(assembled_actor_gradient)):
			#             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
			#             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
			# actor.apply_gradients(assembled_actor_gradient)
			# critic.apply_gradients(assembled_critic_gradient)
			for i in xrange(len(actor_gradient_batch)):
				actor.apply_gradients(actor_gradient_batch[i])
				critic.apply_gradients(critic_gradient_batch[i])

			# log training information
			epoch += 1
			avg_reward = total_reward  / total_agents		# avg reward is for each agent
			avg_td_loss = total_td_loss / total_batch_len	# avg td loss is for each action
			avg_entropy = total_entropy / total_batch_len	# avg entropy is for each action

			logging.info('Epoch: ' + str(epoch) +
						 ' TD_loss: ' + str(avg_td_loss) +
						 ' Avg_reward: ' + str(avg_reward) +
						 ' Avg_entropy: ' + str(avg_entropy))

			summary_str = sess.run(summary_ops, feed_dict={
				summary_vars[0]: avg_td_loss,
				summary_vars[1]: avg_reward,
				summary_vars[2]: avg_entropy
			})

			writer.add_summary(summary_str, epoch)
			writer.flush()

			# if epoch % 100 == 0:
			# 	print("epoch is: " + str(epoch))

			if epoch % MODEL_SAVE_INTERVAL == 0:
				# Save the neural net parameters to disk.
				print("epoch is: " + str(epoch) + ", and going to save")
				save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_s_" + str(int(SERVER_START_UP_TH/MS_IN_S)) + '_ep_' + 
									   str(epoch) + ".ckpt")
				logging.info("Model saved in file: " + save_path)
				print('Epoch: ' + str(epoch) +
						 ' TD_loss: ' + str(avg_td_loss) +
						 ' Avg_reward: ' + str(avg_reward) +
						 ' Avg_entropy: ' + str(avg_entropy))
				
				# no test current
				# testing(epoch, 
				# 	SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
				# 	test_log_file)
		return

def main():

	np.random.seed(RANDOM_SEED)
	# assert len(VIDEO_BIT_RATE) == A_DIM

	# create result directory
	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	# inter-process communication queues
	net_params_queues = []
	exp_queues = []
	for i in xrange(NUM_AGENTS):
		net_params_queues.append(mp.Queue(1))
		exp_queues.append(mp.Queue(1))

	# create a coordinator and multiple agent processes
	# (note: threading is not desirable due to python GIL)
	coordinator = mp.Process(target=central_agent,
							 args=(net_params_queues, exp_queues))
	coordinator.start()

	if not IF_NEW:
		all_cooked_time, all_cooked_bw, _ = load.loadBandwidth(DATA_DIR)		# For bw_traces
	else:
		all_cooked_time, all_cooked_bw, _ = load.new_loadBandwidth(DATA_DIR)	# For new_traces

	# all_cooked_vp, _ = load.loadViewport()
	# print(all_cooked_vp)
	# print(all_cooked_time)
	# print(all_cooked_bw)
	agents = []
	for i in xrange(NUM_AGENTS):
		agents.append(mp.Process(target=agent,
								 args=(i, all_cooked_time, all_cooked_bw, net_params_queues[i], exp_queues[i])))
	for i in xrange(NUM_AGENTS):
		agents[i].start()

	# wait unit training is done
	coordinator.join()


if __name__ == '__main__':
	main()
