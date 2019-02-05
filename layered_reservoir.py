import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import time

N_NODES = 200
SPECT_RADIUS = 0.98

def correct_dimensions(s, targetlength):
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x

def step_function(x):
    if x > 0.5:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1+np.exp(-10*x+1))


class LI_ESN_internal:

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, W=None, W_in=None,
                 noise=0.001, input_shift=None,
                 input_scaling=None, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, time_scale=None):
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        self.time_scale = time_scale
        self.W = W
        self.W_in = W_in

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _update(self, state, input_pattern):
        # leaky integrator model:
        # it can adjust timescales for each neurons.
        preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern))
        state = (1 - self.time_scale) * state + self.time_scale * np.tanh(preactivation)
        # state = (1 - self.time_scale) * state + self.time_scale * preactivation
        return (state + self.noise * self.time_scale * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def calc_lyapunov_exp(self, inputs, initial_distance, n):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        states1 = np.zeros((inputs.shape[0], self.n_reservoir))
        states2 = np.zeros((inputs.shape[0], self.n_reservoir))
        transient = min(int(inputs.shape[0] / 10), 100)
        for i in range(1, transient):
            states1[i, :] = self._update(states1[i-1], inputs[i, :])
        states2[transient-1, :] = states1[transient-1, :]
        states2[transient-1, n] = states2[transient-1, n] + initial_distance
        gamma_k_list = []
        for k in range(transient, inputs.shape[0]):
            states1[k, :] = self._update(states1[k-1], inputs[k, :])
            states2[k, :] = self._update(states2[k-1], inputs[k, :])
            gamma_k = np.linalg.norm(states2[k, :]-states1[k, :])
            gamma_k_list.append(gamma_k/initial_distance)
            states2[k, :] = states1[k, :] + (initial_distance/gamma_k)*(states2[k, :]-states1[k, :])
        lyapunov_exp = np.mean(np.log(gamma_k_list))
        return lyapunov_exp


    def fit(self, inputs, outputs):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = outputs

        # step the reservoir through the given input,output pairs:
        self.states = np.zeros((inputs.shape[0], self.n_reservoir))
        # self.states[0] = np.random.rand(self.n_reservoir)*2-1
        for n in range(1, inputs.shape[0]):
            self.states[n, :] = self._update(self.states[n - 1], inputs_scaled[n, :])
        transient = min(int(inputs.shape[0] / 10), 100)
        extended_states = np.hstack((self.states, inputs_scaled))

        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),teachers_scaled[transient:, :]).T
        # print(self.W_out.shape)

        # remember the last state for later:
        self.laststate = self.states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # apply learned weights to the collected states:
        pred_train = np.dot(extended_states, self.W_out.T)
        return pred_train

    def predict(self, inputs, continuation=True):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, inputs])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :])
            outputs[n + 1, :] = np.dot(self.W_out,np.concatenate([states[n + 1, :], inputs[n + 1, :]]))

        return self.out_activation(outputs[1:])
        # print(outputs[1:])
        # return np.heaviside(outputs[1:]-0.5, 0)*0.3


def make_layered_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    size = N/community_number
    G = np.zeros((N, N))
    for i in range(N):
        com_index = i//size
        for j in range(N):
            if j <= size*(com_index+1) and j >= size*com_index:
                if i != j and np.random.rand() < average_degree*(1-1*mu)/(size-1):
                    G[i][j] = np.random.randn()
            elif j >= size*(com_index-1) and j <= size*(com_index+1):
                if np.random.rand() < average_degree*mu/size:
                    G[i][j] = np.random.randn()

    return G



num_community=4

average_degree_range = np.arange(10, 50, 5)
mu_range = np.arange(0.05, 0.5, 0.05)

for average_degree in average_degree_range:
    for mu in mu_range:
        # average_degree = int(average_degree)
        # mu = int(mu)

        intri_ts1 = []
        intri_ts2 = []
        intri_ts3 = []

        for trial in range(500):
            sin_like_sequence = np.zeros((10, 5000))

            for i in range(10):
                x = np.arange(0, 250, 0.05)
                sin_like_sequence[i] = (np.random.rand(5000)*2-1)

            sin_like_sequence_T = sin_like_sequence.T

            community_size = N_NODES//num_community
            time_scale = np.ones(N_NODES)*0.5

            W = make_layered_network(N_NODES, average_degree, num_community, mu)
            W_IN = (np.random.rand(N_NODES, 10) * 2 - 1)*0.1
            W_IN[int(N_NODES/num_community):] = 0
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            spectral_radius = SPECT_RADIUS
            W = W * (spectral_radius / radius)

            esn = LI_ESN_internal(n_inputs=10,
                                  n_outputs=1,
                                  n_reservoir=N_NODES,
                                  W=W,
                                  W_in=W_IN,
                                  noise=0,
                                  time_scale=time_scale)

            _ = esn.fit(sin_like_sequence_T, np.random.rand(5000))
            states = esn.states


            autocorrelation_list = np.zeros((30,community_size))
            for i in range(community_size):
                for j in range(30):
                    autocorrelation_list[j][i] = np.corrcoef(states.T[community_size+i][101+j:4001+j], states.T[community_size+i][100:4000])[0][1]
            autocorrelation_mean_std_com1 = np.zeros((2, 30))
            for i in range(30):
                autocorrelation_mean_std_com1[0][i] = np.mean(autocorrelation_list[i])
                autocorrelation_mean_std_com1[1][i] = np.std(autocorrelation_list[i])

            autocorrelation_list = np.zeros((30,community_size))
            for i in range(community_size):
                for j in range(30):
                    autocorrelation_list[j][i] = np.corrcoef(states.T[community_size*2+i][101+j:4001+j], states.T[community_size*2+i][100:4000])[0][1]
            autocorrelation_mean_std_com2 = np.zeros((2, 30))
            for i in range(30):
                autocorrelation_mean_std_com2[0][i] = np.mean(autocorrelation_list[i])
                autocorrelation_mean_std_com2[1][i] = np.std(autocorrelation_list[i])

            autocorrelation_list = np.zeros((30,community_size))
            for i in range(community_size):
                for j in range(30):
                    autocorrelation_list[j][i] = np.corrcoef(states.T[community_size*3+i][101+j:4001+j], states.T[community_size*3+i][100:4000])[0][1]
            autocorrelation_mean_std_com3 = np.zeros((2, 30))
            for i in range(30):
                autocorrelation_mean_std_com3[0][i] = np.mean(autocorrelation_list[i])
                autocorrelation_mean_std_com3[1][i] = np.std(autocorrelation_list[i])

            ts1 = np.mean(np.diff(autocorrelation_mean_std_com1[0][:10], 1))
            ts2 = np.mean(np.diff(autocorrelation_mean_std_com2[0][:10], 1))
            ts3 = np.mean(np.diff(autocorrelation_mean_std_com3[0][:10], 1))

            intri_ts1.append(ts1)
            intri_ts2.append(ts2)
            intri_ts3.append(ts3)

        intrinsic_timescale = np.vstack((np.array(intri_ts1), np.array(intri_ts2), np.array(intri_ts3)))
        diff_1_2 = [np.diff(intrinsic_timescale.T[i])[0] for i in range(500)]
        diff_2_3 = [np.diff(intrinsic_timescale.T[i])[1] for i in range(500)]
        print(average_degree, '\t', mu, '\t', np.mean(intri_ts1),'\t',  np.std(intri_ts1), '\t', np.mean(intri_ts2), '\t', np.std(intri_ts2), '\t', np.mean(intri_ts3), '\t', np.std(intri_ts3),
                '\t', np.mean(diff_1_2), '\t', np.std(diff_1_2), '\t', np.mean(diff_2_3), '\t', np.std(diff_2_3))
