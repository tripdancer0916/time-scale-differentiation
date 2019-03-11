import numpy as np
import time

N_NODES = 200
SPECT_RADIUS = 0.98
trainlen = 5000
future = 1000
L = 20
noise = 0.01


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
    return 1 / (1 + np.exp(-10 * x + 1))


class LIESN(object):
    def __init__(self, n_inputs, n_outputs, n_reservoir=200, W=None, W_in=None,
                 noise=0.001, input_shift=None,
                 input_scaling=None,
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

        self.states = None

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
        return state + self.noise * self.time_scale * (self.random_state_.rand(self.n_reservoir) - 0.5)

    def fit(self, inputs, outputs):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = outputs

        # step the reservoir through the given input,output pairs:
        self.states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            self.states[n, :] = self._update(self.states[n - 1], inputs_scaled[n, :])
        transient = min(int(inputs.shape[0] / 10), 100)
        extended_states = np.hstack((self.states, inputs_scaled))

        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]), teachers_scaled[transient:, :]).T

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
            outputs[n + 1, :] = np.dot(self.W_out, np.concatenate([states[n + 1, :], inputs[n + 1, :]]))

        return self.out_activation(outputs[1:])

    def knockout_predict(self, inputs, layer, continuation=True):
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

        knockout_nodes = np.random.choice(range(50 * (layer - 1), 50 * layer), 5)

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :])
            tmp_states = states.copy()
            for ko in knockout_nodes:
                tmp_states.T[ko][:] = 0
            outputs[n + 1, :] = np.dot(self.W_out, np.concatenate([tmp_states[n + 1, :], inputs[n + 1, :]]))

        return self.out_activation(outputs[1:])


def make_layered_network(N, average_degree, community_number, mu):
    assert N % community_number == 0, 'N must be devisible by community_number'
    size = N / community_number
    G = np.zeros((N, N))
    for i in range(N):
        com_index = i // size
        for j in range(N):
            if j <= size * (com_index + 1) and j >= size * com_index:
                if i != j and np.random.rand() < average_degree * (1 - 1 * mu) / (size - 1):
                    G[i][j] = np.random.randn()
            elif j >= size * (com_index - 1) and j <= size * (com_index + 1):
                if np.random.rand() < average_degree * mu / size:
                    G[i][j] = np.random.randn()
    return G


def memory_capacity(L, buffer, data, output_data):
    MC = 0
    for k in range(L):
        cov_matrix = np.cov(
            np.array([data[trainlen + L - (k + 1): trainlen + L - (k + 1) + 1000], output_data.T[k]]))
        MC_k = cov_matrix[0][1] ** 2
        MC_k = MC_k / (np.var(data[trainlen + L:]) * np.var(output_data.T[k]))
        MC += MC_k
    return MC


average_degree = 50
num_community = 4
mu = 0.25
W = make_layered_network(N_NODES, average_degree, num_community, mu)

ko_ratio_all = []
community_size = N_NODES // num_community
time_scale = np.ones(N_NODES)

total_len = future + trainlen + L

start = time.time()
for knockout_layer in range(1, num_community+1):
    ko_ratio_all_each_layer = []

    for trial in range(1000):
        data = np.random.choice([0, 1], total_len)
        W_IN = (np.random.rand(N_NODES, 1) * 2 - 1) * 0.1
        W_IN[int(N_NODES / num_community):] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        spectral_radius = SPECT_RADIUS
        W = W * (spectral_radius / radius)

        esn = LIESN(n_inputs=1,
                    n_outputs=L,
                    n_reservoir=N_NODES,
                    W=W,
                    W_in=W_IN,
                    noise=0.01,
                    time_scale=time_scale)

        target = np.zeros((total_len - L, L))
        for i in range(L):
            target.T[i][:] = data[L - (i + 1):-(i + 1)]
        pred_training = esn.fit(data[L:trainlen + L], target[:trainlen])

        knockout_prediction = esn.knockout_predict(data[trainlen + L:], layer=knockout_layer)
        prediction = esn.predict(data[trainlen + L:])
        pred_loss_list = []
        ko_pred_loss_list = []
        ko_ratio_list = []
        for l in range(20):
            pred_loss = np.linalg.norm([target.T[l][trainlen + L:] - prediction.T[l][L:]])
            ko_pred_loss = np.linalg.norm([target.T[l][trainlen + L:] - knockout_prediction.T[l][L:]])
            ko_ratio = ko_pred_loss / pred_loss
            pred_loss_list.append(pred_loss)
            ko_pred_loss_list.append(ko_pred_loss)
            ko_ratio_list.append(ko_ratio)

        ko_ratio_all_each_layer.append(ko_ratio_list)
    print(time.time() - start)
    ko_ratio_all.append(ko_ratio_all_each_layer)

final_result = np.array(ko_ratio_all)
np.savez('time_scale_differentiation_mu_{}_avgk_{}_noise_{}.npz'.format(mu, average_degree, noise), final_result)
