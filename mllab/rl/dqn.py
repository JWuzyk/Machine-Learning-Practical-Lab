import time
import gym
import numpy as np
import keras
import keras.backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf

from gym.envs.box2d.car_racing import STATE_H as CR_STATE_H, STATE_W as CR_STATE_W
from gym.spaces import Space, Box
from gym.envs.box2d import CarRacing
from collections import deque
from .sum_tree import SumTree


class RealNormalVariable(Space):
    """
    A space for normal real variables.

    The space can be used in OpenAI gym environments to describe
    state and actions spaces.
    """

    def __init__(self, mean, cov, dtype=K.floatx()):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        assert self.mean.ndim == 1
        assert self.cov.shape in (self.mean.shape, self.mean.shape * 2)
        super().__init__(self.mean.shape, dtype=dtype)

    def contains(self, v):
        return np.array(v).shape == self.mean.shape

    def sample(self):
        if self.cov.ndim == 1:
            return self.np_random.normal(self.mean, self.cov)
        else:
            return self.np_random.multivariate_normal(self.mean, self.cov)

class DiscretePoints(Space):
    """
    A space for a discrete set of points (real vectors).
    """

    def __init__(self, points, dtype=K.floatx()):
        self.points = [np.array(point) for point in points]
        assert self.points, "Points must not be empty"
        assert all(p.shape == self.points[0].shape for p in self.points[1:])
        super().__init__(self.points[0].shape, dtype)

    def contains(self, v):
        v = np.array(v)
        return any(np.allclose(v, p) for p in self.points)

    def sample(self, return_index=False):
        index = self.np_random.randint(0, len(self.points))
        if return_index:
            return index, self.points[index]
        else:
            return self.points[index]

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)


class DiscretizableBox(Box):
    def discretize(self, points):
        assert all(self.contains(p) for p in points)
        return DiscretePoints(points, points[0].dtype)


class NumericCarRacing(CarRacing):
    """
    A RL environment based on Gym's car racing.

    The state space is modified by returning information on the
    car like velocity numerically instead of graphically.
    """
    name = 'CarRacing-mllab-v0'

    def __init__(self, verbose=1):
        super().__init__(verbose=verbose)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(CR_STATE_H, CR_STATE_W, 3), dtype=np.uint8),
            # absolute speed, 4 times wheel angular velocity, wheel angle, car angular velocity
            RealNormalVariable(
                mean=[0, 0, 0, 0, 0, 0, 0],
                cov=[2, 1, 1, 1, 1, 2, 2] # those are more or less arbitrary
            )
        ))
        self.action_space = DiscretizableBox(np.array([-1,0,0]), np.array([+1,+1,+1]), dtype=K.floatx()) # steer, gas, brake

    def render(self, mode='human'):
        self.mode = mode
        return super().render(mode=mode)

    def render_indicators(self, W, H):
        if self.mode == 'human':
            return super().render_indicators(W, H)

    def step(self, action):
        state, step_reward, done, extra = super().step(action)
        measurements = np.array([
            np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1])),
            self.car.wheels[0].omega,
            self.car.wheels[1].omega,
            self.car.wheels[2].omega,
            self.car.wheels[3].omega,
            self.car.wheels[0].joint.angle,
            self.car.hull.angularVelocity])
        self.state = (state, measurements)
        return self.state, step_reward, done, extra


# hack for interactive sessions where this code is relaoded:
# We need to remove the environment, it can not be overwritten.
if NumericCarRacing.name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[NumericCarRacing.name]

gym.register(
    id=NumericCarRacing.name,
    entry_point=NumericCarRacing,
    max_episode_steps=1000,
    reward_threshold=900,
)


class ReplayMemory:
    """Replay memory which can store a fixed number of states."""

    def __init__(self, *shapes, capacity=10_000, batch_size=32, random_state=None):
        assert capacity <= 2 ** 31 - 1, "Capacity limited by int32"
        self._shapes = shapes
        self.capacity = capacity
        self.default_batch_size = batch_size
        self.random_state = random_state or np.random.RandomState()
        self.purge()

    def add(self, state, action, reward, new_state):
        """Add an element to the memory, remove oldest one if necessary."""
        self._len = min(self._len + 1, self.capacity)
        idx = self._end
        self._end = (self._end + 1) % self.capacity
        for _s, s in zip(self._state, state):
            _s[idx] = s
        self._reward[idx] = reward
        self._action[idx] = action
        if new_state is not None:
            for _s, s in zip(self._new_state, new_state):
                _s[idx] = s
        self._not_terminal[idx] = new_state is not None

    def sample(self, importance_criterion, progression=0.0, batch_size=None):
        """
        Draw random samples from the memory.

        Parameters
        ==========
        importance_criterion: callable
            This *might* be called with the transitions and must return
            a vector of real values. Often this is the TD-error.
        progression: scalar in [0, 1]
            The percentage of steps taken during training.

        Returns
        =======
        Returns two values. A list of transitions which is a tuple
        of arrays:

        - state
        - action taken (int)
        - reward
        - new state
        - boolean array which indicates which states were not terminal

        The second value is a array of loss weights, or None. The weights must be applied
        to the gradient update (importance sampling). If None is returned all weights
        are considered equal.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        batch_size = min(batch_size, len(self))
        indices = self.random_state.choice(np.arange(0, len(self), dtype='int'), batch_size, replace=False)
        transitions = self._get(indices)
        return transitions, None

    def __len__(self):
        return self._len

    def _get(self, indices):
        nt = self._not_terminal[indices]
        return (
            [s[indices] for s in self._state],
            self._action[indices],
            self._reward[indices],
            [s[indices][nt] for s in self._new_state],
            nt,
        )

    def purge(self):
        self._end = 0
        self._len = 0
        dtype = K.floatx()
        self._state = tuple(np.ones((self.capacity,) + shape, dtype=dtype) for shape in self._shapes)
        self._new_state = tuple(np.ones((self.capacity,) + shape, dtype=dtype) for shape in self._shapes)
        self._reward = np.ones(self.capacity, dtype=K.floatx())
        self._action = np.ones(self.capacity, dtype='int32')
        self._not_terminal = np.ones(self.capacity, dtype='bool')


class ProportionalPrioritizationReplayMemory:
    def __init__(self, *shapes, alpha=0.6, beta=0.4, epsilon=0.05,
                 capacity=10_000, batch_size=32,
                 random_state=None, clip=(-1,1)):
        """
        The two parameters alpha and beta are described in the
        paper (https://arxiv.org/pdf/1511.05952.pdf).
        """
        assert capacity <= 2 ** 31 - 1, "Capacity limited by int32"
        self._shapes = shapes
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        # clip importance to be
        self.clip = clip
        self.epsilon = epsilon
        self.default_batch_size = batch_size
        self.random_state = random_state or np.random.RandomState()
        self.purge()

    def add(self, state, action, reward, new_state):
        p_max = self._tree.max
        if p_max is None:
            p_max = 1
        idx = self._tree.add(p_max)
        for _s, s in zip(self._state, state):
            _s[idx] = s
        self._reward[idx] = reward
        self._action[idx] = action
        if new_state is not None:
            for _s, s in zip(self._new_state, new_state):
                _s[idx] = s
        self._not_terminal[idx] = new_state is not None

    def sample(self, importance_criterion, progression=0.0, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size
        partition = np.linspace(0, self._tree.total, batch_size, endpoint=False)
        size = self._tree.total / batch_size
        uniform = np.random.random(batch_size) * size + partition
        indices = np.zeros(batch_size, dtype='int32')
        P = np.zeros(batch_size, dtype='float64')
        for i, u in enumerate(uniform):
            indices[i], P[i] = self._tree.find(p=u)
            if abs(P[i]) < 1e-7:
                print(f"Save, capacity={self._tree.capacity}")
                np.savez('tree.npz', self._tree.tree)
        P /= self._tree.total

        nt = self._not_terminal[indices]
        transitions = (
            [s[indices] for s in self._state],
            self._action[indices],
            self._reward[indices],
            [s[indices][nt] for s in self._new_state],
            nt,
        )
        # update transition priorities
        importance = importance_criterion(*transitions)
        if self.clip is not None:
            importance = importance.clip(*self.clip)
        importance = (importance + self.epsilon) ** self.alpha
        for i, p in zip(indices, importance):
            self._tree.update(i, p)
        # anneal beta from the starting value to 1
        beta = self.beta * (1  - progression) + progression
        weights = (self.capacity * P) ** (-beta)
        weights /= weights.max()
        if np.any(np.isnan(weights)):
            print("WARNING: Some sample weights are NaN")
            # If any weight is nan, network parameters will become nan
            weights = None
        return transitions, weights

    def __len__(self):
        return self._len

    def purge(self):
        self._len = 0
        dtype = K.floatx()
        self._tree = SumTree(self.capacity)
        self._state = tuple(np.ones((self.capacity,) + shape, dtype=dtype) for shape in self._shapes)
        self._new_state = tuple(np.ones((self.capacity,) + shape, dtype=dtype) for shape in self._shapes)
        self._reward = np.ones(self.capacity, dtype=K.floatx())
        self._action = np.ones(self.capacity, dtype='int32')
        self._not_terminal = np.ones(self.capacity, dtype='bool')


class BaseQNetwork:
    def __init__(self, state_shape, action_space, model=None, clipped_loss=None):
        """Create a network which maps a state action pair to a value."""
        self.action_space = action_space
        self.state_shape = state_shape
        if model is None:
            qmodel, opt = self.build_model(state_shape)
            qmodel.layers[-1].name = 'qlayer'
            assert qmodel.output_shape == (None, len(self.action_space)), "Output shape must match number of actions"

            # We add a layer which gets the labels an input.
            # This is a trick to build the custom loss function we need.
            action = keras.layers.Input(shape=(1,), dtype='int32')
            loss_layer = keras.layers.Lambda(
                self._loss_layer, output_shape=(1,), name='loss')([qmodel.output, action])

            assert isinstance(qmodel.input, list)
            model = keras.Model(inputs=qmodel.input + [action], outputs=loss_layer)
            if clipped_loss is None:
                clipped_loss = True
            model.compile(loss=(self._clipped_mse if clipped_loss else 'mse'), optimizer=opt)
            self.model = model
        else:
            if clipped_loss is not None:
                raise ValueError("Can not give model and clipped_loss argument to constructor")
            self.model = model
        self.qmodel = keras.Model(
            inputs=self.model.input[:2],
            outputs=self.model.get_layer('qlayer').output)

    def _loss_layer(self, args):
        q_layer, action_indices = args
        if K.backend() == 'tensorflow':
            row_indices = tf.range(tf.shape(action_indices)[0])
            full_indices = tf.stack([row_indices, action_indices[:,0]], axis=1)
            qa_values = tf.gather_nd(q_layer, full_indices)
        else:
            batch_size = K.shape(q_layer)[0]
            offsets = K.arange(0, batch_size * len(self.action_space), step=len(self.action_space))
            indices = offsets + K.flatten(action_indices)
            qa_values = K.gather(K.flatten(q_layer), indices)
        qa_values = K.reshape(qa_values, K.shape(action_indices))
        return qa_values

    def _clipped_mse(self, y_true, y_pred):
        diff = y_true - y_pred
        inner = K.greater(diff, -1) & K.less(diff, 1)
        return K.mean(K.switch(inner,
                               K.square(diff) * 0.5 + 0.5,
                               K.abs(diff)))

    def build_model(self, state_shape):
        """
        Build the neural network model.

        Returns
        -------
        model : Compiled Keras model
        """
        raise NotImplementedError

    def __call__(self, states):
        if isinstance(states, tuple):
            return self.qmodel.predict([[states[0]], [states[1]]])[0]
        else:
            return self.qmodel.predict(states)

    def get_weights(self):
        """Get network weights."""
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, filepath):
        """Store the network weights to a file."""
        self.model.save_weights(filepath)

    def load(self, filepath):
        """Load the network weights from a file."""
        self.model.load_weights(filepath)

    def copy(self):
        with keras.utils.generic_utils.custom_object_scope({'_loss_layer': self._loss_layer, '_clipped_mse': self._clipped_mse}):
            model = keras.models.clone_model(self.model)
            model.set_weights(self.model.get_weights())
        return self.__class__(
            self.state_shape,
            self.action_space,
            model=model)


class EpsilonGreedyPolicy:
    """
    An ε-greedy policy strategy.

    It takes with probably ε a random action, otherwise a maximizing action.
    ε is linearly decreased, after more than the final exploration steps
    ε is fixed to its final value.
    """
    initial_exploration = 1.0
    final_exploration = 0.1
    evaluation_exploration = 0.1
    final_exploration_step = 1_000_000

    def __init__(self, q_network, random_state=None):
        self.q_network = q_network
        self.random_state = random_state or np.random.RandomState()

    def __call__(self, state, step=None):
        # This is done to check that the method is called with only one
        # state.
        assert isinstance(state, tuple), "State must be a tuple"
        if step is not None:
            step = min(step, self.final_exploration_step)
            eps = (
                self.initial_exploration * (self.final_exploration_step - step) +
                self.final_exploration * (step - 1)) / (self.final_exploration_step - 1)
        else:
            eps = self.evaluation_exploration
        if self.random_state.rand() <= eps:
            return self.q_network.action_space.sample(return_index=True)
        else:
            index = np.argmax(self.q_network(state))
            return index, self.q_network.action_space[index]

    def sample(self, *args, **kwargs):
        """Sample a random action from the action space of the Q-network."""
        return self.q_network.action_space.sample(*args, **kwargs)

    def gradient_step(self, states, actions, labels, sample_weights=None):
        """Perform a gradient descent step using the given batch."""
        self.q_network.model.train_on_batch(states + [actions], labels.reshape(-1, 1), sample_weight=sample_weights)

    def copy(self):
        """Create an independent copy of the policy."""
        q_network = self.q_network.copy()
        return EpsilonGreedyPolicy(q_network, self.random_state)

    def copy_weights_from(self, policy):
        self.q_network.set_weights(policy.q_network.get_weights())


def render_policy(env, preprocess, policy):
    """Visualize a policy for an environment."""
    env.reset()
    while True:
        state = preprocess(env.state)
        terminal = env.step(policy(state)[1])[2]
        env.render()
        if terminal:
            break


def sample_policy_from_human(env, action_space, preprocess, replay_memory):
    """
    Observe a policy from a human player and return a replay buffer.

    Press ESC to restart and close the window to stop.
    """
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )

    control = {'abort': False, 'restart': False}
    abort = False
    restart = False
    def on_key_press(k, mod):
        if k == key.ESCAPE:
            control['abort'] = True
        elif k == key.ENTER:
            control['restart'] = True
        elif k == key.LEFT:  a[0] = -1.0
        elif k == key.RIGHT: a[0] = +1.0
        elif k == key.UP:    a[1] = +1.0
        elif k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def on_key_release(k, mod):
        if k == key.LEFT  and a[0]== -1.0:   a[0] = 0
        elif k == key.RIGHT and a[0]== +1.0: a[0] = 0
        elif k == key.UP:                    a[1] = 0
        elif k == key.DOWN:                  a[2] = 0

    env.render()
    env.viewer.window.on_key_press = on_key_press
    env.viewer.window.on_key_release = on_key_release
    is_open = True
    while is_open and not control['abort']:
        env.reset()
        control['restart'] = False
        s = preprocess(s)
        while True:
            if control['abort']:
                break
            s2, r, done, info = env.step(a)
            action = action_space.closest(a)
            s2 = preprocess(s2)
            replay_memory.add(s, action, r, s2)
            s = s2
            is_open = env.render()
            if done or control['restart'] or is_open == False:
                break
    return replay_memory
