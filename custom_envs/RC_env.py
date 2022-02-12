import gym
import numpy as np

class RC_env(gym.Env):

    def __init__(self, env_id="Acrobot-v1", RC_seed=0, RC_neurons=512, remove_vel = False, render=False):
        super(RC_env).__init__()

        # import gym
        try:   
            import pybulletgym  # register PyBullet enviroments with open ai gym
        except:
            pass
            # print('No pybulletgym module available')
        self.env = gym.make(env_id)
        self.render = render

        if self.render:
            self.env.render()

        # print('Env observation size:', self.env.observation_space.shape[0])
        input_size = self.env.observation_space.shape[0]

        self.n_reservoir_neurons=RC_neurons
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(self.n_reservoir_neurons,),dtype=np.float64)
        self.action_space = self.env.action_space
        self.reservoir_state = np.zeros((self.n_reservoir_neurons))

        if remove_vel:
            self.idx=[0,1,2,3]
            input_size = len(self.idx)
        else:
            self.idx=np.arange(self.env.observation_space.shape[0])

        from scipy import sparse
        print('RC network has %i neurons.' % self.n_reservoir_neurons)
        self.W_in, self.W_res = self.generate_Ws(
                                    n_reservoir_neurons = self.n_reservoir_neurons, 
                                    bounds = [-1,1], 
                                    input_size = input_size, 
                                    density=0.1, 
                                    input_density=0.1,
                                    spectral_radius=0.90,
                                    seed=RC_seed, )



        
    def seed(self, seed=0):
        # Seed is set separately. This function does nothing. 
        return self.env.seed(seed)

    def reservoir_step(self, rod_state):
        x = self.W_res @ self.reservoir_state[:,np.newaxis] + self.W_in @ rod_state[:,np.newaxis]
        self.reservoir_state = np.squeeze(np.tanh(x))
        state_output = np.hstack([self.reservoir_state,]) * 0.01
        return state_output

    def reset(self):
        rod_state = self.env.reset()
        self.reservoir_state = np.zeros((self.n_reservoir_neurons))

        for _ in range(10):
            # state_output = self.reservoir_step(rod_state[self.idx])
            state_output = self.reservoir_step(rod_state[self.idx])
        return state_output
    
    def step(self, action):
        rod_state, reward, done, info = self.env.step(action)
        state_output = self.reservoir_step(rod_state[self.idx])
        if self.render:
            self.env.render()
        return state_output, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def generate_Ws(
        self,
        n_reservoir_neurons,
        bounds,
        input_size,
        density=0.10, 
        input_density=1.0,
        spectral_radius=0.9, 
        seed=101, 
        ):

        from scipy import sparse

        # import numpy as np
        # from scipy import sparse
        # import scipy
        np.random.seed(seed)
        # Sample W_rin from a random uniform distribution
        # W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))
        W_in = np.random.normal(0, 0.5, (n_reservoir_neurons, input_size))
        # W_in = np.random.gamma(2,0.2, (n_reservoir_neurons, input_size))

        mask = sparse.rand(n_reservoir_neurons, input_size, density=input_density)
        mask = np.array(mask.todense())
        mask[np.where(mask > 0)] = 1
        W_in = W_in * mask
        # np.save('W_in.npy', W_in)
        # Sample W_reservoir from a random uniform distribution
        # W_reservoir = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, n_reservoir_neurons))
        W_reservoir = np.random.normal(0, 0.5, (n_reservoir_neurons, n_reservoir_neurons))
        # W_reservoir = np.random.gamma(2,0.2, (n_reservoir_neurons, n_reservoir_neurons))
        # Create a mask to make W_reservoir a sparse matrix with a set density
        mask = sparse.rand(n_reservoir_neurons, n_reservoir_neurons, density=density)
        mask = np.array(mask.todense())
        mask[np.where(mask > 0)] = 1
        W_reservoir = W_reservoir * mask
        # Set the spectral radius of W_reservoir
        E, _ = np.linalg.eig(W_reservoir)
        e_max = np.max(np.abs(E))
        W_reservoir *= spectral_radius/np.abs(e_max) if np.abs(e_max) != 0 else 0
        # Save W_in and W_reservoir to file
        # np.save('W_reservoir.npy', W_reservoir)

        return W_in, W_reservoir

