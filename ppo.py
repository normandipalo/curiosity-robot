import tensorflow as tf
if __name__ == "__main__":
    tf.enable_eager_execution()

import numpy as np
import gym
import roboschool
import matplotlib.pyplot as plt

layers = tf.keras.layers

class Actor(tf.keras.Model):
    def __init__(self, n_inp, n_out, hidden, init_w = None, bias = None, use_bias = True):
        super(Actor, self).__init__()
        if init_w is None : init_w = 1e-3
        if bias is None : 
            init_b = init_w
        else:
            init_b = bias
        self.n_inp = n_inp
        self.n_out = n_out
        
        if use_bias:
            self.dense1 = layers.Dense(hidden, activation = "tanh", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.dense2 = layers.Dense(hidden, activation = "tanh", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.mu = layers.Dense(n_out, activation = "tanh", kernel_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), bias_initializer = tf.keras.initializers.RandomUniform(minval = -init_b, maxval = init_b), kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.std = layers.Dense(n_out, activation = "softplus", kernel_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), bias_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
        else:
            self.dense1 = layers.Dense(hidden, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.dense2 = layers.Dense(hidden, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
            self.mu = layers.Dense(n_out, activation = "tanh", kernel_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), use_bias= use_bias)
            self.std = layers.Dense(n_out, activation = "softplus", kernel_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), bias_initializer = tf.keras.initializers.RandomUniform(minval = -init_w, maxval = init_w), kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
        
        
    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = self.dense1(x)
        x = self.dense2(x)
        mu = self.mu(x)
        std = self.std(x) + 0.05
        return mu, std
        

class Critic(tf.keras.Model):
    def __init__(self, n_inp, n_out, hidden):
        super(Critic, self).__init__()
        self.n_inp = n_inp
        self.n_out = n_out
        self.dense1 = layers.Dense(hidden, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
        self.dense2 = layers.Dense(hidden, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
        self.dense3 = layers.Dense(1, activation = None, kernel_regularizer=tf.keras.regularizers.l2(l=0.0))
        
    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = self.dense1(x)
        x = self.dense2(x)
        value = self.dense3(x)
        return value


def value_loss(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma):
    states = np.vstack(np.array(states))
    actions = np.vstack(np.array(actions))
    next_states = np.vstack(np.array(next_state))
    rewards = np.vstack(np.array(rewards))
    next_state = next_state.reshape((1,-1))
    if done:
        reward_sum = 0.
    else:
        reward_sum = critic_target(tf.convert_to_tensor(next_state)).numpy()
        
    discounted_rewards = []
    for reward in rewards[::-1]:
        reward_sum = reward + gamma*reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    
    values = critic(tf.convert_to_tensor(states))
    
    advantage = tf.convert_to_tensor(np.array(discounted_rewards).reshape((-1,1))) - values
    
    value_loss = (advantage**2)/2
    value_loss = tf.reduce_mean(value_loss)
    return value_loss



def ppo_iter(mini_batch_size, states, actions, probs_acts, rewards, next_states):
    states = np.vstack(np.array(states))
    actions = np.vstack(np.array(actions))
    probs_acts = np.vstack(np.array(probs_acts))
    next_states = np.vstack(np.array(next_states))
    rewards = np.vstack(np.array(rewards))
    batch_size = states.shape[0]
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], probs_acts[rand_ids, :], rewards[rand_ids, :], next_states[rand_ids, :]
    
        
def actor_update(epochs, mini_batch_size, states, actions, probs_acts, rewards, next_states, actor, critic, critic_target, gamma, ent_level, opt, want_grads = False):
    clip_p = 0.2
    
    act_losses = []
    for state_b, action_b, probs_b, rews_b, next_state_b in ppo_iter(mini_batch_size, states, actions, probs_acts, rewards, next_states):
        for _ in range(epochs):    
            with tf.GradientTape() as tape:
                new_probs_b = []
                entropies = []
                ratio_b = []
                for st, ac, p in zip(state_b, action_b, probs_b):
                    mu, sigma = actor(st.reshape((1,-1)))
                    dist = tf.distributions.Normal(mu, sigma)
                    entropies.append(dist.entropy())
                    new_prob = tf.exp(dist.log_prob(ac))
                    p+=1e-10

                    ratio_b.append( tf.reduce_prod(new_prob/p))

                    new_probs_b.append(new_prob)
                entropies = tf.reduce_sum(entropies)
                new_probs_b = tf.reshape(tf.stack(new_probs_b), shape = (-1,1))

                advantage_b = rews_b + gamma*critic_target(next_state_b) - critic_target(state_b)
                advantage_b = tf.clip_by_value(advantage_b, -1, 1)

                ratio_b = tf.reshape(tf.stack(ratio_b), shape = (-1,1))
                surr1 = ratio_b*tf.stop_gradient(advantage_b)
                clipped = tf.clip_by_value(ratio_b, 1.0 - clip_p, 1.0 + clip_p)
                surr2 = clipped*tf.stop_gradient(advantage_b)
                act_loss = tf.minimum(surr1, surr2)
                act_loss = -tf.reduce_mean(act_loss) - entropies*ent_level
                act_losses.append(act_loss)
            grads = tape.gradient(act_loss, actor.variables)
            grads = [tf.clip_by_norm(g, 1.) for g in grads]
            opt.apply_gradients(zip(grads, actor.variables))
            act_losses_sum = tf.reduce_mean(act_losses)
            if want_grads : return act_losses_sum, grads
            return act_losses_sum
            



def critic_update(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma, opt):
    tau = 0.01
    with tf.GradientTape() as tape:
        loss = value_loss(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma)
        
    grads = tape.gradient(loss, critic.variables)
    grads = [tf.clip_by_norm(g, 1.) for g in grads]
    opt.apply_gradients(zip(grads, critic.variables))
    for p, p_t in zip(critic.variables, critic_target.variables):
        tf.assign(p_t, (p_t*(1-tau) + p*tau))
    return loss


def validate():
    tot_rews = 0
    for ep in range(10):
        rew_ep = 0
        state = env.reset()
        for st in range(500):
            state = np.array(state, dtype = np.float64)
            mu, sigma = actor(state.reshape((1,-1)))
            dist = tf.distributions.Normal(mu, sigma)
            action = np.array(dist.mean().numpy(), dtype = np.float64)
            next_state, reward, done, _ = env.step(action[0])
            next_state = np.array(next_state, dtype = np.float64)

            rew_ep+= reward
            if done:
                break
        tot_rews += rew_ep
    return tot_rews/10
