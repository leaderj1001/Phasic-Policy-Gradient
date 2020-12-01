import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from config import load_args
from model import Actor, Critic


def load_env(args):
    env = gym.make(args.env_name)

    return env


class Memory(object):
    def __init__(self):
        self.data = []

    def _put_data(self, state, next_state, reward, action, log_prob, done):
        self.data.append([state, next_state, reward, action, log_prob, done])

    def _make_batch(self, mode='train'):
        _s, _ns, _r, _a, _p, _d = [], [], [], [], [], []
        for state, next_state, reward, action, prob, done in self.data:
            _s.append(state)
            _ns.append(next_state)
            _r.append([reward / 100.])
            _a.append([action])
            _p.append([prob])
            done_mask = 0. if done else 1.
            _d.append([done_mask])

        if mode == 'train_aux':
            self.data = []

        return torch.tensor(_s).float(), torch.tensor(_ns).float(), torch.tensor(_r), torch.tensor(_a), torch.tensor(_p), torch.tensor(_d)


def _train(actor, critic, p_optimizer, v_optimizer, memory, args):
    state, next_state, reward, action, old_prob, done_mask = memory._make_batch(mode='train')

    actor_losses = 0.
    for _ in range(args.actor_epochs):
        td_target = reward + args.gamma * critic(next_state) * done_mask
        delta = td_target - critic(state)
        delta = delta.detach()

        advantages = torch.zeros([len(delta), 1])
        advantage = 0.
        for idx in reversed(range(len(delta))):
            advantage = args.gamma * args.lmbda * advantage + delta[idx]
            advantages[idx] = advantage

        logits, _ = actor(state)
        new_prob = F.softmax(logits, dim=-1).gather(1, action)
        ratio = torch.exp(torch.log(new_prob) - torch.log(old_prob))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, min=1 - args.clip, max=1 + args.clip) * advantages

        entropy = -(old_prob * torch.log(old_prob)).sum()

        loss = -torch.min(surr1, surr2) + args.entropy_coefficient * entropy

        p_optimizer.zero_grad()
        loss.mean().backward()
        actor_losses += loss.mean().item()
        p_optimizer.step()

    critic_losses = 0.
    for _ in range(args.critic_epochs):
        td_target = reward + args.gamma * critic(next_state) * done_mask

        v_optimizer.zero_grad()
        loss = F.smooth_l1_loss(critic(state) , td_target.detach()) * 0.5
        loss.mean().backward()
        critic_losses += loss.mean().item()
        v_optimizer.step()

    return actor_losses / args.actor_epochs, critic_losses / args.critic_epochs


def _train_aux(actor, critic, p_optimizer, v_optimizer, memory, args):
    state, next_state, reward, action, old_prob, done_mask = memory._make_batch(mode='train_aux')

    aux_losses, critic_losses = 0., 0.
    for _ in range(args.aux_epochs):
        td_target = reward + args.gamma * critic(next_state) * done_mask

        logits, values = actor(state)
        new_prob = F.softmax(logits, dim=-1).gather(1, action)

        aux_loss = F.smooth_l1_loss(values, td_target.detach()) * 0.5
        kl_div = F.kl_div(torch.log(new_prob), torch.log(old_prob), reduction='batchmean') * 1.

        p_optimizer.zero_grad()
        loss = aux_loss + kl_div
        loss.mean().backward()
        aux_losses += loss.mean().item()
        p_optimizer.step()

        v_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(critic(state), td_target.detach()) * 0.5
        critic_loss.mean().backward()
        critic_losses += critic_loss
        v_optimizer.step()

    return aux_losses / args.aux_epochs, critic_losses / args.aux_epochs


def main(args):
    env = load_env(args)
    in_channels, out_channels = env.observation_space.shape[0], env.action_space.n

    actor = Actor(in_channels, out_channels, args)
    critic = Critic(in_channels, args)
    memory = Memory()
    q = deque()
    q.append(0)

    p_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    v_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    if args.cuda:
        actor, critic = actor.cuda(), critic.cuda()

    e_list, a_list, c_list, aux_list, aux_c_list, score_list, cur_score_list = [], [], [], [], [], [], []
    for episode in range(args.max_episodes):
        state = env.reset()
        done = False
        score, step = 0., 0.
        a_loss, c_loss = 0., 0.

        while not done:
            for _ in range(args.n):
                logits, _ = actor(torch.tensor(state).float())
                probs = F.softmax(logits, dim=-1)
                m = Categorical(probs)
                action = m.sample().item()

                next_state, reward, done, info = env.step(action)
                memory._put_data(state, next_state, reward, action, probs[action].item(), done)
                score += 1.
                if done:
                    break

                state = next_state

            a, c = _train(actor, critic, p_optimizer, v_optimizer, memory, args)
            a_loss += a
            c_loss += c
            step += 1
        aux, aux_c = _train_aux(actor, critic, p_optimizer, v_optimizer, memory, args)
        q.append(score)

        a_loss /= step
        c_loss /= step

        e_list.append(episode)
        a_list.append(a_loss)
        c_list.append(c_loss)
        aux_list.append(aux)
        aux_c_list.append(aux_c)
        cur_score_list.append(score)
        score_list.append(np.mean(q))

        if episode % 100 == 0:
            print('[Episode: {0:4d}] avg score: {1:.3f}, cur score: {2}'.format(episode, np.mean(q), score))

            for idx in range(6):
                plt.subplot(3, 2, idx + 1)
                if idx == 0:
                    show = a_list
                    y_label = 'actor loss'
                elif idx == 1:
                    show = c_list
                    y_label = 'critic loss'
                elif idx == 2:
                    show = aux_list
                    y_label = 'aux loss'
                elif idx == 3:
                    show = aux_c_list
                    y_label = 'aux critic loss'
                elif idx == 4:
                    show = cur_score_list
                    y_label = 'current score'
                else:
                    show = score_list
                    y_label = 'mean score'

                plt.plot(e_list, show)
                plt.title(y_label, fontsize=8)
                plt.yticks(fontsize=6)
                plt.xticks(fontsize=6)
            plt.subplots_adjust(left=0.125,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.35)
            plt.savefig('test.png', dpi=300)


if __name__ == '__main__':
    args = load_args()
    main(args)