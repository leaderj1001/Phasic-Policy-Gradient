import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--max_episodes', type=int, default=5000)
    parser.add_argument('--cuda', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--actor_epochs', type=int, default=1)
    parser.add_argument('--critic_epochs', type=int, default=1)
    parser.add_argument('--aux_epochs', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--entropy_coefficient', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=0.2)

    args = parser.parse_args()

    return args
