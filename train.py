import os
# import gym
import json
import yaml
import time
import torch
import numpy as np
from tqdm import trange
import maml_rl.envs
from maml_rl.envs.laser import all_envs
from maml_rl.envs.laser.wrappers import gym
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
np.set_printoptions(precision=3, sign=' ', floatmode="fixed", linewidth=2000)

def get_time(start_time):
        delta = time.gmtime(time.time()-start_time)
        return f"{delta.tm_mday-1}-{time.strftime('%H:%M:%S', delta)}"

LOG_DIR = "./logging/logs/pt"

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.env_name in all_envs:
        config.update({"env-name":args.env_name+"-v0", "env-kwargs":{}, "fast-batch-size":16, "num-batches":2000, "meta-batch-size":1})
    if args.output_folder is not None:
        if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')
        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model_name = "maml"
    os.makedirs(f"{LOG_DIR}/{model_name}/{config['env-name']}/", exist_ok=True)
    run_num = len([n for n in os.listdir(f"{LOG_DIR}/{model_name}/{config['env-name']}/")])
    log_path = f"{LOG_DIR}/{model_name}/{config['env-name']}/logs_{run_num}.txt"

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()
    policy = get_policy_for_env(env, hidden_sizes=config['hidden-sizes'], nonlinearity=config['nonlinearity'])
    policy.share_memory()
    baseline = LinearFeatureBaseline(get_input_size(env))
    sampler = MultiTaskSampler(config['env-name'], env_kwargs=config.get('env-kwargs', {}), batch_size=config['fast-batch-size'], policy=policy, baseline=baseline, env=env, seed=args.seed, num_workers=args.num_workers)
    metalearner = MAMLTRPO(policy, fast_lr=config['fast-lr'], first_order=config['first-order'], device=args.device)
    num_iterations = 0
    total_rewards = []
    start = time.time()
    step = 0
    # for batch in range(config['num-batches']+1):
    while step <= 500000:
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks, num_steps=config['num-steps'], fast_lr=config['fast-lr'], gamma=config['gamma'], gae_lambda=config['gae-lambda'], device=args.device)
        logs = metalearner.step(*futures, max_kl=config['max-kl'], cg_iters=config['cg-iters'], cg_damping=config['cg-damping'], ls_max_steps=config['ls-max-steps'], ls_backtrack_ratio=config['ls-backtrack-ratio'])
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks, num_iterations=num_iterations, train_returns=get_returns(train_episodes[0]), valid_returns=get_returns(valid_episodes))
        # Save policy
        old_step = step
        step += train_episodes[0][0].lengths[0]
        if old_step==0 or step//1000 > old_step//1000:
            rollouts = logs["valid_returns"][0]
            reward = np.mean(rollouts, -1)
            ep = step//1000
            total_rewards.append(reward)
            string = f"Step: {1000*ep:7d}, Reward: {total_rewards[-1]:9.3f} [{np.std(rollouts):8.3f}], Avg: {np.mean(total_rewards, axis=0):9.3f} ({0.0:.3f}) <{get_time(start)}> ({{}})"
            print(string)
            with open(log_path, "a+") as f:
                f.write(f"{string}\n")

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description='Reinforcement learning with Model-Agnostic Meta-Learning (MAML) - Train')
    parser.add_argument("--env_name", type=str, default=None, choices=all_envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(all_envs), metavar="env_name")
    parser.add_argument('--config', type=str, default="configs/maml/halfcheetah-vel.yaml", help='path to the configuration file.')
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None, help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true', help='use cuda (default: false, use cpu). WARNING: Full upport for cuda is not guaranteed. Using CPU is encouraged.')
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    main(args)
