import time
from statistics import mean, stdev

from tqdm import tqdm 
from joblib import Parallel, delayed

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from datetime import datetime

import json

import sys
import os

from models.video_recorder import IMAGEIO_AVAILABLE, VideoRecorder

cyborg_version = CYBORG_VERSION
EPISODE_LENGTH = 500

def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)


def load_submission(source: str):
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission.submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission


def evaluate_one_episode(cyborg, wrapped_cyborg, agent, write_to_file, i,tot):
    observations, _ = wrapped_cyborg.reset()
    r = []
    a = []
    o = []
    count = 0
    for j in tqdm(range(EPISODE_LENGTH), desc=f'({i+1}/{tot})'):
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], wrapped_cyborg.action_space(agent_name)
            )
            for agent_name, agent in submission.AGENTS.items()
            if agent_name in wrapped_cyborg.agents
        }
        observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
        done = {
            agent: term.get(agent, False) or trunc.get(agent, False)
            for agent in wrapped_cyborg.agents
        }
        if all(done.values()):
            break
        r.append(mean(rew.values()))

        if write_to_file:
            a.append(
                {
                    agent_name: cyborg.get_last_action(agent_name)
                    for agent_name in wrapped_cyborg.agents
                }       
            )
            o.append(
                {
                    agent_name: observations[agent_name]
                    for agent_name in observations.keys()
                }
            )
    total_reward = sum(r)
    return total_reward, a, o

def run_evaluation_parallel(submission, log_path, max_eps=100, write_to_file=False, seed=None, workers=32):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    envs = []
    for _ in range(workers):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=EPISODE_LENGTH,
        )
        cyborg = CybORG(sg, "sim", seed=seed)
        wrapped_cyborg = submission.wrap(cyborg)
        envs.append((cyborg, wrapped_cyborg))
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    outs = Parallel(prefer='processes', n_jobs=workers)(
        delayed(evaluate_one_episode)(*envs[i % workers], submission.AGENTS, write_to_file, i, max_eps)
        for i in range(max_eps)
    )
    total_reward, actions_log, obs_log = zip(*outs)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")

def run_evaluation(submission, log_path, max_eps=100, write_to_file=False, seed=None):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    total_reward = []
    actions_log = []
    obs_log = []
    for i in tqdm(range(max_eps)):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        count = 0
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            if write_to_file:
                a.append(
                    {
                        agent_name: cyborg.get_last_action(agent_name)
                        for agent_name in wrapped_cyborg.agents
                    }       
                )
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
        total_reward.append(sum(r))

        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")


def run_evaluation_with_video(submission, log_path, max_eps=1, seed=None, fps=10, frame_skip=1):
    """Run evaluation and record video of the environment.

    Parameters
    ----------
    submission : Submission
        The submission to evaluate.
    log_path : str
        Path to save output files and videos.
    max_eps : int
        Number of episodes to record (default=1, since video recording is slow).
    seed : int, optional
        Random seed for reproducibility.
    fps : int
        Frames per second for the output video (default=10).
    frame_skip : int
        Record every N steps (default=1, record every step).
    """
    if not IMAGEIO_AVAILABLE:
        raise ImportError("Video recording requires imageio. Install with: pip install imageio imageio-ffmpeg")

    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)

    print(version_header)
    print(author_header)
    print(f"Recording {max_eps} episode(s) to video with FPS={fps}, frame_skip={frame_skip}")
    print(f"Using agents {submission.AGENTS}")

    if not log_path.endswith("/"):
        log_path += "/"
    print(f"Videos will be saved to {log_path}")

    start = datetime.now()

    total_rewards = []

    for episode in range(max_eps):
        print(f"\n=== Recording Episode {episode + 1}/{max_eps} ===")

        # Reset environment
        observations, _ = wrapped_cyborg.reset()

        # Initialize video recorder after reset (so network is set up properly)
        recorder = VideoRecorder(cyborg, log_path, fps=fps)

        # Capture initial frame
        recorder.capture_frame(step=0, reward=0.0)

        episode_rewards = []

        for step in tqdm(range(EPISODE_LENGTH), desc=f"Episode {episode + 1}"):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }

            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)

            step_reward = mean(rew.values())
            episode_rewards.append(step_reward)

            # Capture frame (with optional frame skipping)
            if (step + 1) % frame_skip == 0:
                recorder.capture_frame(step=step + 1, reward=sum(episode_rewards))

            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
            if all(done.values()):
                # Capture final frame
                recorder.capture_frame(step=step + 1, reward=sum(episode_rewards))
                break

        total_episode_reward = sum(episode_rewards)
        total_rewards.append(total_episode_reward)

        # Save video for this episode
        video_filename = f"episode_{episode + 1}_reward_{total_episode_reward:.2f}.mp4"
        recorder.save_video(video_filename)

        print(f"Episode {episode + 1} finished with reward: {total_episode_reward:.2f}")

    end = datetime.now()
    difference = end - start

    if len(total_rewards) > 1:
        reward_mean = mean(total_rewards)
        reward_stdev = stdev(total_rewards)
    else:
        reward_mean = total_rewards[0] if total_rewards else 0
        reward_stdev = 0

    print(f"\n=== Video Recording Complete ===")
    print(f"Recorded {max_eps} episode(s)")
    print(f"Average reward: {reward_mean:.2f} (std: {reward_stdev:.2f})")
    print(f"Total time: {difference}")
    print(f"Videos saved to: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Evaluation Script")
    parser.add_argument(
        "--append-timestamp",
        action="store_true",
        help="Appends timestamp to output_path",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set the seed for CybORG"
    )

    # Added to speed up evaluation 
    parser.add_argument(
        '--distribute', type=int, default=1, help="How many parallel workers to use"
    )
    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")

    # Video recording options
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of the evaluation (runs single-threaded, slower)"
    )
    parser.add_argument(
        "--video-fps", type=int, default=10,
        help="Frames per second for video output (default: 10)"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Record every N steps to reduce video size (default: 1 = every step)"
    )

    args = parser.parse_args()
    args.output_path = os.path.abspath('tmp')
    args.submission_path = os.path.abspath('')

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    if args.append_timestamp:
        args.output_path += time.strftime("%Y%m%d_%H%M%S") + "/"

    rmkdir(args.output_path)

    submission = load_submission(args.submission_path)

    if args.record_video:
        # Video recording mode (single-threaded)
        run_evaluation_with_video(
            submission,
            log_path=args.output_path,
            max_eps=args.max_eps,
            seed=args.seed,
            fps=args.video_fps,
            frame_skip=args.frame_skip
        )
    elif args.distribute == 1:
        run_evaluation(
            submission, max_eps=args.max_eps, log_path=args.output_path, seed=args.seed
        )
    else: 
        run_evaluation_parallel(
            submission, max_eps=args.max_eps, log_path=args.output_path, seed=args.seed, workers=args.distribute
        )