import os

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video generation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import SUBNET


class VideoRecorder:
    """Records environment state as video frames using network visualization."""

    def __init__(self, cyborg, output_path, fps=10, dpi=100):
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is required for video recording. Install with: pip install imageio imageio-ffmpeg")

        self.cyborg = cyborg
        self.output_path = output_path
        self.fps = fps
        self.dpi = dpi
        self.frames = []
        self.env = cyborg.environment_controller

        # Initialize network visualization
        self._setup_network()

    def _setup_network(self):
        """Setup the network graph visualization."""
        env_netmap = self.env.state.link_diagram.copy()
        self.node_label_mapping = self._get_node_label_mapping(env_netmap)
        self.host_nodes = self._get_host_nodes(env_netmap)
        self.host_interfaces = list(env_netmap.edges()).copy()

        # Get initial agents
        all_session_agents = {"blue": [], "red": []}
        all_host_sessions = []

        for hostname, host in self.env.state.hosts.items():
            for agent, sids in host.sessions.items():
                if sids:
                    if "blue" in agent:
                        all_session_agents["blue"].append(agent)
                        all_host_sessions.append((hostname, agent))
                    elif "red" in agent:
                        all_session_agents["red"].append(agent)
                        all_host_sessions.append((hostname, agent))

        all_session_agents["blue"] = list(set(all_session_agents["blue"]))
        all_session_agents["red"] = list(set(all_session_agents["red"]))

        env_netmap.add_nodes_from(all_session_agents['blue'])
        env_netmap.add_edges_from(all_host_sessions)

        self.pos = self._set_network_positions(env_netmap)
        self.base_network = env_netmap

    def _get_node_label_mapping(self, env_netmap):
        """Create node labels for visualization."""
        node_label_mapping = {}
        for node in env_netmap._node.keys():
            if "user_host" not in node and "server_host" not in node:
                new_node_label = ""
                if "restricted_zone" in node:
                    new_node_label = "RZ"
                elif "operational_zone" in node:
                    new_node_label = "OZ"
                elif "contractor_network" in node:
                    new_node_label = "CN"
                elif "public_access_zone" in node:
                    new_node_label = "PAZ"
                elif "admin_network" in node:
                    new_node_label = "AN"
                elif "office_network" in node:
                    new_node_label = "ON"
                else:
                    new_node_label = "Internet Root"
                    node_label_mapping[node] = new_node_label
                    continue

                if "_a_" in node:
                    new_node_label += "A"
                elif "_b_" in node:
                    new_node_label += "B"

                node_label_mapping[node] = new_node_label
        return node_label_mapping

    def _get_host_nodes(self, env_netmap):
        """Categorize host nodes by type."""
        all_host_nodes = list(env_netmap.nodes()).copy()
        return {
            'servers': [h for h in all_host_nodes if 'server' in h],
            'users': [h for h in all_host_nodes if 'user' in h],
            'other': [h for h in all_host_nodes if 'user' not in h and 'server' not in h]
        }

    def _set_network_positions(self, env_netmap):
        """Calculate positions for network nodes."""
        positions = nx.spring_layout(env_netmap, seed=2, iterations=300)

        # Get subnet host positions for red agents
        subnet_host_positions = {}
        for subnet in self.env.state.subnets.values():
            subnet_name = subnet.name
            subnet_host_positions[subnet_name] = []
            for host_name in self.env.state.hosts.keys():
                if subnet_name in host_name:
                    subnet_host_positions[subnet_name].append(list(positions[host_name]))

        red_agent_allowed_subnets = [
            [SUBNET.CONTRACTOR_NETWORK.value],
            [SUBNET.RESTRICTED_ZONE_A.value],
            [SUBNET.OPERATIONAL_ZONE_A.value],
            [SUBNET.RESTRICTED_ZONE_B.value],
            [SUBNET.OPERATIONAL_ZONE_B.value],
            [SUBNET.PUBLIC_ACCESS_ZONE.value, SUBNET.ADMIN_NETWORK.value, SUBNET.OFFICE_NETWORK.value]
        ]

        for r in range(6):
            red_agent_name = f'red_agent_{r}'
            if len(red_agent_allowed_subnets[r]) == 1:
                subnet_positions = subnet_host_positions.get(red_agent_allowed_subnets[r][0], [[0, 0]])
                positions[red_agent_name] = np.array(subnet_positions).mean(axis=0) * 1.15
            else:
                combined = []
                for s in red_agent_allowed_subnets[r]:
                    combined.extend(subnet_host_positions.get(s, []))
                if combined:
                    positions[red_agent_name] = np.array(combined).mean(axis=0) * 1.15
                else:
                    positions[red_agent_name] = np.array([0, 0])

        return positions

    def _get_current_state(self):
        """Get the current network state for visualization."""
        compromised_hosts = []
        red_root_nodes = []
        active_red_agents = []
        active_blue_agents = []
        host_sessions = []
        agent_labels = {}

        # Get root sessions for each red agent
        agent_root_sessions = {}
        for agent, sessions in self.env.state.sessions.items():
            if 'red' in agent:
                agent_root_sessions[agent] = []
                for sid, sess in sessions.items():
                    if sess.username == "root":
                        agent_root_sessions[agent].append(sid)

        for hostname, host in self.env.state.hosts.items():
            for agent, sids in host.sessions.items():
                if sids:
                    if "blue" in agent:
                        if agent not in active_blue_agents:
                            active_blue_agents.append(agent)
                            agent_labels[agent] = "B" + agent.split("_")[-1]
                        host_sessions.append((hostname, agent))
                    elif "red" in agent:
                        if agent not in active_red_agents:
                            active_red_agents.append(agent)
                            agent_labels[agent] = "R" + agent.split("_")[-1]
                        host_sessions.append((hostname, agent))
                        compromised_hosts.append(hostname)

                        for sid in sids:
                            if agent in agent_root_sessions and sid in agent_root_sessions[agent]:
                                red_root_nodes.append(hostname)

        return {
            'compromised_hosts': list(set(compromised_hosts)),
            'red_root_nodes': list(set(red_root_nodes)),
            'active_red': list(set(active_red_agents)),
            'active_blue': list(set(active_blue_agents)),
            'host_sessions': host_sessions,
            'agent_labels': agent_labels
        }

    def capture_frame(self, step=None, reward=None):
        """Capture current state as a frame."""
        state = self._get_current_state()

        fig, ax = plt.subplots(figsize=(14, 10))
        step_str = str(step) if step is not None else '?'
        reward_str = f"{reward:.2f}" if reward is not None else 'N/A'
        ax.set_title(f"Step: {step_str} | Reward: {reward_str}", fontsize=14)

        # Build current network
        current_network = self.base_network.copy()
        for agent in state['active_red']:
            if agent not in current_network:
                current_network.add_node(agent)

        # Draw nodes
        nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                              nodelist=self.host_nodes['users'], node_size=200,
                              node_color='#C0C0C0', alpha=0.9, node_shape='o')
        nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                              nodelist=self.host_nodes['servers'], node_size=200,
                              node_color='#C0C0C0', alpha=0.9, node_shape='s')
        nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                              nodelist=self.host_nodes['other'], node_size=400,
                              node_color='#C0C0C0', alpha=0.9, node_shape='H')

        # Draw agents
        red_agents_in_pos = [a for a in state['active_red'] if a in self.pos]
        blue_agents_in_pos = [a for a in state['active_blue'] if a in self.pos]

        if red_agents_in_pos:
            nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                                  nodelist=red_agents_in_pos, node_size=200,
                                  node_color='#EE4B2B', node_shape='^')
        if blue_agents_in_pos:
            nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                                  nodelist=blue_agents_in_pos, node_size=200,
                                  node_color='#0096FF', node_shape='^')

        # Draw edges
        nx.draw_networkx_edges(current_network, self.pos, ax=ax,
                              edgelist=self.host_interfaces)
        valid_sessions = [(h, a) for h, a in state['host_sessions'] if h in self.pos and a in self.pos]
        if valid_sessions:
            nx.draw_networkx_edges(current_network, self.pos, ax=ax,
                                  edgelist=valid_sessions, style=':')

        # Draw labels
        nx.draw_networkx_labels(current_network, self.pos, ax=ax,
                               labels=self.node_label_mapping, font_size=8)
        valid_agent_labels = {k: v for k, v in state['agent_labels'].items() if k in self.pos}
        if valid_agent_labels:
            nx.draw_networkx_labels(current_network, self.pos, ax=ax,
                                   labels=valid_agent_labels, font_size=8)

        # Draw compromised hosts
        compromised_in_pos = [h for h in state['compromised_hosts'] if h in self.pos]
        root_in_pos = [h for h in state['red_root_nodes'] if h in self.pos]

        if compromised_in_pos:
            nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                                  nodelist=compromised_in_pos, node_size=200,
                                  node_color='#FFA500', alpha=0.8)
        if root_in_pos:
            nx.draw_networkx_nodes(current_network, self.pos, ax=ax,
                                  nodelist=root_in_pos, node_size=200,
                                  node_color='#EE4B2B', alpha=0.8)

        ax.legend(['user host', 'server host', 'router', 'red agent', 'blue agent',
                  'host link', 'session link', 'user compromised', 'root compromised'],
                 loc='upper left', fontsize=8)

        # Convert to image array
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[:, :, :3]  # Convert RGBA to RGB
        self.frames.append(image.copy())

        plt.close(fig)

    def save_video(self, filename):
        """Save captured frames as a video file."""
        if not self.frames:
            print("No frames to save!")
            return

        video_path = os.path.join(self.output_path, filename)

        # Ensure frames have even dimensions (required by most video codecs)
        processed_frames = []
        for frame in self.frames:
            h, w = frame.shape[:2]
            # Crop to even dimensions if needed
            new_h = h - (h % 2)
            new_w = w - (w % 2)
            processed_frames.append(frame[:new_h, :new_w])

        # Convert frames to numpy array stack
        frames_array = np.stack(processed_frames)

        saved = False

        # Method 1: Try using imageio-ffmpeg with explicit format
        try:
            # Force ffmpeg format
            imageio.mimwrite(
                video_path,
                frames_array,
                format='FFMPEG',
                fps=self.fps,
                codec='libx264',
                pixelformat='yuv420p',
                output_params=['-crf', '23']  # Quality setting
            )
            saved = True
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"FFMPEG method failed: {e}")

        # Method 2: If MP4 fails, save as GIF
        if not saved:
            gif_path = video_path.replace('.mp4', '.gif')
            print(f"Falling back to GIF format: {gif_path}")
            try:
                # For GIF, reduce frame rate and subsample frames
                skip = max(1, len(processed_frames) // 100)  # Max ~100 frames for GIF
                gif_frames = processed_frames[::skip]
                imageio.mimsave(gif_path, gif_frames, duration=1000//(self.fps or 10), loop=0)
                video_path = gif_path
                saved = True
                print(f"GIF saved to: {gif_path}")
            except Exception as e:
                print(f"GIF also failed: {e}")

        if not saved:
            print("Failed to save video! Make sure imageio-ffmpeg is installed:")
            print("  pip install imageio-ffmpeg")

        self.frames = []  # Clear frames after saving

    def reset(self):
        """Reset for a new episode."""
        self.frames = []
        self._setup_network()


