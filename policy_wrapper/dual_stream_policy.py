"""
Dual-Stream VTLA Policy wrapper for UniVTAC evaluation
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add VTLA_design to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'VTLA_design'))

from models.dual_stream import DualStreamVTLAPolicy


class DualStreamPolicy(nn.Module):
    """Wrapper for dual-stream VTLA policy compatible with UniVTAC evaluation"""

    def __init__(self, args_override):
        super().__init__()

        # Build dual-stream model
        self.model = DualStreamVTLAPolicy(args_override)
        self.model.eval()

        self.kl_weight = args_override.get("kl_weight", 10.0)
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, cam_image, tac_image, actions=None, is_pad=None):
        """Forward pass - compatible with ACT interface"""
        if actions is not None:  # training time
            # This won't be used in evaluation, but keep for compatibility
            loss_dict = self.model(
                qpos=qpos,
                cam_image=cam_image,
                tac_image=tac_image,
                actions=actions,
                is_pad=is_pad
            )
            return loss_dict
        else:  # inference time
            # Sample from the model
            a_hat, _ = self.model.forward(
                qpos=qpos,
                cam_image=cam_image,
                tac_image=tac_image,
                actions=None,
                is_pad=None
            )
            return a_hat

    def configure_optimizers(self):
        # Not used in evaluation
        return None


class DualStream:
    """Main policy class for evaluation - compatible with UniVTAC"""

    def __init__(self, args_override=None):
        if args_override is None:
            args_override = {
                "kl_weight": 10.0,
                "device": "cuda:0",
            }

        self.policy = DualStreamPolicy(args_override)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", True)
        self.num_queries = args_override.get("chunk_size", 50)
        self.state_dim = args_override.get("state_dim", 9)
        self.max_timesteps = 3000
        self.camera_names = args_override.get("camera_names", ["cam_high"])
        self.tactile_names = args_override.get("tactile_names", ["tac_left", "tac_right"])

        # Query frequency
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0

        # Load checkpoint and stats
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy checkpoint
            ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")

            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                loading_status = self.policy.load_state_dict(state_dict, strict=False)
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        """Get action from observation - main evaluation interface"""
        if obs is None:
            return None

        # Normalize qpos
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare camera images
        if len(self.camera_names) > 0:
            cam_image = []
            for cam_name in self.camera_names:
                cam_image.append(obs[cam_name])
            cam_image = torch.stack(cam_image, dim=0).to(self.device).unsqueeze(0)
        else:
            cam_image = torch.tensor([]).to(self.device)

        # Prepare tactile images
        if len(self.tactile_names) > 0:
            tac_image = []
            for tac_name in self.tactile_names:
                tac_image.append(obs[tac_name])
            tac_image = torch.stack(tac_image, dim=0).to(self.device).unsqueeze(0)
        else:
            tac_image = torch.tensor([]).to(self.device)

        with torch.no_grad():
            # Get action sequence from model
            all_actions = self.policy(qpos, cam_image, tac_image)

            if self.temporal_agg:
                # Temporal aggregation
                all_actions = all_actions.squeeze(0).cpu().numpy()
                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = torch.from_numpy(all_actions).to(self.device)

                # Compute weighted average
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Exponential weighting
                exp_weights = torch.exp(-torch.arange(len(actions_for_curr_step)).to(self.device))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.unsqueeze(dim=1)

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                raw_action = raw_action.squeeze(0).cpu().numpy()
            else:
                # No temporal aggregation - use first action
                raw_action = all_actions[0, 0].cpu().numpy()

        # Denormalize and return
        self.t += 1
        action = self.post_process(raw_action)
        return action

    def reset(self):
        """Reset policy state"""
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions.zero_()


# Policy class - main interface for UniVTAC
Policy = DualStream
