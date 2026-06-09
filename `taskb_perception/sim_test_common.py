"""仿真手动测试共用 — 不含 AppLauncher (避免 import 时重复启动 Isaac)"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

CLASS_COLORS_BGR = {
    "sugar_box": (0, 255, 255),
    "mustard_bottle": (255, 0, 255),
    "banana": (0, 255, 0),
    "unknown": (180, 180, 180),
}


def resolve_policy(project_root: str, policy_arg: str = "") -> str:
    if policy_arg:
        return policy_arg
    for p in [
        os.path.join(project_root, "demo", "policy.pt"),
        os.path.join(os.path.dirname(__file__), "policy.pt"),
    ]:
        if os.path.isfile(p):
            return p
    return ""


class SimpleWalk:
    def __init__(self, device: str, policy_path: str = ""):
        self.device = device
        self.policy = None
        self.mode = "scripted"
        self.vx, self.vy, self.wz = 0.5, 0.0, 0.0
        if policy_path and os.path.isfile(policy_path):
            self.policy = torch.jit.load(policy_path, map_location=device)
            self.policy.eval()
            self.mode = "rl"
            self.env_to_train = torch.tensor(
                [4.0, 2.0, 2.0] * 4, device=device, dtype=torch.float32).view(1, -1)
            self.train_to_env = torch.tensor(
                [0.25, 0.5, 0.5] * 4, device=device, dtype=torch.float32).view(1, -1)
            print(f"[walk] RL policy: {policy_path}", flush=True)
        else:
            print("[walk] scripted gait — copy demo/policy.pt for smooth forward walk", flush=True)

    def set_cmd(self, vx, wz):
        self.vx, self.wz = vx, wz

    def get_action(self, obs, step: int, turn_bias: float = 0.0):
        if self.mode == "rl":
            return self._rl(obs)
        a = torch.zeros(1, 20, dtype=torch.float32, device=self.device)
        t = step * 0.02
        s1, s2 = np.sin(t * 3), np.sin(t * 3 + np.pi)

        def leg(ih, it, ic, s):
            a[0, ih] = 0.25 * s
            a[0, it] = 0.55 + 0.45 * s
            a[0, ic] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1); leg(3, 4, 5, s2)
        leg(6, 7, 8, s2); leg(9, 10, 11, s1)
        if turn_bias:
            a[0, 0] += turn_bias; a[0, 6] += turn_bias
            a[0, 3] -= turn_bias; a[0, 9] -= turn_bias
        return a

    def _rl(self, obs):
        p = obs["proprio"].to(self.device).float()
        leg_pos = p[:, 12:24]
        leg_vel = p[:, 32:44]
        leg_act = p[:, 52:64] * self.env_to_train
        cmd = torch.tensor([[self.vx, self.vy, self.wz]], device=self.device, dtype=p.dtype)
        pol_in = torch.cat([p[:, 3:6] * 0.25, p[:, 9:12], cmd, leg_pos, leg_vel * 0.05, leg_act], dim=-1)
        with torch.inference_mode():
            leg = self.policy(pol_in)
        full = torch.zeros(1, 20, device=self.device, dtype=p.dtype)
        full[:, :12] = leg * self.train_to_env
        return full


class ManualKeyboard:
    def __init__(self, device: str, policy_path: str):
        self.walk = SimpleWalk(device, policy_path)
        self._input = self._kb = self._ki = None
        self.snap = self.quit = False
        self._p_was = False
        try:
            import carb.input
            import omni.appwindow
            self._input = carb.input.acquire_input_interface()
            self._kb = omni.appwindow.get_default_app_window().get_keyboard()
            self._ki = carb.input.KeyboardInput
            print("[keyboard] OK — click Isaac Sim window first", flush=True)
        except Exception as e:
            print(f"[keyboard] unavailable: {e}", flush=True)

    def _down(self, key):
        return self._input is not None and self._input.get_keyboard_value(self._kb, key) > 0

    def poll(self):
        self.snap = False
        if self._ki is None:
            return
        p_key = self._down(self._ki.P)
        if p_key and not self._p_was:
            self.snap = True
        self._p_was = p_key
        if self._down(self._ki.Q):
            self.quit = True

    def get_action(self, obs, step: int):
        self.poll()
        if self._ki is None:
            return torch.zeros(1, 20, dtype=torch.float32, device=self.walk.device)
        w = self._down(self._ki.W)
        s = self._down(self._ki.S)
        a = self._down(self._ki.A)
        d = self._down(self._ki.D)
        if not (w or s or a or d):
            return torch.zeros(1, 20, dtype=torch.float32, device=self.walk.device)

        vx = 0.5 if w else (-0.25 if s else 0.0)
        wz = 0.45 if a else (-0.45 if d else 0.0)
        if w and (a or d):
            vx = 0.35
        tb = 0.5 if a else (-0.5 if d else 0.0)
        if self.walk.mode == "rl":
            self.walk.set_cmd(vx, wz)
            return self.walk.get_action(obs, step)
        return self.walk.get_action(obs, step, turn_bias=tb)


def draw_panel(vis, img, x, y, label):
    h, w = img.shape[:2]
    vis[y:y + h, x:x + w] = img
    cv2.putText(vis, label, (x + 4, y + 16), 0, 0.4, (255, 255, 255), 1)
