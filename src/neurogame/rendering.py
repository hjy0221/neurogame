from __future__ import annotations

from math import cos, pi, sin

import numpy as np

from neurogame.brain import BrainSnapshot
from neurogame.config import EnvConfig, RenderConfig
from neurogame.environment import FoodWorld
from neurogame.reinforcement import LearningSnapshot


class PygameRenderer:
    def __init__(self, pygame_module, env_config: EnvConfig, render_config: RenderConfig, brain):
        self.pg = pygame_module
        self.env_config = env_config
        self.render_config = render_config
        total_width = env_config.width + render_config.sidebar_width
        self.screen = self.pg.display.set_mode((total_width, env_config.height))
        self.pg.display.set_caption("NeuroGame v0.1")
        self.font = self.pg.font.SysFont("Menlo", 15)
        self.small_font = self.pg.font.SysFont("Menlo", 12)
        self.input_groups = [
            np.arange(0, 4), np.arange(4, 8), np.arange(8, 11), np.array([11]),
            np.arange(12, 15), np.arange(15, 18)
        ]
        self.hidden_groups = [group for group in np.array_split(brain.hidden_idx, 6) if len(group)]
        self.input_links = np.array([[np.abs(brain.input_weights[np.ix_(a, b)]).sum()
                                     for b in self.hidden_groups] for a in self.input_groups])
        self.output_links = np.array([np.abs(brain.motor_weights[group]).sum(axis=0)
                                      for group in self.hidden_groups])

    def draw(
        self,
        world: FoodWorld,
        snapshot: BrainSnapshot,
        observation: np.ndarray,
        paused: bool,
        plasticity_enabled: bool,
        rl_enabled: bool,
        learning: LearningSnapshot,
    ) -> None:
        pg = self.pg
        cfg = self.env_config
        self.screen.fill((13, 17, 21))

        game_rect = pg.Rect(0, 0, cfg.width, cfg.height)
        pg.draw.rect(self.screen, (19, 25, 30), game_rect)

        entry_top, entry_bottom = world.exit_bounds
        entry_height = max(12, int(entry_bottom - entry_top - 10))
        entry_y = int(entry_top + 5)
        pg.draw.rect(self.screen, (47, 118, 164), pg.Rect(25, entry_y, 25, entry_height), border_radius=4)
        pg.draw.rect(self.screen, (55, 171, 109), pg.Rect(670, entry_y, 25, entry_height), border_radius=4)
        label_y = int((entry_top + entry_bottom) / 2 - 7)
        self._text("IN", 27, label_y, (225, 241, 250), self.small_font)
        self._text("OUT", 667, label_y, (225, 250, 233), self.small_font)

        for wall in world.walls:
            rect = pg.Rect(int(wall.x), int(wall.y), int(wall.width), int(wall.height))
            pg.draw.rect(self.screen, (79, 91, 101), rect, border_radius=3)
            pg.draw.line(self.screen, (130, 145, 156), rect.topleft, rect.topright, 2)

        for food in world.food[world.food_active]:
            pg.draw.circle(self.screen, (104, 211, 145), food.astype(int), int(cfg.food_radius))
            pg.draw.circle(self.screen, (25, 70, 44), food.astype(int), int(cfg.food_radius), 1)

        agent = world.agent
        pos = (int(agent.x), int(agent.y))
        self._draw_bug(agent)
        pg.draw.rect(self.screen, (43, 50, 57), game_rect, 2)

        self._draw_sidebar(
            world, snapshot, observation, paused, plasticity_enabled, rl_enabled, learning
        )
        pg.display.flip()

    def _draw_bug(self, agent) -> None:
        """Draw a readable insect avatar aligned with the agent heading."""
        pg = self.pg
        heading = agent.heading
        forward = np.array([cos(heading), sin(heading)], dtype=float)
        side = np.array([-forward[1], forward[0]], dtype=float)
        center = np.array([agent.x, agent.y], dtype=float)

        def point(offset):
            value = center + forward * offset[0] + side * offset[1]
            return int(value[0]), int(value[1])

        # Six legs give the agent a clear insect silhouette at game scale.
        for offset in (-8, 0, 8):
            for sign in (-1, 1):
                hip = point((offset, sign * 5))
                knee = point((offset - 3, sign * 14))
                foot = point((offset - 7, sign * 17))
                pg.draw.line(self.screen, (111, 159, 126), hip, knee, 2)
                pg.draw.line(self.screen, (111, 159, 126), knee, foot, 2)

        for offset, radius in ((-8, 6), (0, 8), (9, 10)):
            pg.draw.circle(self.screen, (196, 226, 112), point((offset, 0)), radius)
            pg.draw.circle(self.screen, (55, 100, 72), point((offset, 0)), radius, 1)

        head = np.array(point((13, 0)), dtype=float)
        for sign in (-1, 1):
            base = np.array(point((16, sign * 4)), dtype=float)
            tip = np.array(point((25, sign * 11)), dtype=float)
            pg.draw.line(self.screen, (235, 205, 116), base.astype(int), tip.astype(int), 1)
            pg.draw.circle(self.screen, (255, 211, 92), tip.astype(int), 2)
            eye = np.array(point((15, sign * 4)), dtype=float)
            pg.draw.circle(self.screen, (35, 43, 38), eye.astype(int), 2)

    def _draw_sensor_rays(self, world: FoodWorld, observation: np.ndarray) -> None:
        pg = self.pg
        agent = world.agent
        ray_angles = np.linspace(-pi * 0.75, pi * 0.75, 8, dtype=np.float32)
        for value, rel_angle in zip(observation[:8], ray_angles):
            length = 40 + 150 * float(value)
            end = (
                int(agent.x + cos(agent.heading + float(rel_angle)) * length),
                int(agent.y + sin(agent.heading + float(rel_angle)) * length),
            )
            color = (35, int(80 + 120 * value), int(95 + 95 * value))
            pg.draw.line(self.screen, color, (int(agent.x), int(agent.y)), end, 1)

    def _draw_sidebar(
        self,
        world: FoodWorld,
        snapshot: BrainSnapshot,
        observation: np.ndarray,
        paused: bool,
        plasticity_enabled: bool,
        rl_enabled: bool,
        learning: LearningSnapshot,
    ) -> None:
        pg = self.pg
        x0 = self.env_config.width
        panel = pg.Rect(x0, 0, self.render_config.sidebar_width, self.env_config.height)
        pg.draw.rect(self.screen, (245, 248, 250), panel)
        pg.draw.line(self.screen, (60, 69, 77), (x0, 0), (x0, self.env_config.height), 2)

        rows = [
            f"NeuroGame / {len(snapshot.spikes)} neurons",
            f"food: {world.food_eaten}/{world.config.food_count}   reward: {world.total_reward:.2f}",
            f"goal: {'EXIT' if not np.any(world.food_active) else 'COLLECT ALL FOOD'}",
            f"maze exits: {world.exits_completed}",
            f"maze level: {world.maze_level}   grid: {world.maze_columns}x{world.maze_rows}",
            f"spikes: {snapshot.mean_rate:.1%}   {'paused' if paused else 'running'}",
            f"plasticity: {'on' if plasticity_enabled else 'off'}",
            f"RL: {'learning' if rl_enabled else 'off'}   updates: {learning.updates}",
            f"value: {learning.value:+.3f}   TD: {learning.td_error:+.3f}   explore: {learning.exploration:.2f}",
        ]
        y = 18
        for text in rows:
            self._text(text, x0 + 18, y, (39, 48, 58), self.font)
            y += 20

        self._draw_network(snapshot, observation)

        help_text = "Space pause / R reset / P plasticity / L RL / Esc"
        self._text(help_text, x0 + 18, self.env_config.height - 28, (128, 140, 151), self.small_font)

    def _draw_bars(self, values: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        pg = self.pg
        for idx, value in enumerate(values):
            bar_width = int(width * float(np.clip(value, 0.0, 1.0)))
            rect = pg.Rect(x, y + idx * (height + 4), bar_width, height)
            pg.draw.rect(self.screen, color, rect)
            pg.draw.rect(self.screen, (55, 63, 70), pg.Rect(x, y + idx * (height + 4), width, height), 1)

    def _draw_network(self, snapshot: BrainSnapshot, observation: np.ndarray) -> None:
        pg = self.pg
        x0 = self.env_config.width
        columns = [x0 + 70, x0 + 260, x0 + 450]
        values = [np.array([observation[g].mean() for g in self.input_groups]),
                  np.array([np.clip(snapshot.activity_trace[g].mean() / 5, 0, 1) for g in self.hidden_groups]),
                  (snapshot.motor_values + 1) / 2]
        positions = [[(x, int(y)) for y in np.linspace(185, 405, len(v))]
                     for x, v in zip(columns, values)]
        for x, title in zip(columns, ["INPUT", "RECURRENT", "OUTPUT"]):
            self._center(title, x, 120, (39, 48, 58))
        for layer, links in enumerate([self.input_links, self.output_links]):
            for a, b in zip(*np.nonzero(links)):
                start = np.array(positions[layer][a], dtype=float)
                end = np.array(positions[layer + 1][b], dtype=float)
                direction = (end - start) / np.linalg.norm(end - start)
                start += direction * 20
                end -= direction * 23
                strength = float(values[layer][a])
                color = (int(198 - 95 * strength), int(209 - 70 * strength), int(218 - 45 * strength))
                pg.draw.aaline(self.screen, color, start, end)
                normal = np.array([-direction[1], direction[0]])
                pg.draw.polygon(self.screen, color, [end, end - direction * 6 + normal * 3, end - direction * 6 - normal * 3])
        labels = [["Left food", "Right food", "Walls", "Food dist", "Exit", "Nearest food"],
                  [f"{len(g)} neurons" for g in self.hidden_groups],
                  ["Left", "Forward", "Right"]]
        accents = [(40, 143, 196), (35, 155, 121), (212, 134, 42)]
        for layer, points in enumerate(positions):
            for idx, point in enumerate(points):
                value = float(np.clip(values[layer][idx], 0, 1))
                accent = accents[layer]
                fill = tuple(int(245 * (1 - value) + c * value) for c in accent)
                if layer == 1:
                    loop = pg.Rect(point[0] + 12, point[1] - 22, 22, 24)
                    pg.draw.arc(self.screen, accent, loop, -pi / 2, pi, 2)
                pg.draw.circle(self.screen, fill, point, 19)
                pg.draw.circle(self.screen, accent, point, 19, 2)
                self._center(str(int(value * 100)), point[0], point[1] - 7, (27, 43, 51))
                self._center(labels[layer][idx], point[0], point[1] + 22, (65, 77, 87))
        self._center("Grouped activity (%) / recurrent connections within", x0 + 260, 455, (95, 108, 118))

    def _center(self, text, x, y, color):
        surface = self.small_font.render(text, True, color)
        self.screen.blit(surface, (x - surface.get_width() // 2, y))

    def _draw_voltage_trace(self, values: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        pg = self.pg
        rect = pg.Rect(x, y, width, height)
        pg.draw.rect(self.screen, (17, 20, 24), rect)
        pg.draw.rect(self.screen, (55, 63, 70), rect, 1)
        if len(values) < 2:
            return
        scaled = np.clip(values, -0.2, 1.2)
        points = []
        for idx, value in enumerate(scaled):
            px = x + int(idx / (len(scaled) - 1) * (width - 1))
            py = y + height - int((float(value) + 0.2) / 1.4 * (height - 1))
            points.append((px, py))
        pg.draw.lines(self.screen, (219, 103, 116), False, points, 2)

    def _text(self, text: str, x: int, y: int, color: tuple[int, int, int], font) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
