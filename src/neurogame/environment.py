from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from neurogame.config import EnvConfig


@dataclass
class AgentState:
    x: float
    y: float
    heading: float
    speed: float = 0.0


@dataclass
class EnvStep:
    observation: np.ndarray
    reward: float
    ate_food: bool
    reached_exit: bool
    distance_to_nearest_food: float


@dataclass(frozen=True)
class MazeWall:
    x: float
    y: float
    width: float
    height: float


class FoodWorld:
    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.maze_seed = int(seed if seed is not None else self.rng.integers(0, 2**31))
        self.maze_level = 1
        self.maze_columns = 7
        self.maze_rows = 5
        self.entrance_position = np.array([72.0, self.config.height / 2], dtype=np.float32)
        self.exit_bounds = (213.0, 307.0)
        self.walls: list[MazeWall] = []
        self.agent = AgentState(float(self.entrance_position[0]), float(self.entrance_position[1]), 0.0)
        self.food = np.zeros((self.config.food_count, 2), dtype=np.float32)
        self.food_active = np.ones(self.config.food_count, dtype=bool)
        self.total_reward = 0.0
        self.food_eaten = 0
        self.exits_completed = 0
        self.path_checkpoint = 0
        self.steps = 0
        self.visited_cells: dict[tuple[int, int], int] = {}
        self.reset()

    def reset(self) -> np.ndarray:
        self.maze_level = 1
        self.walls = self._make_maze()
        self.agent = AgentState(
            x=float(self.entrance_position[0]),
            y=float(self.entrance_position[1]),
            heading=float(self.rng.uniform(-pi, pi)),
            speed=0.0,
        )
        self.food_eaten = 0
        self.exits_completed = 0
        self.path_checkpoint = 0
        self.total_reward = 0.0
        self.steps = 0
        self.visited_cells = {}
        self.food_active.fill(True)
        for idx in range(self.config.food_count):
            self.food[idx] = self._random_food_position()
        return self.observe()

    def _random_food_position(self) -> np.ndarray:
        margin = 45
        for _ in range(1_000):
            position = np.array([
                self.rng.uniform(margin, self.config.width - margin),
                self.rng.uniform(margin, self.config.height - margin),
            ], dtype=np.float32)
            if not any(point_near_wall(position, wall, self.config.food_radius + 5) for wall in self.walls):
                return position
        raise RuntimeError("Could not place food outside maze walls.")

    def _make_maze(self) -> list[MazeWall]:
        """Build a deterministic perfect maze with branches and dead ends."""
        if not self.config.maze_enabled:
            self.maze_columns, self.maze_rows = 1, 1
            self.maze_geometry = (25.0, 25.0, 670.0, 470.0)
            self.entrance_position = np.array([55.0, 260.0], dtype=np.float32)
            self.exit_bounds = (210.0, 310.0)
            self.maze_passages = set()
            self.exit_path = np.array([[55.0, 260.0], [690.0, 260.0]], dtype=np.float32)
            return []
        columns = min(self.config.base_maze_columns + self.maze_level - 1, 17)
        rows = min(self.config.base_maze_rows + 2 * ((self.maze_level - 1) // 2), 13)
        self.maze_columns, self.maze_rows = columns, rows
        left, top, right, bottom = 25.0, 25.0, 695.0, 495.0
        cell_width = (right - left) / columns
        cell_height = (bottom - top) / rows
        self.maze_geometry = (left, top, cell_width, cell_height)
        thickness = max(5.0, 10.0 - 0.4 * (self.maze_level - 1))
        entrance_row = rows // 2
        entrance_y = top + (entrance_row + 0.5) * cell_height
        self.entrance_position = np.array([left + cell_width * 0.5, entrance_y], dtype=np.float32)
        self.exit_bounds = (
            top + entrance_row * cell_height,
            top + (entrance_row + 1) * cell_height,
        )

        maze_rng = np.random.default_rng(self.maze_seed + self.maze_level * 7919)
        visited = {(0, entrance_row)}
        stack = [(0, entrance_row)]
        passages: set[frozenset[tuple[int, int]]] = set()
        while stack:
            column, row = stack[-1]
            neighbors = [
                (next_column, next_row)
                for next_column, next_row in (
                    (column - 1, row), (column + 1, row),
                    (column, row - 1), (column, row + 1),
                )
                if 0 <= next_column < columns and 0 <= next_row < rows
                and (next_column, next_row) not in visited
            ]
            if not neighbors:
                stack.pop()
                continue
            neighbor = neighbors[int(maze_rng.integers(len(neighbors)))]
            passages.add(frozenset(((column, row), neighbor)))
            visited.add(neighbor)
            stack.append(neighbor)
        self.maze_passages = passages

        start, goal = (0, entrance_row), (columns - 1, entrance_row)
        frontier = [start]
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            cell = frontier.pop(0)
            if cell == goal:
                break
            for passage in passages:
                if cell not in passage:
                    continue
                neighbor = next(item for item in passage if item != cell)
                if neighbor not in parents:
                    parents[neighbor] = cell
                    frontier.append(neighbor)
        path = [goal]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        self.exit_path = np.array([
            [left + (column + 0.5) * cell_width, top + (row + 0.5) * cell_height]
            for column, row in path
        ], dtype=np.float32)

        walls: list[MazeWall] = []
        for column in range(columns):
            x = left + column * cell_width
            walls.append(MazeWall(x, top, cell_width + thickness, thickness))
            walls.append(MazeWall(x, bottom - thickness, cell_width + thickness, thickness))
        for row in range(rows):
            y = top + row * cell_height
            if row != entrance_row:
                walls.append(MazeWall(left, y, thickness, cell_height + thickness))
                walls.append(MazeWall(right - thickness, y, thickness, cell_height + thickness))
        for column in range(1, columns):
            x = left + column * cell_width - thickness / 2
            for row in range(rows):
                if frozenset(((column - 1, row), (column, row))) not in passages:
                    walls.append(MazeWall(x, top + row * cell_height, thickness, cell_height + thickness))
        for row in range(1, rows):
            y = top + row * cell_height - thickness / 2
            for column in range(columns):
                if frozenset(((column, row - 1), (column, row))) not in passages:
                    walls.append(MazeWall(left + column * cell_width, y, cell_width + thickness, thickness))
        return walls

    def step(self, turn: float, thrust: float) -> EnvStep:
        cfg = self.config
        old_agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        active_food = self.food[self.food_active]
        old_nearest_distance = (
            float(np.min(np.linalg.norm(active_food - old_agent_pos, axis=1)))
            if len(active_food) else 0.0
        )
        turn = float(np.clip(turn, -1.0, 1.0))
        thrust = float(np.clip(thrust, 0.0, 1.0))

        self.agent.heading = wrap_angle(self.agent.heading + turn * cfg.turn_rate)
        target_speed = thrust * cfg.max_speed
        self.agent.speed = 0.82 * self.agent.speed + 0.18 * target_speed
        self.agent.x += cos(self.agent.heading) * self.agent.speed
        self.agent.y += sin(self.agent.heading) * self.agent.speed

        reward = cfg.step_penalty
        hit_wall = False
        if self.agent.x < cfg.agent_radius:
            self.agent.x = cfg.agent_radius
            hit_wall = True
        elif self.agent.x > cfg.width - cfg.agent_radius:
            self.agent.x = cfg.width - cfg.agent_radius
            hit_wall = True
        if self.agent.y < cfg.agent_radius:
            self.agent.y = cfg.agent_radius
            hit_wall = True
        elif self.agent.y > cfg.height - cfg.agent_radius:
            self.agent.y = cfg.height - cfg.agent_radius
            hit_wall = True

        if hit_wall:
            reward += cfg.wall_penalty
            self.agent.heading = wrap_angle(self.agent.heading + self.rng.uniform(-0.5, 0.5))

        if any(circle_hits_wall(self.agent.x, self.agent.y, cfg.agent_radius, wall) for wall in self.walls):
            self.agent.x = float(old_agent_pos[0])
            self.agent.y = float(old_agent_pos[1])
            self.agent.speed *= 0.25
            reward += cfg.wall_penalty
            hit_wall = True
            self.agent.heading = wrap_angle(self.agent.heading + self.rng.uniform(-0.8, 0.8))

        ate_food = False
        reached_exit = False
        agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        active_indices = np.flatnonzero(self.food_active)
        distances = np.linalg.norm(self.food[active_indices] - agent_pos, axis=1)
        if len(distances) and float(np.min(distances)) <= cfg.agent_radius + cfg.food_radius:
            nearest_idx = int(active_indices[int(np.argmin(distances))])
            reward += cfg.food_reward
            ate_food = True
            self.food_eaten += 1
            self.food_active[nearest_idx] = False
            if not np.any(self.food_active):
                reward += cfg.all_food_bonus

        at_exit = self.agent.x >= 665.0 and self.exit_bounds[0] <= self.agent.y <= self.exit_bounds[1]
        if at_exit and np.any(self.food_active):
            reward += cfg.locked_exit_penalty
            self.agent.x = 650.0
            self.agent.heading = wrap_angle(self.agent.heading + pi)
            self.agent.speed = 0.0
        elif at_exit:
            reward += cfg.exit_reward
            reached_exit = True
            self.exits_completed += 1
            self.maze_level += 1
            self.walls = self._make_maze()
            self.path_checkpoint = 0
            self.visited_cells = {}
            for idx in range(self.config.food_count):
                self.food[idx] = self._random_food_position()
            self.food_active.fill(True)
            self.agent.x = float(self.entrance_position[0])
            self.agent.y = float(self.entrance_position[1])
            self.agent.heading = 0.0
            self.agent.speed = 0.0

        observation = self.observe()
        current_agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        if not reached_exit:
            cell = self._cell_for(current_agent_pos)
            visits = self.visited_cells.get(cell, 0)
            if visits == 0 or cell != self._cell_for(old_agent_pos):
                reward += cfg.novelty_reward if visits == 0 else -cfg.revisit_penalty * min(visits, 10)
                self.visited_cells[cell] = visits + 1
        remaining_food = self.food[self.food_active]
        new_distances = (
            np.linalg.norm(remaining_food - current_agent_pos, axis=1)
            if len(remaining_food) else np.array([0.0], dtype=np.float32)
        )
        if len(remaining_food) and not ate_food and not reached_exit:
            progress = old_nearest_distance - float(np.min(new_distances))
            shaped_reward = cfg.progress_reward_scale * progress / max(cfg.max_speed, 1e-6)
            reward += float(np.clip(shaped_reward, -cfg.progress_reward_scale, cfg.progress_reward_scale))

        self.steps += 1
        self.total_reward += reward
        return EnvStep(
            observation=observation,
            reward=float(reward),
            ate_food=ate_food,
            reached_exit=reached_exit,
            distance_to_nearest_food=float(np.min(new_distances)),
        )

    def _path_progress(self, position: np.ndarray) -> float:
        distances = np.linalg.norm(self.exit_path - position, axis=1)
        index = int(np.argmin(distances))
        return float(index - distances[index] / 100.0)

    def _cell_for(self, position: np.ndarray) -> tuple[int, int]:
        left, top, cell_width, cell_height = self.maze_geometry
        return (
            int(np.clip((float(position[0]) - left) / cell_width, 0, self.maze_columns - 1)),
            int(np.clip((float(position[1]) - top) / cell_height, 0, self.maze_rows - 1)),
        )

    def observe(self) -> np.ndarray:
        cfg = self.config
        agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        active_food = self.food[self.food_active]
        rel = active_food - agent_pos
        distances = np.linalg.norm(rel, axis=1)
        angles = np.array([wrap_angle(atan2(y, x) - self.agent.heading) for x, y in rel], dtype=np.float32)

        lidar_angles = np.linspace(-pi, pi, 8, endpoint=False, dtype=np.float32)
        wall_rays = np.array([
            1.0 - np.clip(
                distance_to_wall(self.agent, cfg, float(ray), self.walls, self.exit_bounds)
                / cfg.sensor_range,
                0.0,
                1.0,
            )
            for ray in lidar_angles
        ], dtype=np.float32)

        front_wall = distance_to_wall(self.agent, cfg, 0.0, self.walls, self.exit_bounds) / cfg.sensor_range
        left_wall = distance_to_wall(self.agent, cfg, -pi / 3, self.walls, self.exit_bounds) / cfg.sensor_range
        right_wall = distance_to_wall(self.agent, cfg, pi / 3, self.walls, self.exit_bounds) / cfg.sensor_range
        nearest_food = (float(np.min(distances)) / cfg.sensor_range) if len(distances) else 1.0

        wall_sensors = 1.0 - np.clip(np.array([left_wall, front_wall, right_wall]), 0.0, 1.0)
        food_distance = np.array([1.0 - np.clip(nearest_food, 0.0, 1.0)], dtype=np.float32)
        exit_goal = np.array([690.0, float(sum(self.exit_bounds) / 2)], dtype=np.float32)
        exit_delta = exit_goal - agent_pos
        exit_distance = float(np.linalg.norm(exit_delta))
        exit_angle = float(wrap_angle(atan2(exit_delta[1], exit_delta[0]) - self.agent.heading))
        exit_sensors = np.array([
            (sin(exit_angle) + 1.0) * 0.5,
            (cos(exit_angle) + 1.0) * 0.5,
            1.0 - np.clip(exit_distance / np.hypot(cfg.width, cfg.height), 0.0, 1.0),
        ], dtype=np.float32)
        if len(distances):
            nearest_idx = int(np.argmin(distances))
            nearest_angle = float(angles[nearest_idx])
            nearest_food_sensors = np.array([
                (sin(nearest_angle) + 1.0) * 0.5,
                (cos(nearest_angle) + 1.0) * 0.5,
                float(len(distances) / self.config.food_count),
            ], dtype=np.float32)
        else:
            nearest_food_sensors = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        return np.concatenate([
            wall_rays, wall_sensors.astype(np.float32), food_distance, exit_sensors,
            nearest_food_sensors,
        ]).astype(np.float32)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + pi) % (2 * pi) - pi


def distance_to_wall(agent: AgentState, cfg: EnvConfig, relative_angle: float,
                     walls: list[MazeWall] | None = None,
                     opening_bounds: tuple[float, float] | None = None) -> float:
    angle = agent.heading + relative_angle
    dx = cos(angle)
    dy = sin(angle)
    candidates: list[float] = []

    if abs(dx) > 1e-6:
        left_distance = (cfg.agent_radius - agent.x) / dx
        right_distance = (cfg.width - cfg.agent_radius - agent.x) / dx
        left_hit_y = agent.y + dy * left_distance
        right_hit_y = agent.y + dy * right_distance
        if not (dx < 0 and opening_bounds and opening_bounds[0] <= left_hit_y <= opening_bounds[1]):
            candidates.append(left_distance)
        if not (dx > 0 and opening_bounds and opening_bounds[0] <= right_hit_y <= opening_bounds[1]):
            candidates.append(right_distance)
    if abs(dy) > 1e-6:
        candidates.append((cfg.agent_radius - agent.y) / dy)
        candidates.append((cfg.height - cfg.agent_radius - agent.y) / dy)

    positive = [value for value in candidates if value > 0]
    nearest = min(positive) if positive else cfg.sensor_range
    for wall in walls or []:
        distance = ray_wall_distance(agent.x, agent.y, dx, dy, wall)
        if distance is not None:
            nearest = min(nearest, distance)
    return nearest


def ray_wall_distance(x: float, y: float, dx: float, dy: float,
                      wall: MazeWall) -> float | None:
    """Return the first positive intersection of a ray and rectangle."""
    low = 0.0
    high = float("inf")
    for origin, direction, minimum, maximum in (
        (x, dx, wall.x, wall.x + wall.width),
        (y, dy, wall.y, wall.y + wall.height),
    ):
        if abs(direction) < 1e-8:
            if origin < minimum or origin > maximum:
                return None
            continue
        a = (minimum - origin) / direction
        b = (maximum - origin) / direction
        low = max(low, min(a, b))
        high = min(high, max(a, b))
        if high < low:
            return None
    return low if high >= max(low, 0.0) else None


def circle_hits_wall(x: float, y: float, radius: float, wall: MazeWall) -> bool:
    nearest_x = float(np.clip(x, wall.x, wall.x + wall.width))
    nearest_y = float(np.clip(y, wall.y, wall.y + wall.height))
    return (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius**2


def point_near_wall(point: np.ndarray, wall: MazeWall, padding: float) -> bool:
    return (
        wall.x - padding <= float(point[0]) <= wall.x + wall.width + padding
        and wall.y - padding <= float(point[1]) <= wall.y + wall.height + padding
    )
