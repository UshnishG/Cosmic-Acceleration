"""
Barnes-Hut O(N log N) Tree-Code Approximation.
Implements hierarchical space partitioning (Octree for 3D, Quadtree for 2D)
with multipole acceptance criterion (MAC: s/d < theta).
"""
import numpy as np
from typing import List, Optional, Tuple
from src.constants import SimulationConfig
from src.integrators import velocity_verlet_step, compute_system_metrics

class BHNode:
    __slots__ = (
        'center', 'half_size', 'mass', 'com', 'children',
        'is_leaf', 'particle_idx'
    )
    def __init__(self, center: np.ndarray, half_size: float):
        self.center = center              # (3,) node spatial center
        self.half_size = half_size        # half-width of bounding cube
        self.mass = 0.0                   # total mass in cell
        self.com = np.zeros(3, dtype=np.float64) # center of mass
        self.children: List[Optional['BHNode']] = [None] * 8
        self.is_leaf = True
        self.particle_idx: Optional[int] = None # set if node contains exactly 1 particle

class BarnesHutTree:
    def __init__(self, pos: np.ndarray, mass: np.ndarray, theta: float = 0.5):
        self.pos = pos
        self.mass = mass
        self.theta = theta
        self.N = len(mass)
        
        # Determine bounding box enclosing all particles
        min_coords = np.min(pos, axis=0)
        max_coords = np.max(pos, axis=0)
        center = 0.5 * (min_coords + max_coords)
        half_size = max(float(np.max(max_coords - min_coords) * 0.51), 1e-3)
        
        self.root = BHNode(center, half_size)
        self._build_tree()

    def _get_octant(self, center: np.ndarray, p: np.ndarray) -> int:
        octant = 0
        if p[0] >= center[0]: octant |= 1
        if p[1] >= center[1]: octant |= 2
        if p[2] >= center[2]: octant |= 4
        return octant

    def _get_child_center(self, center: np.ndarray, half_size: float, octant: int) -> np.ndarray:
        child_hs = 0.5 * half_size
        dx = child_hs if (octant & 1) else -child_hs
        dy = child_hs if (octant & 2) else -child_hs
        dz = child_hs if (octant & 4) else -child_hs
        return center + np.array([dx, dy, dz], dtype=np.float64)

    def _insert(self, node: BHNode, p_idx: int):
        p_pos = self.pos[p_idx]
        p_mass = self.mass[p_idx]
        
        # If node is completely empty
        if node.mass == 0.0 and node.is_leaf and node.particle_idx is None:
            node.particle_idx = p_idx
            node.mass = p_mass
            node.com = p_pos.copy()
            return
            
        # If node contains 1 particle and is still a leaf, subdivide
        if node.is_leaf:
            existing_idx = node.particle_idx
            node.is_leaf = False
            node.particle_idx = None
            
            # Re-insert existing particle into appropriate child
            if existing_idx is not None:
                oct1 = self._get_octant(node.center, self.pos[existing_idx])
                if node.children[oct1] is None:
                    node.children[oct1] = BHNode(
                        self._get_child_center(node.center, node.half_size, oct1),
                        0.5 * node.half_size
                    )
                self._insert(node.children[oct1], existing_idx)
                
        # Insert the new particle into child
        oct2 = self._get_octant(node.center, p_pos)
        if node.children[oct2] is None:
            node.children[oct2] = BHNode(
                self._get_child_center(node.center, node.half_size, oct2),
                0.5 * node.half_size
            )
        self._insert(node.children[oct2], p_idx)
        
        # Update center of mass and total mass
        total_m = node.mass + p_mass
        if total_m > 0:
            node.com = (node.mass * node.com + p_mass * p_pos) / total_m
            node.mass = total_m

    def _build_tree(self):
        for i in range(self.N):
            self._insert(self.root, i)

    def compute_force_on_particle(
        self,
        target_idx: int,
        target_pos: np.ndarray,
        G: float,
        softening_sq: float
    ) -> np.ndarray:
        """
        Traverses tree to compute acceleration on particle target_idx.
        Stack-based non-recursive traversal for speed.
        """
        acc = np.zeros(3, dtype=np.float64)
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            if node is None or node.mass <= 0.0:
                continue
                
            diff = node.com - target_pos
            dist_sq = float(np.sum(diff ** 2)) + softening_sq
            dist = np.sqrt(dist_sq)
            
            # If leaf node
            if node.is_leaf:
                if node.particle_idx is not None and node.particle_idx != target_idx:
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    acc += (G * node.mass * inv_dist3) * diff
                continue
                
            # Multipole Acceptance Criterion (MAC): s / d < theta
            # s = node width = 2 * half_size
            node_size = 2.0 * node.half_size
            if (node_size / dist) < self.theta:
                inv_dist3 = 1.0 / (dist_sq * dist)
                acc += (G * node.mass * inv_dist3) * diff
            else:
                for child in node.children:
                    if child is not None and child.mass > 0.0:
                        stack.append(child)
                        
        return acc

    def compute_accelerations(self, G: float, softening: float) -> np.ndarray:
        softening_sq = softening ** 2
        accels = np.zeros_like(self.pos)
        for i in range(self.N):
            accels[i] = self.compute_force_on_particle(i, self.pos[i], G, softening_sq)
        return accels

class BarnesHutSolver:
    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        config: Optional[SimulationConfig] = None
    ):
        self.config = config or SimulationConfig()
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.mass = mass.copy()
        self.N = len(mass)
        self.G = self.config.G
        self.softening = self.config.softening
        self.theta = self.config.theta
        
        self.acc = self.compute_acceleration(self.pos)
        self.time = 0.0
        self.step_count = 0

    def compute_acceleration(self, pos: np.ndarray) -> np.ndarray:
        tree = BarnesHutTree(pos, self.mass, theta=self.theta)
        return tree.compute_accelerations(self.G, self.softening)

    def step(self, dt: float) -> float:
        self.pos, self.vel, self.acc = velocity_verlet_step(
            self.pos, self.vel, self.acc, self.compute_acceleration, dt
        )
        self.time += dt
        self.step_count += 1
        return dt

    def get_state(self):
        """Retrieve current pos, vel, acc, mass as NumPy arrays."""
        return (
            self.pos.copy(),
            self.vel.copy(),
            self.acc.copy(),
            self.mass.copy()
        )
