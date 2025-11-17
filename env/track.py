import pygame
import numpy as np
from env.constants import CELL_SIZE
from env.race_track import board

board = board

class Track:
    def __init__(self):
        self.board = board
        self.height = len(board)
        self.width = len(board[0])
        self.collision_mask = None
        self.checkpoints = {}
        self.start_pos = None
        self.spawn_positions = []
        self._parse_track()
        
    def _parse_track(self):
        self.collision_mask = np.zeros((self.height, self.width), dtype=bool)
        
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell == '#':
                    self.collision_mask[y, x] = True
                elif cell in ('p', 'q', 'r', 's', 'a', 'b', 'c', 'd'):
                    pos = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                    self.spawn_positions.append(pos)
                    if cell == 'p' and not self.start_pos:
                        self.start_pos = pos
                elif cell.isdigit():
                    checkpoint_id = int(cell)
                    if checkpoint_id not in self.checkpoints:
                        self.checkpoints[checkpoint_id] = []
                    self.checkpoints[checkpoint_id].append((x, y))
    
    def check_collision(self, x, y):
        grid_x = int(x / CELL_SIZE)
        grid_y = int(y / CELL_SIZE)
        
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return True
        
        return self.collision_mask[grid_y, grid_x]
    
    def check_checkpoint(self, x, y):
        grid_x = int(x / CELL_SIZE)
        grid_y = int(y / CELL_SIZE)
        
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return None
        
        cell = self.board[grid_y][grid_x]
        if cell.isdigit():
            return int(cell)
        return None
    
    def render(self, screen, camera=None):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.board[y][x]
                color = None
                
                if cell == '#':
                    color = (100, 100, 100)
                elif cell == '.':
                    color = (50, 50, 50)
                elif cell == 'p':
                    color = (0, 255, 0)
                elif cell.isdigit():
                    color = (255, 255, 0)
                
                if color:
                    world_x = x * CELL_SIZE
                    world_y = y * CELL_SIZE
                    if camera:
                        sx = int((world_x - camera.offset_x) * camera.zoom)
                        sy = int((world_y - camera.offset_y) * camera.zoom)
                        size = max(1, int(CELL_SIZE * camera.zoom))
                        screen_rect = (sx, sy, size, size)
                    else:
                        screen_rect = (world_x, world_y, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, color, screen_rect)
    
    def get_start_position(self):
        if self.start_pos:
            return self.start_pos
        if self.spawn_positions:
            return self.spawn_positions[0]
        return None

    def get_start_positions(self):
        return list(self.spawn_positions)
