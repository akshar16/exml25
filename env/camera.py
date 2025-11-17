import pygame

class Camera:
    def __init__(self, screen_w, screen_h, world_w, world_h, zoom=1.0):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.world_w = int(world_w)
        self.world_h = int(world_h)
        self.zoom = float(zoom)
        self.offset_x = 0
        self.offset_y = 0

    def update(self, target):
        tx, ty = target.get_position()
        view_w = int(self.screen_w / self.zoom)
        view_h = int(self.screen_h / self.zoom)

        self.offset_x = int(tx - view_w // 2)
        self.offset_y = int(ty - view_h // 2)

        if self.offset_x < 0:
            self.offset_x = 0
        if self.offset_y < 0:
            self.offset_y = 0
        if self.offset_x + view_w > self.world_w:
            self.offset_x = max(0, self.world_w - view_w)
        if self.offset_y + view_h > self.world_h:
            self.offset_y = max(0, self.world_h - view_h)

    def apply(self, pos):
        x, y = pos
        sx = int((x - self.offset_x) * self.zoom)
        sy = int((y - self.offset_y) * self.zoom)
        return (sx, sy)

    def apply_rect(self, rect):
        x, y = rect.topleft
        sx, sy = self.apply((x, y))
        return pygame.Rect(sx, sy, int(rect.width * self.zoom), int(rect.height * self.zoom))
