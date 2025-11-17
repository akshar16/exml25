import pygame
import math
from env.constants import CELL_SIZE, CAR_GAP


SENSOR_RAYS_VISIBLE = True
SENSOR_RAY_COLOR = (0, 220, 255)
SENSOR_RAY_THICKNESS = 1
SENSOR_RAY_MAX_DISTANCE = 300.0
SENSOR_RAY_STEPS = 25
SENSOR_RAY_OFFSETS = [-150.0, -120.0, -90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0, 120.0]


class Car:
    def __init__(self, x, y, image_path, uni, start_x, start_y, game, track):
        self.startX = start_x
        self.startY = start_y
        self._uni_index = uni
        self._game = game
        self._track = track

        self._original_image = pygame.image.load(image_path).convert_alpha()
        self._original_image = pygame.transform.scale(self._original_image, (12, 18))
        self._image = self._original_image
        self._rect = self._image.get_rect()
        
        self._hitbox = pygame.Rect(0, 0, int(12 * 0.7), int(18 * 0.7))
        
        self._x = x
        self._y = y
        self._rect.center = (x, y)
        self._hitbox.center = (x, y)
        
        self._velocity = 0
        self._angle = 0
        self._steering_angle = 0
        
        self._max_velocity = 5
        self._acceleration_rate = 0.2
        self._friction = 0.95
        self._turn_speed = 6
        
        self._max_steering = 100
        self._min_steering = -100
        self._rev = 0.0
        self._accelerating = False
        self._boost_energy = 1.0
        self._boost_active = False
        self._boost_request_time = 0
        self._boost_start_time = 0
        self._boost_lag_ms = 600
        self._boost_power = 1.5
        self._boost_consumption_per_ms = 0.0005
        self._boost_recharge_per_ms = 0.0002
        self._last_time = pygame.time.get_ticks()
        self._collision_end_time = 0
        self._recoil_factor = 0.5
        self._collision_points = []  # (x, y, timestamp)
        self._boost_points = []  # (x, y, timestamp)
        
    def _accelerate(self, direction):
        now = pygame.time.get_ticks()
        if now < getattr(self, '_collision_end_time', 0):
            return
        mult = 1.0
        if self._boost_request_time:
            elapsed = now - self._boost_request_time
            if elapsed < self._boost_lag_ms:
                warm = elapsed / max(1, self._boost_lag_ms)
                mult += warm * self._boost_power
        if self._boost_active:
            mult += self._boost_power
        def _bezier_ease(t, p1=0.25, p2=0.75):
            u = 1.0 - t
            return (3*u*u*t*p1) + (3*u*t*t*p2) + (t**3)
        if direction > 0:
            ease = _bezier_ease(min(1.0, max(0.0, self._rev)))
            self._velocity += direction * self._acceleration_rate * ease * mult
        else:
            self._velocity += direction * self._acceleration_rate * mult
        self._velocity = max(-self._max_velocity, min(self._velocity, self._max_velocity))
        if direction > 0:
            self._rev = min(1.0, self._rev + 0.02)
            self._accelerating = True
        else:
            self._accelerating = False
    
    def _steer(self, amount):
        now = pygame.time.get_ticks()
        if now < getattr(self, '_collision_end_time', 0):
            return
        self._steering_angle += amount
        self._steering_angle = max(self._min_steering, min(self._steering_angle, self._max_steering))

    def request_boost(self):
        now = pygame.time.get_ticks()
        if self._boost_energy <= 0:
            return
        if self._rev < 0.35 or self._rev > 0.75:
            return
        if self._boost_active:
            return
        # Record boost request location with timestamp
        self._boost_points.append((self._x, self._y, now))
        if len(self._boost_points) > 50:
            self._boost_points.pop(0)
        self._boost_request_time = now

    def brake(self):
        strength = 0.6
        now = pygame.time.get_ticks()
        if now < self._collision_end_time:
            return
        strength = max(0.0, min(1.0, strength))
        self._velocity *= (1.0 - strength)
    
    def update(self, all_cars=None):
        now = pygame.time.get_ticks()
        dt = now - self._last_time
        if dt < 0:
            dt = 0
        self._last_time = now

        if abs(self._velocity) > 0.1:
            turn_factor = self._steering_angle / 100.0
            self._angle += turn_factor * self._turn_speed * (self._velocity / self._max_velocity)
        
        self._steering_angle *= 0.9
        
        if not self._accelerating:
            self._rev = max(0.0, self._rev - dt * 0.0008)
        rad = math.radians(self._angle)
        dx = math.sin(rad) * self._velocity
        dy = -math.cos(rad) * self._velocity
        
        new_x = self._x + dx
        new_y = self._y + dy
        
        if not self._track.check_collision(new_x, new_y):
            self._x = new_x
            self._y = new_y
            self._rect.center = (self._x, self._y)
            self._hitbox.center = (self._x, self._y)
        else:
            now2 = pygame.time.get_ticks()
            if now2 >= getattr(self, '_collision_end_time', 0):
                impact_speed = abs(self._velocity)
                s = 0.0
                if getattr(self, '_max_velocity', 0) > 0:
                    s = min(1.0, impact_speed / float(self._max_velocity))
                def _bezier(t, p1=0.2, p2=0.8):
                    u = 1.0 - t
                    return (3*u*u*t*p1) + (3*u*t*t*p2) + (t**3)
                scale = _bezier(s)
                impact_dir = -1 if self._velocity < 0 else 1
                recoil = impact_speed * scale * getattr(self, '_recoil_factor', 0.5) * impact_dir
                self._velocity = -recoil
                self._steering_angle = 0
                self._collision_end_time = now2 + 1000
                self._collision_points.append((self._x, self._y, now2))
                if len(self._collision_points) > 50:
                    self._collision_points.pop(0)

        if all_cars:
            for other in all_cars:
                if other is self or other._is_in_collision():
                    continue
                if self._hitbox.colliderect(other._hitbox):
                    impact_speed = abs(self._velocity)
                    self._velocity = -impact_speed * getattr(self, '_recoil_factor', 0.5)
                    self._steering_angle = 0
                    self._collision_end_time = now + 1000
                    other._velocity = 0
                    other._steering_angle = 0
                    other._collision_end_time = now + 1000
                    collision_point = ((self._x + other._x) / 2, (self._y + other._y) / 2, now)
                    self._collision_points.append(collision_point)
                    other._collision_points.append(collision_point)
                    if len(self._collision_points) > 50:
                        self._collision_points.pop(0)
                    if len(other._collision_points) > 50:
                        other._collision_points.pop(0)

        self._velocity *= self._friction

        if self._boost_active:
            consume = self._boost_consumption_per_ms * dt
            self._boost_energy = max(0.0, self._boost_energy - consume)
            if self._boost_energy <= 0:
                self._boost_active = False
                self._boost_request_time = 0
        else:
            if self._boost_request_time:
                if now - self._boost_request_time >= self._boost_lag_ms:
                    if self._boost_energy > 0:
                        self._boost_active = True
                        self._boost_start_time = now
                        self._boost_request_time = 0
                        self._boost_points.append((self._x, self._y, now))
                        if len(self._boost_points) > 50:
                            self._boost_points.pop(0)
            else:
                recharge = self._boost_recharge_per_ms * dt
                self._boost_energy = min(1.0, self._boost_energy + recharge)
        
        self._image = pygame.transform.rotate(self._original_image, -self._angle)
        self._rect = self._image.get_rect(center=(self._x, self._y))
        self._hitbox.center = (self._x, self._y)


    def render(self, screen, camera=None):
        if camera:
            center = (self._x, self._y)
            screen_pos = camera.apply(center)
            z = int(max(1, round(camera.zoom)))
            ow = self._original_image.get_width()
            oh = self._original_image.get_height()
            scaled = pygame.transform.scale(self._original_image, (ow * z, oh * z))
            rotated = pygame.transform.rotate(scaled, -self._angle)
            rect = rotated.get_rect(center=screen_pos)
            screen.blit(rotated, rect)
            # draw car number above car
            font = pygame.font.Font(None, 20)
            label = font.render(str(self._uni_index + 1), True, (255, 255, 0))
            label_pos = (int(screen_pos[0] - label.get_width() / 2), int(screen_pos[1] - 25))
            screen.blit(label, label_pos)
        else:
            screen.blit(self._image, self._rect)
            font = pygame.font.Font(None, 20)
            label = font.render(str(self._uni_index + 1), True, (255, 255, 0))
            label_pos = (int(self._x - label.get_width() / 2), int(self._y - 25))
            screen.blit(label, label_pos)

        if SENSOR_RAYS_VISIBLE:
            self._render_sensor_rays(screen, camera)

        current_time = pygame.time.get_ticks()
        for cx, cy, ts in self._collision_points:
            if current_time - ts <= 1000:
                if camera:
                    screen_pos = camera.apply((cx, cy))
                    pygame.draw.circle(screen, (255, 0, 0), (int(screen_pos[0]), int(screen_pos[1])), 5)
                else:
                    pygame.draw.circle(screen, (255, 0, 0), (int(cx), int(cy)), 5)

        for bx, by, ts in self._boost_points:
            if current_time - ts <= 1000:
                if camera:
                    screen_pos = camera.apply((bx, by))
                    pygame.draw.circle(screen, (0, 255, 0), (int(screen_pos[0]), int(screen_pos[1])), 4)
                else:
                    pygame.draw.circle(screen, (0, 255, 0), (int(bx), int(by)), 4)
        
    def get_position(self):
        return (self._x, self._y)
    
    def reset(self):
        self._x = self.startX
        self._y = self.startY
        self._velocity = 0
        self._angle = 0
        self._steering_angle = 0
        self._collision_end_time = 0
        self._rect.center = (self.startX, self.startY)
        self._hitbox.center = (self.startX, self.startY)
        self._image = self._original_image
        self._collision_points = []
        self._boost_points = []

    def _is_in_collision(self):
        return (pygame.time.get_ticks() < self._collision_end_time) or self._track.check_collision(self._x, self._y)
    
    def get_observation(self):
        x, y = self.get_position()
        gx = int(x // CELL_SIZE)
        gy = int(y // CELL_SIZE)
    
        obs = {}
        obs['x'] = x
        obs['y'] = y
        obs['angle_degrees'] = float(getattr(self, '_angle', 0.0))
        obs['steering_angle'] = float(getattr(self, '_steering_angle', 0.0))
        obs['speed'] = float(getattr(self, '_velocity', 0.0))
        obs['track_coords'] = self._getTrackRecords()
        obs['lap_progress'] = self._get_lap_progress()
        obs['lap_number'] = self._get_lap_number()
        lap_times, current = self._get_lap_timings()
        obs['lap_times'] = lap_times
        obs['current_lap_time'] = current
        obs['collided'] = bool(self._is_in_collision())
        obs['all_coords'] = self._game.all_coords(self._uni_index)
        return obs
    
    def steer_right(self):
        self._steer(10)
    
    def steer_left(self):
        self._steer(-10)
    
    def accelerate_fwd(self):
        self._accelerate(1.0)
    
    def accelerate_bck(self):
        self._accelerate(-1.0)

    def _get_lap_timings(self):
        lap_times = list([] if self._uni_index not in self._game._lap_times else self._game._lap_times[self._uni_index])
        start = None if self._uni_index not in self._game._lap_start_time else self._game._lap_start_time[self._uni_index]
        if start is None:
            current = 0.0
        else:
            current = (pygame.time.get_ticks() - start) / 1000.0
        return lap_times, current

    def _get_lap_number(self):
        return int(0 if self._uni_index not in self._game._laps_completed else self._game._laps_completed[self._uni_index]) + 1

    def _getTrackRecords(self):
        x, y = self.get_position()
        gx = int(x // CELL_SIZE)
        gy = int(y // CELL_SIZE)
        return (gx, gy)

    def _checkpoint_centroid(self, cid):
        cells = self._track.checkpoints.get(cid)
        if not cells:
            return None
        sx = 0.0
        sy = 0.0
        for (cx, cy) in cells:
            sx += (cx * CELL_SIZE + CELL_SIZE / 2.0)
            sy += (cy * CELL_SIZE + CELL_SIZE / 2.0)
        n = len(cells)
        return (sx / n, sy / n)

    def _get_lap_progress(self):
        total = len(self._track.checkpoints) if self._track.checkpoints else 0
        if total == 0:
            return 0.0
        collected = set() if self._uni_index not in self._game._checkpoints_collected else self._game._checkpoints_collected[self._uni_index]
        now = None
        completed = len(collected)
        if completed >= total:
            return 1.0
        prev_id = 0
        if collected:
            prev_id = max(collected)

        if prev_id == 0:
            start_x, start_y = self._track.get_start_position()
            prev_pos = (start_x + self._uni_index * CAR_GAP, start_y)
        else:
            prev_pos = self._checkpoint_centroid(prev_id)

        next_id = prev_id + 1 if prev_id < total else 1
        next_pos = self._checkpoint_centroid(next_id)
        if not prev_pos or not next_pos:
            return float(completed) / float(total)
        car_x, car_y = self.get_position()
        dist_total = math.hypot(next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1])
        if dist_total <= 0.0:
            frac = 0.0
        else:
            dist_to_car = math.hypot(car_x - prev_pos[0], car_y - prev_pos[1])
            frac = max(0.0, min(1.0, dist_to_car / dist_total))
        progress = (float(prev_id) + frac) / float(total)
        return max(0.0, min(1.0, progress))

    def _render_sensor_rays(self, screen, camera):
        center = (self._x, self._y)
        base_angle = self._angle
        if camera:
            start_point = camera.apply(center)
        else:
            start_point = center

        step_distance = SENSOR_RAY_MAX_DISTANCE / float(SENSOR_RAY_STEPS)

        for offset in SENSOR_RAY_OFFSETS:
            ray_angle = base_angle + offset
            rad = math.radians(ray_angle)
            hit_x = None
            hit_y = None

            for step in range(1, SENSOR_RAY_STEPS + 1):
                dist = step * step_distance
                check_x = center[0] + dist * math.sin(rad)
                check_y = center[1] - dist * math.cos(rad)

                blocked = False
                if self._track.check_collision(check_x, check_y):
                    blocked = True
                elif hasattr(self._track, 'is_drivable'):
                    try:
                        if not self._track.is_drivable(check_x, check_y):
                            blocked = True
                    except TypeError:
                        pass

                if not blocked:
                    for other in getattr(self._game, '_cars', []):
                        if other is self:
                            continue
                        if other._hitbox.collidepoint(int(check_x), int(check_y)):
                            blocked = True
                            break

                if blocked:
                    hit_x = check_x
                    hit_y = check_y
                    break

            if hit_x is None:
                hit_x = center[0] + SENSOR_RAY_MAX_DISTANCE * math.sin(rad)
                hit_y = center[1] - SENSOR_RAY_MAX_DISTANCE * math.cos(rad)

            if camera:
                end_point = camera.apply((hit_x, hit_y))
                start_draw = (int(round(start_point[0])), int(round(start_point[1])))
                end_draw = (int(round(end_point[0])), int(round(end_point[1])))
            else:
                start_draw = (int(round(start_point[0])), int(round(start_point[1])))
                end_draw = (int(round(hit_x)), int(round(hit_y)))

            pygame.draw.line(screen, SENSOR_RAY_COLOR, start_draw, end_draw, SENSOR_RAY_THICKNESS)
            pygame.draw.circle(screen, SENSOR_RAY_COLOR, end_draw, max(2, SENSOR_RAY_THICKNESS))
