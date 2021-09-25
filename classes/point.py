import random
from time import sleep

import redis
from PIL import ImageDraw
from loguru import logger

from classes.pathfinder import Gridworld, Cell, astar
from utils.profiler import time_profile


class Point:
    def __init__(self, uid, x, y, img):
        self.uid = uid
        self.x = x
        self.y = y
        self.img = img
        self.target = None
        self._redis = redis.Redis(host='localhost', port=6379, db=0)

    def move(self):
        # target = (random.randint(1, 1920), random.randint(1, 1080))
        # if not self.target:
        #     self.target = target
        # self._get_position()
        # path = self._get_path(*self.target)
        # self.x, self.y = path[1][0], path[1][1]
        # logger.info(f'{self.uid}: move -> {path[0]}')
        # self._set_position()
        target = random.choice((
            self._move_right,
            self._move_down,
            self._move_left,
            self._move_up,
        ))
        self._get_position()
        target()
        self._set_position()

    @time_profile
    def _get_path(self, x, y):
        world = Gridworld((1920, 1080))
        start = Cell()
        start.position = (self.x, self.y)
        goal = Cell()
        goal.position = (x, y)
        path = astar(world, start, goal)
        return path

    def _get_position(self):
        self.x = int(self._redis.get(f"point_{self.uid}_x") or 500)
        self.y = int(self._redis.get(f"point_{self.uid}_y") or 500)

    def _set_position(self):
        self._redis.set(f"point_{self.uid}_x", self.x)
        self._redis.set(f"point_{self.uid}_y", self.y)

    def draw(self):
        canvas = ImageDraw.Draw(self.img)
        for x in range(self.x, self.x + 10):
            for y in range(self.y, self.y + 10):
                canvas.point((x, y), (0, 255, 0))
                canvas.text((self.x + 10, self.y - 10), f'Point: {self.uid}')

    def _move_right(self):
        if self.x + 10 < 1920:
            self.x += 10

    def _move_left(self):
        if self.x > 0:
            self.x -= 10

    def _move_up(self):
        if self.y > 0:
            self.y -= 10

    def _move_down(self):
        if self.y + 10 < 1080:
            self.y += 10
