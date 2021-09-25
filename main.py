import os
import random
import sys
from PIL import Image, ImageDraw

from classes.point import Point


def manager():
    img = Image.new('RGB', (1920, 1080), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    points = [Point(x, random.randint(500, 1000), random.randint(500, 1000), img) for x in range(10)]
    while True:
        for point in points:
            point.move()
            point.draw()
        img.save('map.png')
        os.system(f'feh --bg-scale {os.getcwd()}/map.png')
        d.rectangle((0, 0, 1920, 1080), fill=(0, 0, 0, 0))
        img.save('map.png')


if __name__ == '__main__':
    manager()
