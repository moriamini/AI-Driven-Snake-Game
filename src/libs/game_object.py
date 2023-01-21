from collections import namedtuple

Point = namedtuple('Point', 'x, y')

class GameObject:
    def __init__(self, x_post, y_post, tag='object'):
        self.tag = tag
        self.position = Point(x_post, y_post)