import bpy
import numpy as np

boids = []

BOIDS = 50  # Number of boids

DIMENSIONS = 2  # Number of dimensions (either 2 or 3)
MAX_SPEED = 2  # Maximum speed of our boids
MAX_FORCE = 0.1  # Maximum forces our boids can experience

BORDER = 200  # Size of the simulation field

W_SEP = 2  # The separation coefficient
W_ALI = 1.1  # The alignment coefficient
W_COH = 1  # The cohesion coefficient

DESIRED_SEP = 30  # The desired distance between the boids

NEIGHBOUR_DIST = 50  # The maximum distance for finding neighbors
NEIGHBOUR_CACHE_LIFE = 10  # We update the neighbors every x frames

STARTING_DISPERSION = 200  # The size of the spawn field

ENABLE_INTERACTIVITY = True  # Whether the boids should interact with one another

FIELD_OF_VIEW = 270  # The field of view of our boids, in degrees (0-360)

DEBUG = False


def angle_between(v1, v2):
    """
    Computes the angle between two vectors (not oriented).
    """
    if not np.any(v1) or not np.any(v2):
        return 0

    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def oriented_angle_between(v1, v2):
    """
    Computes the oriented angle between two vectors.
    """
    if not np.any(v1) or not np.any(v2):
        return 0

    v1_u = normalize(v1)
    v2_u = normalize(v2)

    sign = np.array(np.sign(np.cross(v1_u, v2_u)))
    return np.arccos(np.dot(v1_u, v2_u)) * sign


def normalize(v):
    """
    Normalizes the given vector so that it has a norm of 1
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def limit(v, maximum):
    """
    Caps the vector so that its norm does not exceed len
    """
    if np.linalg.norm(v) > maximum:
        v = maximum * normalize(v)

    return v


class Boid:
    """
    An object that will represent our Boid.
    It is defined by its position, acceleration, radius, angle, and velocity.
    
    Methods :
      - tick : compute the Boid's new parameters after one time tick
      - find_around : Updates the list of neighbours
      - flock : Computes the resulting vector from all behaviours
      - separate : Computes the resulting vector from the "separate" behaviour
      - align : Computes the resulting vector from the "align" behaviour
      - cohesion : Computes the resulting vector from the "cohesion" behaviour
      - seek : Computes the resulting vector when seeking a target
      - update : Computes the new position/velocity based on position/velocity/acceleration
      - borders : If the boid crosses the border, put him on the other side
      - render :
    """

    def __init__(self, parent):
        self.parent = parent
        self.position: np.array = np.random.random(DIMENSIONS) * STARTING_DISPERSION * 2 - STARTING_DISPERSION
        self.acceleration: np.array = np.zeros(DIMENSIONS)
        self.r = 2

        angle = np.random.random() * 2 * np.pi

        self.velocity: np.array = np.array([np.cos(angle), np.sin(angle)])
        self.velocity: np.array = [*self.velocity, *np.zeros(DIMENSIONS - len(self.velocity))]
        # append missing dimensions if any

        self.neighboursCacheLife = NEIGHBOUR_CACHE_LIFE
        self.neighboursCache = []

    def tick(self, boids):

        # self.acceleration = (self.seek(np.zeros(2)))

        if ENABLE_INTERACTIVITY:
            self.position = np.array([self.parent.location[0], self.parent.location[1]])

        self.flock(boids)
        self.update()
        self.borders()
        self.render()

    def find_around(self, boids):
        for b in boids:
            d = np.linalg.norm(b.position - self.position)
            if 0 < d < NEIGHBOUR_DIST:
                # The boid is inside the right distance range
                if angle_between(self.velocity, b.position - self.position) * 180 / np.pi < FIELD_OF_VIEW / 2:
                    # The boid is inside the field of view
                    yield b

    def flock(self, boids):

        self.neighboursCacheLife -= 1
        if self.neighboursCacheLife <= 0:
            self.neighboursCache = list(self.find_around(boids))
            self.neighboursCacheLife = NEIGHBOUR_CACHE_LIFE

        sep = self.separate(self.neighboursCache)
        ali = self.align(self.neighboursCache)
        coh = self.cohesion(self.neighboursCache)

        sep *= W_SEP
        ali *= W_ALI
        coh *= W_COH

        self.acceleration += sep + ali + coh

    def separate(self, boids):
        steer = np.zeros(DIMENSIONS)
        count = 0
        for b in boids:
            d = np.linalg.norm(b.position - self.position)
            if 0 < d < DESIRED_SEP:
                diff = self.position - b.position
                diff = normalize(diff)
                steer += diff
                count += 1

        if count > 0:
            steer /= float(count)

        if np.any(steer):
            steer = normalize(steer)
            steer *= MAX_SPEED
            steer -= self.velocity
            steer = limit(steer, MAX_FORCE)

        return steer

    def align(self, boids):
        sum = np.zeros(DIMENSIONS)
        count = 0

        for b in boids:
            d = np.linalg.norm(self.position - b.position)
            if 0 < d < NEIGHBOUR_DIST:
                sum += b.velocity
                count += 1

        if count > 0:
            sum = normalize(sum) * MAX_SPEED
            steer = sum - self.velocity
            steer = limit(steer, MAX_FORCE)
            return steer

        return np.zeros(DIMENSIONS)

    def cohesion(self, boids):
        sum = 0
        count = 0

        for b in boids:
            d = np.linalg.norm(self.position - b.position)
            if 0 < d < NEIGHBOUR_DIST:
                sum += b.position
                count += 1

        if count > 0:
            sum /= count
            return self.seek(sum)

        return np.zeros(DIMENSIONS)

    def seek(self, target: np.array):
        desired = target - self.position
        desired = normalize(desired) * MAX_SPEED
        steer = desired - self.velocity

        if np.linalg.norm(steer) > MAX_FORCE:
            steer = normalize(steer) * MAX_FORCE

        return steer

    def update(self):
        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, -MAX_SPEED, +MAX_SPEED)
        self.position += self.velocity
        self.acceleration *= 0

    def borders(self):
        for i in range(DIMENSIONS):
            if self.position[i] > BORDER or self.position[i] < -BORDER:
                self.position[i] = -self.position[i]

    def render(self):
        self.parent.location = self.position[0], self.position[1], 0
        self.parent.rotation_euler[2] = oriented_angle_between([1, 0], self.velocity)


def globalTick(scene):
    """
    Calls tick() for every boid on scene.
    """
    for b in boids:
        b.tick(boids)


def register():
    """
    Calls globalTick every frame of the simulation.
    """
    bpy.app.handlers.frame_change_post.append(globalTick)


def unregister():
    """
    Stops calling globalTick every frame.
    """
    bpy.app.handlers.frame_change_post.remove(globalTick)


def reload():
    """
    
    """
    path = bpy.data.filepath
    bpy.ops.wm.save_mainfile()
    bpy.ops.wm.open_mainfile(filepath=path)


# Remove all objects from scene

bpy.ops.object.select_all(action='SELECT')
bpy.data.objects['template'].select_set(False)
bpy.ops.object.delete()


def move_to_collection(old_collection, new_collection, object):
    new_collection.objects.link(object)
    old_collection.objects.unlink(object)
    

for i in range(BOIDS):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['template'].select_set(True)
    bpy.ops.object.duplicate()
    bpy.data.objects['template'].select_set(False)
    move_to_collection(bpy.context.scene.collection, bpy.data.collections['boids'], bpy.context.selected_objects[0])
    


# Create Boid objects for every cone
for object in bpy.data.objects:
    
    if object.name == 'template':
        continue
    
    object.name = 'boid'
    
    b = Boid(parent=object)
    boids.append(b)
    b.render()

# Links the simulation with Blender's animation
register()
