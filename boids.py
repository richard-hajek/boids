import numpy as np
import bpy

boids = []

BOIDS = 50

DIMENSIONS = 2
MAX_SPEED = 2
MAX_FORCE = 0.1

BORDER = 200

W_SEP = 1.5
W_ALI = 1.1 
W_COH = 1

DESIRED_SEP = 25

NEIGHBOUR_DIST = 50

STARTING_DISPERSION = 200

ENABLE_INTERACTIVITY = True

NEIGHBOURS_CACHE_LIFE = 10

DEBUG = False

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    
    if not np.any(v1) or not np.any(v2):
        return 0
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def limit(v, len):
    
    if np.linalg.norm(v) > len:
        v = len * normalize(v)
    
    return v

class Boid:
    
    def __init__(self, parent):
        self.parent = parent
        self.position = np.random.random(DIMENSIONS) * STARTING_DISPERSION * 2 - STARTING_DISPERSION
        self.acceleration = np.zeros(DIMENSIONS)
        self.r = 2
        angle = np.random.randint(45) * 8
        self.velocity = np.array([np.cos(angle), np.sin(angle)])
        self.velocity = [*self.velocity, *np.zeros(DIMENSIONS - len(self.velocity))] # append missing dimensions if anny
        self.neighboursLife = NEIGHBOURS_CACHE_LIFE
        self.neighbours = []
        
    
    def tick(self, boids):
        
        #self.acceleration = (self.seek(np.zeros(2)))
        
        if ENABLE_INTERACTIVITY:
            self.position = np.array([self.parent.location[0], self.parent.location[1]])
        
        self.flock(boids)
        self.update()
        self.borders()
        self.render()
    
    
    def find_around(self, boids):
        for b in boids:
            d = np.linalg.norm(b.position - self.position)
            if d > 0 and d < NEIGHBOUR_DIST:
                yield b
    
    
    def flock(self, boids):
        
        self.neighboursLife -= 1
        if self.neighboursLife <= 0:
            self.neighbours = list(self.find_around(boids))
            self.neighboursLife = NEIGHBOURS_CACHE_LIFE
        
        
        sep = self.separate(self.neighbours)
        ali = self.align(self.neighbours)
        coh = self.cohesion(self.neighbours)
        
        sep *= W_SEP
        ali *= W_ALI
        coh *= W_COH
        
        self.acceleration += sep + ali + coh
    
    def separate(self, boids):
        steer = np.zeros(DIMENSIONS)
        count = 0
        for b in boids:
            d = np.linalg.norm(b.position - self.position)
            if ((d > 0) and (d < DESIRED_SEP)):
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
            if ((d > 0) and (d < NEIGHBOUR_DIST)  ):
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
            if d > 0 and d < NEIGHBOUR_DIST:
                sum += b.position
                count += 1
        
        if count > 0:
            sum /= count
            return self.seek(sum)
        
        return np.zeros(DIMENSIONS)
    
    def seek(self, target):
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
        self.parent.rotation_euler[2] = angle_between(self.velocity,  [1, 0])

def globalTick(scene):
    for b in boids:
        b.tick(boids)

def register():
    bpy.app.handlers.frame_change_post.append(globalTick)

def unregister():
    bpy.app.handlers.frame_change_post.remove(globalTick)

def reload():
    path = bpy.data.filepath
    bpy.ops.wm.save_mainfile()
    bpy.ops.wm.open_mainfile(filepath=path)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete() 

for i in range(BOIDS):  
    #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    bpy.ops.mesh.primitive_cone_add(location=(0.0,0.0,0.0), enter_editmode=False, align='WORLD', radius1=3, depth=12, radius2=0.1, rotation=(0,3.1415/2,0))


for object in bpy.data.objects:
    b = Boid(parent=object)
    boids.append(b)
    b.render()

register()    