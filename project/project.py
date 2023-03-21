import pygame
import numpy as np
import math 
from numba import jit, guvectorize, int32, float64
import threading
import concurrent.futures

# Initialize the game window
# https://www.youtube.com/watch?v=p4YirERTVF0

# setup 
pygame.init()
scale = 1 #time scale 
size = 800 # spawn range  
interact_range = size/12
window_size = (1920, 1080)
screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN)
import time 

# calculate the sector that a particle is in (based on distance split into 32 sectors)
@jit(nopython=True)
def distance_calc(other_dist, interact_range):
    other_dist = other_dist/(math.sqrt(2*interact_range**2)/32)
    dx = other_dist%1 
    sector = other_dist-dx 
    return dx,sector

# get velx and vely from velTotal and posx and posy 
@jit(nopython=True)
def trig(self_pos0, self_pos1, other_pos0, other_pos1, vel_mag):
    a = self_pos1-other_pos1
    b = self_pos0-other_pos0
    try:
        angle = (a)/(b) 
        sign = abs(b)/(b)
    except:
        return(0,0)
    sqrt = math.sqrt(1+angle**2)
    self_vel0 = (-vel_mag/sqrt)*sign
    self_vel1 = self_vel0*angle
    return self_vel0,self_vel1

# particle class 
class Color(pygame.sprite.Sprite):
    def __init__(self, r, g, b, attraction_matrix):
        super().__init__()
        self.color = (r, g, b)
        self.attraction_matrix = attraction_matrix 
        self.vel = [0,0]
        if r == 255 and g == 255:
            self.type = "white"
        elif r == 255:
            self.type = "red"
        elif g == 255:
            self.type = "green"
        elif b == 255:
            self.type = "blue"
        elif r == 200:
            self.type = "purple"
        elif b == 200:
            self.type = "cyan"

        # create sprite image 
        self.radius = 2
        self.image = pygame.Surface((self.radius*2, self.radius*2))
        self.image.fill((0,0,0))
        self.image.set_colorkey((0,0,0))
        pygame.draw.circle(self.image, (r,g,b), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect()
        
    # perform calculations to update velocity  
    def update(self, obj, other_dist, size, interact_range):
        other_pos = obj.pos 
        dx,sector = distance_calc(other_dist, interact_range) # get sector the particles are in 
        if sector <= 4: # if close by, calculate remainder of the sector and repel based on that + inverse square law 
            #dy/dx = 3
            dx = (dx + sector +1)/5
            vel_mag = -1 *(1/dx)**2
        elif sector > 4 and sector <= 16: #medium distance: scale from 0 attraction to attraction matrrix attraction
            dx = dx + sector-5
            grad = self.attraction_matrix[obj.type]/12
            vel_mag = grad*dx 
        elif sector >16 and sector <= 26: # large distance: scale from attraction matrix attraction down to 0 
            dx = dx + sector-16
            grad = self.attraction_matrix[obj.type]
            vel_mag = grad -(grad/10)*dx
        else: # too far - 0 velocity
            vel_mag= 0 
        vels = trig(self.pos[0],self.pos[1],other_pos[0], other_pos[1], vel_mag) # get what the velocity should be 
        self.vel[0] += vels[0]
        self.vel[1] += vels[1]

    def reset_vel(self):
        self.vel = [0,0]

    def timestep(self,x,y): # update positions 
        self.rect.x = x
        self.rect.y = y
    

# Generate the initial state of the automaton
spriteList = pygame.sprite.Group()
particles = [] 
# randomly generate attraction matrices for each colour 
matrices = [{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    },{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    },{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    },{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    },{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    },{
    "white":np.random.rand()*1.5-0.5,
    "blue":np.random.rand()*1.5-0.5,
    "red":np.random.rand()*1.5-0.5, 
    "green":np.random.rand()*1.5-0.5,
    "purple":np.random.rand()*1.5-0.5, 
    "cyan":np.random.rand()*1.5-0.5,
    }
    ]

positions = [] 

# add the particles to the sprite list and spawn them randomly 
for i in range(100): 
    particles.append(Color(255, 255, 255, matrices[0]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 0, 255, matrices[1]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(255, 0, 0, matrices[2]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 255, 0, matrices[3]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(200, 0, 200, matrices[4]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 200, 200, matrices[5]))
    positions.append([np.random.randint(0,size),np.random.randint(0,size), 0, 0])
    spriteList.add(particles[-1])
positions = np.asarray(positions)

@jit(nopython=True)
def in_range(dist, interact_range):
    if dist < math.sqrt(2*interact_range**2):
        return True
    else:
        return False 

# calculate the sector that a particle is in (based on distance split into 32 sectors)
@jit(nopython=True)
def distance_calc(other_dist, interact_range):
    other_dist = other_dist/(math.sqrt(2*interact_range**2)/32)
    dx = other_dist%1 
    sector = other_dist-dx 
    return dx,sector

# get velx and vely from velTotal and posx and posy 
@jit(nopython=True)
def trig(a,b, vel_mag):
    angle = a/b
    sign = np.sign(b)
    sqrt = math.sqrt(1+angle**2)
    self_vel0 = (-vel_mag/sqrt)*sign
    self_vel1 = self_vel0*angle
    return self_vel0,self_vel1

@guvectorize([(int32[:], int32[:],  int32, float64[:], float64[:])],'(m),(n),(),(p)->(n)', nopython=True)
def mathStuff(ipos, j, interact_range, matrix, vels2):
    if (ipos[0] != j[0] and ipos[1] != j[1]):
        delta1 = (ipos[0]-j[0])
        delta2 = (ipos[1]-j[1])
        dist = int(math.sqrt(delta1**2 + delta2**2))
        if in_range(dist, interact_range): # if j is in range 
            dx,sector = distance_calc(dist, interact_range) # get sector the particles are in
            if sector <= 4: # if close by, calculate remainder of the sector and repel based on that + inverse square law 
                #dy/dx = 3
                dx = (dx + sector +1)/5
                vel_mag = -(1/dx)**2
            elif sector > 4 and sector <= 16: #medium distance: scale from 0 attraction to attraction matrrix attraction
                dx = dx + sector-5
                grad = matrix[int(j[4])]/12
                vel_mag = grad*dx 
            elif sector >16 and sector <= 26: # large distance: scale from attraction matrix attraction down to 0 
                dx = dx + sector-16
                grad = matrix[int(j[4])]
                vel_mag = grad -(grad/10)*dx
            else: # too far - 0 velocity
                vel_mag= 0 
            vels = trig(delta2,delta1, vel_mag) # get what the velocity should be 
            vels2[0] = vels[0]
            vels2[1] = vels[1]
        else:
            vels2[0] = 0.0
            vels2[1] = 0.0
    else:
        vels2[0] = 0.0
        vels2[1] = 0.0
# vectorization params 
colourToNumber = {"white":0,"blue":1,"red":2,"green":3,"purple":4,"cyan":5}
types = [colourToNumber[j.type] for j in particles]

# Main game loop
running = True
while running:
    for event in pygame.event.get(): # allow quitting 
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN: # speed up or slow down time with q and e 
            if event.key == pygame.K_q:
                scale = scale*0.9
            if event.key == pygame.K_e:
                scale = scale*1.1
    keys = pygame.key.get_pressed() # move camera around by moving all particles 
    if keys[pygame.K_w]:
        for i in range(len(particles)):
            positions[i][1] += 10 
    if keys[pygame.K_s]:
        for i in range(len(particles)):
            positions[i][1] -= 10 
    if keys[pygame.K_a]:
        for i in range(len(particles)):
            positions[i][0] += 10 
    if keys[pygame.K_d]:
        for i in range(len(particles)):
            positions[i][0] -= 10 
    screen.fill((0, 0, 0))
    for c1,i in enumerate(particles): # for each particle i
        i.timestep(positions[c1][0],positions[c1][1]) # update position  
        positions[c1][2] = 0 
        positions[c1][3] = 0
        matrixes = [i.attraction_matrix[key] for key in i.attraction_matrix]
        expand_pos = np.concatenate((positions,np.asarray([types]).T), axis=1)
        v = np.asarray(mathStuff([int(y) for y in positions[c1]],expand_pos, int(interact_range), matrixes))
        v = np.sum(v, axis=0)
        positions[c1][0] += v[0]*scale
        positions[c1][1] += v[1]*scale
    spriteList.draw(screen) # display particles
    pygame.display.flip()

# Quit the game
pygame.quit()
