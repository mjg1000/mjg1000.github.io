import pygame
import numpy as np
import math 
from numba import jit

# Initialize the game window
pygame.init()
scale = 0.1
size = 800
interact_range = size/8
window_size = (1920, 1080)
screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN)

@jit(nopython=True)
def distance_calc(other_dist, interact_range):
    other_dist = other_dist/(math.sqrt(2*interact_range**2)/32)
    dx = other_dist%1 
    sector = other_dist//1 
    return dx,sector

class Color(pygame.sprite.Sprite):
    def __init__(self, r, g, b, attraction_matrix,pos):
        super().__init__()
        self.color = (r, g, b)
        self.attraction_matrix = attraction_matrix
        self.vel = [0,0]
        self.pos = pos
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

        self.radius = 2
        self.image = pygame.Surface((self.radius*2, self.radius*2))
        self.image.fill((0,0,0))
        self.image.set_colorkey((0,0,0))
        pygame.draw.circle(self.image, (r,g,b), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect()
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
        
    
    def update(self, obj, other_dist, size, interact_range):
        other_pos = obj.pos 
        dx,sector = distance_calc(other_dist, interact_range)
        if sector <= 4:
            #dy/dx = 3
            dx = (dx + sector +1)/5
            vel_mag = -1 *(1/dx)**2
        elif sector > 4 and sector <= 16:
            dx = dx + sector-5
            grad = self.attraction_matrix[obj.type]/12
            vel_mag = grad*dx 
        elif sector >16 and sector <= 26:
            dx = dx + sector-16
            grad = self.attraction_matrix[obj.type]
            vel_mag = grad -(grad/10)*dx
        else:
            vel_mag= 0 
        angle = math.atan2((self.pos[1]-other_pos[1]),(self.pos[0]-other_pos[0]+0.001))
        self.vel[0] -= vel_mag*math.cos(angle)
        self.vel[1] -= vel_mag*math.sin(angle)

    def reset_vel(self):
        self.vel = [0,0]
    def timestep(self, scale):
        self.pos[0] += self.vel[0]*scale
        self.pos[1] += self.vel[1]*scale
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
    


# Generate the initial state of the automaton
spriteList = pygame.sprite.Group()
particles = [] 
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

# matrices = [{
#     "white":2,
#     "blue":0.5,
#     "red":0, 
#     "green":0,
#     },{
#     "white":0,
#     "blue":2,
#     "red":0.5, 
#     "green":0,
#     },{
#     "white":0,
#     "blue":0,
#     "red":2, 
#     "green":0.5,
#     },{
#     "white":0.5,
#     "blue":0,
#     "red":0, 
#     "green":2,
#     }
#     ]
for i in range(50): 
    particles.append(Color(255, 255, 255, matrices[0], [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])
    particles.append(Color(0, 0, 255, matrices[1], [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])
    particles.append(Color(255, 0, 0, matrices[2], 
    [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])
    particles.append(Color(0, 255, 0, matrices[3], [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])
    particles.append(Color(200, 0, 200, matrices[4], [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])
    particles.append(Color(0, 200, 200, matrices[5], [np.random.randint(0,size),np.random.randint(0,size)]))
    spriteList.add(particles[-1])

# Main game loop
@jit(nopython=True)
def in_range(dist, interact_range):
    if dist < math.sqrt(2*interact_range**2):
        return True
    else:
        return False 

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                scale = scale*0.9
            if event.key == pygame.K_e:
                scale = scale*1.1
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        for i in particles:
            i.pos[1] += 10 
    if keys[pygame.K_s]:
        for i in particles:
            i.pos[1] -= 10 
    if keys[pygame.K_a]:
        for i in particles:
            i.pos[0] += 10 
    if keys[pygame.K_d]:
        for i in particles:
            i.pos[0] -= 10 
            
            
    
    
    # Render the automaton
    screen.fill((0, 0, 0))
    for i in particles:
        i.reset_vel()
        for j in particles:
            dist = math.dist(i.pos, j.pos)
            if in_range(dist, interact_range) and  i.pos != j.pos:
                i.update(j, dist, size, interact_range)
        i.timestep(scale)
    spriteList.draw(screen)
    
    pygame.display.flip()

# Quit the game
pygame.quit()
