import pygame
import numpy as np
import math 
from numba import jit, guvectorize, int32, float64
from numba.typed import Dict 
import threading
import concurrent.futures
import copy 
import precalc_data
# Initialize the game window
# https://www.youtube.com/watch?v=p4YirERTVF0

# setup 
pygame.init()
scale = 0.3 #time scale 
size = 800 # spawn range  
interact_range = size/12
window_size = (1920, 1080)
screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN)
import time 

global hashmap 
hashmap = np.full(200000, np.nan, dtype='f,f')
precalced_data = precalc_data.compute_values()
for i in range(len(precalced_data)):
    for j in range(len(precalced_data[i])):
        hash = int(0.5*(i+j)*(i+j+1)+j)
        hashmap[hash] = (precalced_data[i][j][0],precalced_data[i][j][1])


# calculate the sector that a particle is in (based on distance split into 32 sectors)
@jit(nopython=True)
def distance_calc(other_dist, interact_range):
    other_dist = other_dist/(math.sqrt(2*interact_range**2)/32)
    dx = other_dist%1 
    sector = other_dist-dx 
    return dx,sector


# particle class 
class Color(pygame.sprite.Sprite):
    def __init__(self, r, g, b, attraction_matrix):
        super().__init__()
        self.color = (r, g, b)
        self.attraction_matrix = copy.copy(attraction_matrix)
        self.vel = [0,0]
        if r == 255 and g == 255: # setting type 
            self.type = "white"
            self.code = 0
        elif r == 255:
            self.type = "red"
            self.code = 2 
        elif g == 255:
            self.type = "green"
            self.code =3 
        elif b == 255:
            self.type = "blue"
            self.code =1 
        elif r == 200:
            self.type = "purple"
            self.code = 4
        elif b == 200:
            self.type = "cyan"
            self.code = 5 
        
        self.genes = {"breedchance":np.random.rand()/10, 
                      "white":0,
                      "blue":0,
                      "red":0, 
                      "green":0,
                      "purple":0, 
                      "cyan":0,
                      "shareRate":copy.copy(shareRate)[self.type]}
        self.hp = 100
        # defining genetic drift 
        drift = [abs(self.genes[k]) for k in self.genes if k != "breedchance"]
        drift = np.sum(drift)/(1.5*6)
        # create sprite image 
        self.radius = 2
        self.image = pygame.Surface((self.radius*2, self.radius*2))
        self.image.fill((0,0,0))
        self.image.set_colorkey((0,0,0))
        pygame.draw.circle(self.image, (r,g,b), (self.radius, self.radius), self.radius)
        pygame.draw.circle(self.image, (255*drift,255*drift,255*drift), (self.radius, self.radius), self.radius/2)
        self.rect = self.image.get_rect()
        
    def set_genes(self,genes):
        sr = self.genes["shareRate"]
        self.genes = genes
        self.genes["shareRate"] += sr 
        drift = [abs(self.genes[k]) for k in self.genes if k != "breedchance"]
        drift = np.sum(drift)/(1.5*6)
        drift = drift*5
        pygame.draw.circle(self.image, (min(255,255*drift),min(255,255*drift),min(255,255*drift)), (self.radius, self.radius), self.radius/2)
    
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
global shareRate
shareRate = {
    "white":np.random.rand()*0.5,
    "blue":np.random.rand()*0.5,
    "red":np.random.rand()*0.5, 
    "green":np.random.rand()*0.5,
    "purple":np.random.rand()*0.5, 
    "cyan":np.random.rand()*0.5,
    }
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
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 0, 255, matrices[1]))
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(255, 0, 0, matrices[2]))
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 255, 0, matrices[3]))
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(200, 0, 200, matrices[4]))
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
    particles.append(Color(0, 200, 200, matrices[5]))
    positions.append([np.random.uniform(0,size),np.random.uniform(0,size), 0, 0])
    spriteList.add(particles[-1])
positions = np.asarray(positions)
@jit(nopython=True)
def in_range(dist, interact_range):
    if dist < 2*interact_range**2: 
        return True
    else:
        return False 

# calculate the sector that a particle is in (based on distance split into 32 sectors)
@jit(nopython=True)
def distance_calc(other_dist, interact_range):
    other_dist = math.sqrt(other_dist/((2*interact_range**2)))*32
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


@guvectorize([(float64[:], float64[:],  int32, float64[:], float64[:])],'(m),(n),(),(p)->(n)', nopython=True)
def mathStuff(ipos, j, interact_range, matrix, vels2):
    vels2[2] = 0 
    vels2[3] = 0
    if (ipos[0] != j[0] and ipos[1] != j[1]):
        delta1 = (ipos[0]-j[0])
        delta2 = (ipos[1]-j[1]) 
        dist = delta1**2 + delta2**2 #dist squared for computation timee
        if dist < 2*interact_range**2: # if j is in range 
            #dx,sector = distance_calc(dist, interact_range) # get sector the particles are in
            other_dist = math.sqrt(dist/((2*interact_range**2)))*32
            dx = other_dist%1 
            sector = other_dist-dx 

            if sector <= 4: # if close by, calculate remainder of the sector and repel based on that + inverse square law 
                #dy/dx = 3
                dx = (dx + sector +1)/5
                vel_mag = -(4/(dx)**2)-0.5
            elif sector <= 16: #medium distance: scale from 0 attraction to attraction matrrix attraction
                dx = dx + sector-5
                grad = matrix[int(j[4])]/12
                vel_mag = grad*dx 
            elif sector <= 26: # large distance: scale from attraction matrix attraction down to 0 
                dx = dx + sector-17
                grad = matrix[int(j[4])]
                vel_mag = grad -(grad/10)*dx
            else: # too far - 0 velocity
                vel_mag= 0 
            # vels = precalced_data[int(delta2)+101][int(delta1)+101]*vel_mag
            m = int(delta2) +101 
            n = int(delta1) + 101 
            hash = 0.5*(m+n)*(m+n+1)+n
            vels = hashmap[int(hash)]
            vels = (vels[0]*vel_mag, vels[1]*vel_mag)
            # print(vels)
            # vels = trig(delta2,delta1, vel_mag) # get what the velocity should be 
            # print(vels)   
            vels2[0] = vels[0]
            vels2[1] = vels[1]
        else:
            vels2[0] = 0.0
            vels2[1] = 0.0
        if dist < 2*(interact_range/4)**2:
            vels2[2] = int(j[4])+1
    else:
        vels2[0] = 0.0
        vels2[1] = 0.0

@jit(nopython=True)
def share(v, i_hp, i_genes_sr ):
    quantity = 0 
    close2 = -1 
    if i_hp >= 100*(1-i_genes_sr):
        close = v[:,2]
        count = 0 
        for x in close:
            if x != 0:
                close2 = count 
                break
            count += 1 
        if close2 != -1: 
            quantity = 1.5*scale

    return close2, quantity
@jit(nopython=True)
def fastsum(v):
    ans = np.zeros(len(v[0]))
    for i in v:
        ans += i
    return ans 

## vectorization params 
colourToNumber = {"white":0,"blue":1,"red":2,"green":3,"purple":4,"cyan":5}
types = [colourToNumber[j.type] for j in particles]
def menu(position):
    font = pygame.font.Font('freesansbold.ttf', 50)
    title = font.render("Particle Life Simulation",True, (255,255,255),(0,0,0))
    rect = title.get_rect()
    rect = (1920//3, 1080//6)
    screen.blit(title, rect)
    play_button_pos = [700,1080//3 - 25, 425,100]
    pygame.draw.rect(screen, (81,210,112), play_button_pos,border_radius=25)
    font = pygame.font.Font('freesansbold.ttf', 40)
    start_text = font.render("start",True, (240,116,35),(81,210,112))
    start_text_rect = start_text.get_rect()
    start_text_rect = (860, 1080//3)
    screen.blit(start_text, start_text_rect)

    if position[0] > play_button_pos[0] and position[0] < play_button_pos[0]+play_button_pos[2] and position[1] > play_button_pos[1] and position[1] < play_button_pos[1]+play_button_pos[3]:
        return True
    return False

    
# Main game loop
running = False
data_major = {"Particle Loop":[], "Kill":[],"Create Children":[]}
data_loop = {"Get Movement":[], "Share Food":[], "Breed":[]}
fps = [] 
clock = pygame.time.Clock()

# Testing 
matrices_test = [{
    "white":0,
    "blue": 4000,
    "red": -1, 
    "green": -1,
    "purple": -1,
    "cyan": -1,
    },{
    "white":0,
    "blue":0,
    "red":-1,
    "green":-1,
    "purple":-1,
    "cyan":-1,
    }]
particles_test = [Color(255, 255, 255, matrices_test[0]), Color(0, 0, 255, matrices_test[1])] 
TestList = pygame.sprite.Group()
TestList.add(particles_test[0])
TestList.add(particles_test[1])
positions_test = [[1920//2,1080//2, 0,0],[1920//2+15,1080//2+5.5,0,0]]
types_test = [0,1]
interact_range_test = interact_range
offset = 15.9
up = -1 
dist_test = 16 
test_mode = False
while True:
    if test_mode == False:
        break
    for event in pygame.event.get(): # allow quitting 
        if event.type == pygame.QUIT:
            pygame.quit()
    screen.fill((0,0,0))        
    particles_test[0].timestep(positions_test[0][0],positions_test[0][1])
    particles_test[1].timestep(positions_test[1][0],positions_test[1][1])
    expand_pos = np.column_stack(([positions_test[1]], [types_test[1]]))
    matrixes = [particles_test[0].attraction_matrix[key] for key in particles_test[0].attraction_matrix] # negligible time 
    ipos = [(y) for y in positions_test[0]]
    v = np.asarray(mathStuff(ipos,expand_pos, int(interact_range_test), matrixes))
    v = fastsum(v)
    mag = math.sqrt(v[0]**2 + v[1]**2)
    if mag > 0.1:
        v[0] = v[0]*50/mag
        v[1] = v[1]*50/mag
    pygame.draw.line(screen, (255,0,0),(positions_test[0][0]+1,positions_test[0][1]+1),(positions_test[0][0]+v[0]+1,positions_test[0][1]+v[1]+1),2)
    TestList.draw(screen)
    pygame.display.flip()
    offset -= 0.01
    positions_test[1] = [1920//2,1080//2, 0,0]
    positions_test[1][0] += offset 
    positions_test[1][1] += math.sqrt(dist_test**2 - offset**2)*up
    
    if round(offset,3) == -dist_test+0.02 and up == -1 :
        offset = dist_test-0.02
        up = 1 
    elif round(offset,3) == -dist_test+0.02 and dist_test == 16:
        dist_test = 4
        offset = dist_test-0.02
        up = -1 
    elif round(offset,3) == -dist_test+0.02:
        break 
    time.sleep(0.00001*abs(offset)**1.4)

while True:
    screen.fill((0, 0, 0))
    if running == False:
        position = (-1,-1)
        for event in pygame.event.get(): # allow quitting 
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONUP:
                position = pygame.mouse.get_pos() 
            
        running = menu(position) 
    elif running == "debug":
        for event in pygame.event.get(): # allow quitting 
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN: # speed up or slow down time with q and e 
                if event.key == pygame.K_9:
                    running == True
        #major data
        loop_data = [] 
        for i,data in enumerate(data_major["Particle Loop"]):
            loop_data.append((i,350-data))
        pygame.draw.aalines(screen, (255,0,0),False,loop_data)
        loop_data = [] 
        for i,data in enumerate(data_major["Kill"]):
            loop_data.append((i,350-data))
        pygame.draw.aalines(screen, (0,255,0),False,loop_data)
        loop_data = [] 
        for i,data in enumerate(data_major["Create Children"]):
            loop_data.append((i,350-data))
        pygame.draw.aalines(screen, (0,0,255),False,loop_data)

        # loop data
        loop_data = [] 
        for i,data in enumerate(data_loop["Get Movement"]):
            loop_data.append((i,700-data))
        pygame.draw.aalines(screen, (255,0,0),False,loop_data)
        loop_data = [] 
        for i,data in enumerate(data_loop["Share Food"]):
            loop_data.append((i,700-data))
        pygame.draw.aalines(screen, (0,255,0),False,loop_data)
        loop_data = [] 
        for i,data in enumerate(data_loop["Breed"]):
            loop_data.append((i,700-data))
        pygame.draw.aalines(screen, (0,0,255),False,loop_data)
        
        #fps data 
        loop_data = [] 
        for i,data in enumerate(fps):
            loop_data.append((i,1050-data))
        pygame.draw.aalines(screen, (255,0,0),False,loop_data)


    else:
        for event in pygame.event.get(): # allow quitting 
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN: # speed up or slow down time with q and e 
                if event.key == pygame.K_q:
                    scale = scale*0.9
                if event.key == pygame.K_e:
                    scale = scale*1.1
                if event.key == pygame.K_9:
                    running = "debug"
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
        
        children = []
        ckpt = time.time_ns() 
        data_store = [0,0,0]
        expand_pos = np.column_stack((positions, types))
        for c1,i in enumerate(particles): # for each particle
            movement = time.time_ns()
            i.timestep(positions[c1][0],positions[c1][1]) # update position  - negligible time 
            
            positions[c1][2] = 0  
            positions[c1][3] = 0
            
            matrixes = [i.attraction_matrix[key] for key in i.attraction_matrix] # negligible time 
            
            
            #expand_pos = np.concatenate((positions,np.asarray([types]).T), axis=1)   
            ipos = [(y) for y in positions[c1]]
            v = np.asarray(mathStuff(ipos,expand_pos, int(interact_range), matrixes)) # low time
            movement = time.time_ns()-movement
            data_store[0] += movement
            sharing_time = time.time_ns()
            close = [-1]
            sharing = share(v, i.hp, i.genes["shareRate"])
            if sharing[0] != -1:
                particles[sharing[0]].hp += sharing[1] 
                i.hp -= sharing[1] 
            sharing_time = time.time_ns() - sharing_time
            data_store[1] += sharing_time
            breeding = time.time_ns()
            if i.genes["breedchance"]*scale >= 0.001 and np.random.randint(0,1/((i.genes["breedchance"]*scale*3)**2)) == 0 and i.hp >= 30:
                if close[0] != -1:
                    parents = [a for a,x in enumerate(close) if x != 0 and x-1 == int(types[c1])]
                else: 
                    parents = [a for a,x in enumerate(v) if x[2] != 0 and x[2]-1 == int(types[c1])]
                if len(parents) != 0:
                    parent = np.random.choice(parents)
                    if particles[parent].hp >= 30:
                        i.hp -= 30
                        particles[parent].hp -= 30 
                        newGenes = {}
                        for gene in i.genes:
                            prop = np.random.rand()
                            value = (i.genes[gene]*prop+particles[parent].genes[gene]*(1-prop))
                            if gene != "breedchance":
                                value += np.random.rand()/10 - 0.05 
                            newGenes[gene]=value
                        children.append([i.color,newGenes,i.code,positions[parent]])
            
            
            v = fastsum(v)

            i.hp -=scale*(math.sqrt(v[0]**2+v[1]**2)/10)
            
            positions[c1][0] += v[0]*scale
            positions[c1][1] += v[1]*scale
            expand_pos[c1][0] = positions[c1][0]
            expand_pos[c1][1] = positions[c1][1]
            
            breeding = time.time_ns()-breeding
            data_store[2] += breeding     
                   
        ckpt = time.time_ns()-ckpt
        data_major["Particle Loop"].append(ckpt)
        data_loop["Get Movement"].append(data_store[0])
        data_loop["Share Food"].append(data_store[1])
        data_loop["Breed"].append(data_store[2])
        ckpt = time.time_ns()
        offset = 0 
        for c1,i in enumerate(particles):
            if i.hp <= 0:
                i.kill()
                #del i 
                del types[c1-offset]
                positions = np.delete(positions, c1-offset, 0)
                del particles[c1-offset] 
                offset += 1
                break
        ckpt = time.time_ns() - ckpt 
        data_major["Kill"].append(ckpt)
        spriteList.draw(screen) # display particles
        ckpt = time.time_ns()
        for i in children:
            particles.append(Color(i[0][0],i[0][1],i[0][2],matrices[i[2]]))
            particles[-1].set_genes(i[1])
            for col in particles[-1].attraction_matrix:
                particles[-1].attraction_matrix[col] += particles[-1].genes[col]
            pos = [i[3][0]+0.12,i[3][1]-0.3,i[3][2],i[3][3]]
            positions = np.append(positions,[pos], axis = 0)
            spriteList.add(particles[-1])
            types.append(i[2])
        ckpt = time.time_ns() - ckpt 
        data_major["Create Children"].append(ckpt)
        fps_now = clock.get_fps()
        fps.append(fps_now)
        font = pygame.font.SysFont("Arial" , 18 , bold = True)
        fps_text = font.render(str(int(fps_now)), 1, (255,255,255))
        screen.blit(fps_text,(1700,150))
        if running == "debug": #data processing - normalising between 0 and 300 and applying a convolution 
            #major
            total_data = []
            for key in data_major:
                total_data.append(data_major[key])
            total_data = np.asarray(total_data)
            total_data = total_data.flatten()
            total_data = np.convolve(total_data,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 'same')
            for key in data_major: #normalise between 0 and 300 
                data_major[key] = ((data_major[key] - np.min(total_data))/(np.max(total_data)-np.min(total_data)))*300
                data_major[key] = np.convolve(data_major[key],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 'same')
            #loop
            total_data = []
            for key in data_loop:
                total_data.append(data_loop[key])
            total_data = np.asarray(total_data)
            total_data = total_data.flatten()
            total_data = np.convolve(total_data,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 'same')
            for key in data_loop: #normalise between 0 and 300 
                data_loop[key] = ((data_loop[key] - np.min(total_data))/(np.max(total_data)-np.min(total_data)))*300
                data_loop[key] = np.convolve(data_loop[key],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 'same')
            #fps 
            fps = np.convolve(fps,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 'same')
            fps = ((fps - np.min(fps))/(np.max(fps)-np.min(fps)))*300
    clock.tick(144)
    pygame.display.flip()

# Quit the game
