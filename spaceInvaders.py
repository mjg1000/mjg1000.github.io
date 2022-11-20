 import pygame as pg
import numpy as np 
import math 
import time 
global R,G,B,K,W
R = (255,0,0)
G = (0,255,0)
B = (0,0,255)
K = (0,0,0)
W = (255,255,255)

def angle(x1,x2,y1,y2, vel):
    angle = math.atan((y2-y1)/(x2-x1))
    mag = (vel[0]**2+vel[1]**2)**0.5 
    newvel = [0,0] 
    newvel[1] = math.sin(angle)*mag
    newvel[0] = math.cos(angle)*mag
    return newvel 

def colour(p1,c1,p2,c2):
    c1 = tuple(c/2*((4-p1)/4) for c in c1)
    c2 = tuple(c/2*((4-p2)/4) for c in c2) 
    #print(c1,c2)
    return tuple(c1[i]+c2[i] for i in range(len(c1)))
#colour combines 2 colours taking in brightness for c1 (0-4), colour1, and same for colour 2 
class Enemy(pg.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = "enemy"
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\private.png").convert()
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.hp = 1 
        self.pos = [500,500]
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1] 
        self.vel = 100
        #self.size = 1 
    
    def loadPos(self):
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1] 

    def moveR(self):
        self.pos[0] += self.vel
        self.loadPos()
    def moveL(self):
        self.pos[0] -= self.vel
        self.loadPos()
    def moveD(self):
        self.pos[1] += self.vel 
        self.loadPos()
    
class Projectile(pg.sprite.Sprite):
    def __init__(self, vel, damage, pos):
        super().__init__()
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\missile.png").convert()
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect() 
        
        self.vel = vel
        self.pos = pos 
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
        self.dmg = damage 
        self.colour = (0,0,0)
    def loadPos(self):
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
    def tick(self):
        self.pos[1] -= self.vel
        self.loadPos()
    
class Private(Enemy):
    def __init__(self,pos):
        super().__init__()
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\private.png").convert()
        self.image.set_colorkey(W)
        self.pos = pos 
        self.loadPos()
class Missile(Projectile):
    def __init__(self, pos):
        super().__init__(-10, 1, pos)
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\missile.png").convert()
        self.image.set_colorkey(W)
class Player(pg.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = "player"
        self.maxhp = 10
        self.hp = 5 
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\player.png").convert()
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.rect.x = 800
        self.rect.y = 700
        self.vel = 10 
        self.bullets = [] 
    def moveR(self):
        self.rect.x += self.vel 
    def moveL(self):
        self.rect.x -= self.vel 
    def shoot(self):
        self.bullets.append(riflePulse([self.rect.x, self.rect.y]))
        spriteList.add(self.bullets[-1])
    def tick(self):
        for i in self.bullets:
            i.tick()
            hits = pg.sprite.spritecollide(i, spriteList, False)
            for col in hits:
                try:
                    if col.name == "enemy":
                        col.hp -= i.dmg
                except:
                    pass
        
class spark(pg.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\spark.png")
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.rect.x = pos[0] 
        self.rect.y = pos[1] 
    def tick(self):
        if np.random.randint(0,5) == 3:
            self.rect.x += np.random.randint(-10,10)
            self.rect.y += np.random.randint(-10,10)
        if np.random.randint(0,40) == 20:
            self.kill()
        self.rect.y -= 5
class riflePulse(Projectile):
    def __init__(self, pos):
        super().__init__(10, 10, pos)
        self.image = pg.image.load("C:\\Users\\user\\Documents\\gameImgs\\riflePulse.png")
        self.image.set_colorkey(W)
        self.sparks = [] 
    def tick(self):
        super().tick()
        if np.random.randint(0,20) == 10:
            self.sparks.append(spark([self.rect.x,self.rect.y]))
            spriteList.add(self.sparks[-1])
        for i in self.sparks:
            i.tick()
        if self.rect.y < -200:
            for i in self.sparks:
                i.kill()
                del i
            self.kill()
    
    

width = 1600
height = 900
size = (width,height)
screen = pg.display.set_mode(size)#, flags=pg.FULLSCREEN)

done = False
clock = pg.time.Clock()
pg.font.init()
font = pg.font.SysFont('OCR-A extended', 128)
x = 0 
y = 0 
enemies = []
bullets = [] 
player = Player()
global spriteList
spriteList = pg.sprite.Group()
dir = "r"

#test = Private([100,100])

for i in range(5):
    enemies.append(Private([100+i*150,100]))
    spriteList.add(enemies[i])
spriteList.add(player)
runT = 0 
lastMove = 0
while not done:
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        player.moveR()
    if keys[pg.K_a]:
        player.moveL()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                player.shoot()
            pass
    #print(time.perf_counter()-t1)
    screen.fill(K)
    spriteList.draw(screen)
    if len(enemies) != 0:
        if runT - lastMove > 300:
            lastMove = runT
            if enemies[-1].pos[0] < width-100 and dir == "r":
                for i in enemies:
                    i.moveR()
            elif dir == "l" and enemies[0].pos[0] > 100: 
                for i in enemies:
                    i.moveL()
            elif dir == "r":
                dir = "l"
                for i in enemies:
                    i.moveD()
            elif dir == "l":
                dir = "r" 
                for i in enemies:
                    i.moveD()
            todel = [] 
            for i in range(len(enemies)):
                if np.random.randint(1,10) == 1:
                    bullets.append(Missile([enemies[i].pos[0],enemies[i].pos[1]]))
                    spriteList.add(bullets[-1])
                if enemies[i].hp <= 0:

                    enemies[i].kill()
                    todel.append(i)
            offset = 0 
            
            for i in todel:

                del enemies[i-offset]
                offset += 1 
            for i in bullets:
                if i.rect.y > 1500: 
                    i.kill()
                hits = pg.sprite.spritecollide(i, spriteList, False)
                for x in hits:
                    try:
                        if x.name == "player":
                            player.hp -= i.dmg
                    except:
                        pass
    else:
        pass 
    if len(bullets) != 0:
        for i in bullets:
            i.tick()
    player.tick()
    #Health bar 
    pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*800+50, 850], 5)
    pg.draw.line(screen, R, [((player.hp/player.maxhp))*800+50, 850],[850, 850],5)

    pg.display.flip()
    print(bullets)
    print(enemies)
    print(player.bullets)
    runT += clock.tick(60)
    #print(runT)
pg.quit()

    
