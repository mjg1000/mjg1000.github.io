
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
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\private.png").convert()
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.hp = 1 
        self.pos = [500,500]
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1] 
        self.vel = 100
        self.lastHit = runT
        self.sparks = []
        #self.size = 1 
    def decaySparks(self):
        if len(self.sparks) > 0:
            offset = 0 
            count = 0 
            for i in self.sparks:
                i.rect.x += np.random.randint(-50,50)
                i.rect.y += np.random.randint(-50,50)
                if np.random.randint(0, 10) >= 2:
                    self.sparks[-1].kill()
                    del self.sparks[-1] 
                    offset += 1 
                if count - offset >= len(self.sparks):
                    break
                count += 1 
                

    def loadPos(self):
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]   
        self.decaySparks()
        

    def moveR(self):
        self.pos[0] += self.vel
        self.loadPos()
    def moveL(self):
        self.pos[0] -= self.vel
        self.loadPos()
    def moveD(self):
        self.pos[1] += self.vel 
        self.loadPos()
        if self.rect.y >= 650:
            return True
        else:
            return False
    def hit(self):
        self.lastHit = runT
        for i in range(5):
            self.sparks.append(spark([self.rect.x+np.random.randint(-50,50),self.rect.y+np.random.randint(-50,50)]))
            spriteList.add(self.sparks[-1])
    
class Projectile(pg.sprite.Sprite):
    def __init__(self, vel, damage, pos):
        super().__init__()
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\missile.png").convert()
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
        self.type = "private"
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\private.png").convert()
        self.image.set_colorkey(W)
        self.pos = pos 
        self.loadPos()
        self.hp = 3 
class Commander(Enemy):
    def __init__(self, pos):
        super().__init__()
        self.type = "commander"
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\commander.png").convert()
        self.image.set_colorkey(W)
        self.pos = pos 
        self.loadPos()
        self.hp = 10 
class Creator(Enemy):
    def __init__(self, pos):
        super().__init__()
        self.type = "creator"
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\creator.png").convert()
        self.image.set_colorkey(W)
        self.pos = pos 
        self.loadPos()
        self.hp = 10 
class Missile(Projectile):
    def __init__(self, pos):
        super().__init__(-10, 1, pos)
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\missile.png").convert()
        self.image.set_colorkey(W)
        self.sparks = [] 
    
class Player(pg.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = "player"
        self.maxhp = 5
        self.hp = 5
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\player.png").convert()
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.rect.x = 800
        self.rect.y = 700
        self.vel = 10 
  
        self.bullets = [] 
        self.gold = 0 
        self.dmgMod = 0 
    def moveR(self):
        self.rect.x += self.vel 
    def moveL(self):
        self.rect.x -= self.vel 
    def shoot(self):
        self.bullets.append(riflePulse([self.rect.x, self.rect.y], self.dmgMod ))
        spriteList.add(self.bullets[-1])
    def tick(self):
        for i in self.bullets:
            i.tick()
            hits = pg.sprite.spritecollide(i, spriteList, False)
            for col in hits:
                try:
                    if col.name == "enemy":
                        if runT - col.lastHit > 500:
                            col.hp -= i.dmg
                            col.hit()
                except:
                    pass
        
class spark(pg.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\spark.png")
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
    def __init__(self, pos, dmg):
        super().__init__(10, 1+dmg, pos)
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\riflePulse.png")
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
    
#PowerUps
class powerUp(pg.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.name = ""
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\powerUp.png")
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.rect.x = 500
        self.rect.y = 200
        self.hit = False
        self.cost = 0 
    def tick(self):
        hits = pg.sprite.spritecollide(self, spriteList, False)
        if len(hits) != 1:
            self.hit = True
            print("hits")
class skip(powerUp):
    def __init__(self):
        super().__init__()
        self.name = "skip"
        self.desc = "pass the option provided"
        self.cost = 0
        self.image = pg.image.load("C:\\Users\\mjgar\\Documents\\SpaceInvaders\\powerUp2.png")
        self.image.set_colorkey(W)
        self.rect = self.image.get_rect()
        self.rect.x = 1000
        self.rect.y = 200
        
    def tick(self):
        super().tick()
        if self.hit == True:
            pass
        return True
class dmgBoost(powerUp):
    def __init__(self):
        super().__init__()
        self.name = "damage"
        self.desc = "gives you + 4 dmg"
        self.cost = 15
        
    def tick(self):
        super().tick()
        if self.hit == True:
            if player.gold >= 15:
                player.gold -= 15 
                player.dmgMod += 4 
        return True
class hpBoost(powerUp):
    def __init__(self):
        super().__init__()
        self.name = "health"
        self.desc = "gives you +2 hp"
        self.cost = 3 

        
        
    def tick(self):
        super().tick()
        if self.hit == True:
            if player.gold >= 3:
                player.gold -= 3 
                player.hp += 1 
        return True
class speedBoost(powerUp):
    def __init__(self):
        super().__init__()
        self.name = "speed"
        self.desc = "gives you +10 speed"
        self.cost = 5
        
        
    def tick(self):
        super().tick()
        if self.hit == True:
            if player.gold >= 5:
                player.gold -= 5 
                player.vel += 10
        return True
class goldBoost(powerUp):
    def __init__(self):
        super().__init__()
        self.name = "gold"
        self.desc = "doubles your gold \n and +1 to all stats"
        self.cost = 12
        
        
    def tick(self):
        super().tick()
        if self.hit == True:
            if player.gold >= 12:
                player.gold -= 12
                player.gold = player.gold * 2
                player.hp += 1 
                player.vel += 5 
                player.dmgMod += 1  
            
        return True
global runT
runT = 0 
global width,height
width = 1600
height = 900
size = (width,height)
screen = pg.display.set_mode(size)#, flags=pg.FULLSCREEN)

done = False
clock = pg.time.Clock()
pg.font.init()
font = pg.font.SysFont('OCR-A extended', 128)
goldFont = pg.font.SysFont('ariel', 40)
x = 0 
y = 0 
global player, spriteList, enemies, bullets
bullets = []
enemies = []
player = Player()
spriteList = pg.sprite.Group()
#test = Private([100,100])
level = 0


gameLost = False

def setup0():
    global enemies, bullets, dir, lastMove 
    lastMove = 1  
    enemies = []
    bullets = [] 
    for i in range(5):
        enemies.append(Private([100+i*150,100]))
        spriteList.add(enemies[i])
    spriteList.add(player)
    dir = "r"
def setup2():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(5):
            enemies.append(Private([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def setup4():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(5):
            if i == 2:
                enemies.append(Commander([100+i*150,x*150+100]))
            else:
                enemies.append(Private([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def setup6():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(8):
            if i == 3 or i == 4:
                enemies.append(Commander([100+i*150,x*150+100]))
            else:
                enemies.append(Private([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def setup8():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(8):
            enemies.append(Commander([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def setup10():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(8):
            if i == 3 or i == 4:
                enemies.append(Creator([100+i*150,x*150+100]))
            else:
                enemies.append(Commander([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def setup12():
    global enemies, bullets, dir, lastMove
    lastMove = 1  
    enemies = []
    bullets = [] 
    for x in range(2):
        for i in range(8):

            enemies.append(Creator([100+i*150,x*150+100]))
            spriteList.add(enemies[-1 ])
    dir = "r"
def moveEnemies():
    global lastMove, level, dir, enemies
    gameLost = False
    if runT - lastMove > 600:
        for i in enemies:
            if i.rect.y > 650:
                level = -1
                gameLost = True
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
                if i.moveD() == True:
                    gameLost = True
                    level = -1 

        elif dir == "l":
            dir = "r" 
            for i in enemies:
                if i.moveD() == True:
                    gameLost = True
                    level = -1 
        todel = [] 
        for i in range(len(enemies)):
            if enemies[i].type == "creator":
                if np.random.randint(1,20) == 1:
                    enemyNum = -np.random.randint(1,6)
                
                    enemies.append(Private([enemies[enemyNum].rect.x, enemies[enemyNum].rect.y + 150]))
                    spriteList.add(enemies[-1])
                
            elif np.random.randint(1,5) == 1:
                bullets.append(Missile([enemies[i].pos[0],enemies[i].pos[1]]))
                spriteList.add(bullets[-1])
            if enemies[i].hp <= 0:
                if enemies[i].type == "private":
                    player.gold += 1 
                elif enemies[i].type == "commander":
                    player.gold += 3
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
                        if player.hp <= 0:
                            level = -1 
                            gameLost = True
                except:
                    pass
    return(gameLost, level)
setup0()
print(enemies)
while not done:
    if level != -1:
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
    if level == -1:
        keys = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
        screen.fill(R)
        lostText = font.render("game Over",False,W)
        screen.blit(lostText, (int(width/2), int(height/2)))
    elif level == 0:
        #print(time.perf_counter()-t1)

        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1  
        if len(bullets) != 0:
            for i in bullets:
                i.tick()
        player.tick()
        #Health bar 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/3)*300+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/3)*300+50, 850],[300*(player.maxhp/3)+50, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 2:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 4:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 6:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 8:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 10:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    elif level == 12:
        screen.fill(K)
        spriteList.draw(screen)
        if len(enemies) != 0:
            state = moveEnemies()
            level = state[1] 
            gameLost = state[0]
        else:
            level += 1 
        if len(bullets) != 0:
            for i in bullets:
                i.tick() 
        player.tick() 
        pg.draw.line(screen, G, [50,850], [(player.hp/player.maxhp)*(player.maxhp/10)*800+50, 850], 5)
        pg.draw.line(screen, R, [((player.hp/player.maxhp))*(player.maxhp/10)*800+50, 850],[850, 850],5)
        pg.draw.line(screen, B, [0, 700], [1600, 700], 5)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))
    
    else:
        #shop level 
        try:
            powerup.cost += 0
        except:
            for i in player.bullets:
                i.sparks = [] 
            
            player.bullets = [] 
            enemies = [] 
            spriteList.empty()
            skipbutton = skip()
            spriteList.add(player)
            spriteList.add(skipbutton)
            powerup = np.random.randint(0,4)
            if powerup == 0:
                powerup = dmgBoost()
            elif powerup == 1:
                powerup = speedBoost()
            elif powerup == 2:
                powerup = hpBoost()
            elif powerup == 3:
                powerup = goldBoost()
            spriteList.add(powerup)
        screen.fill(K)
        spriteList.draw(screen)
        goldText = goldFont.render(("gold : " + str(player.gold)), False, (227, 200, 27))
        screen.blit(goldText, (width-200, 100))  
        powerupName = goldFont.render(str(powerup.name), False, W)
        screen.blit(powerupName, (600,250))
        powerupDesc = goldFont.render(str(powerup.desc), False, W)
        screen.blit(powerupDesc, (600,300))
        powerupCost = goldFont.render(("cost : " + str(powerup.cost)), False, W)
        screen.blit(powerupCost, (600,350))
        
        skipName = goldFont.render(str(skipbutton.name), False, W)
        screen.blit(skipName, (1000,250))
        skipDesc = goldFont.render(str(skipbutton.desc), False, W)
        screen.blit(skipDesc, (1000,300))
        skipCost = goldFont.render(("cost : " + str(skipbutton.cost)), False, W)
        screen.blit(skipCost, (1000,350))
        if len(player.bullets) != 0:
            for i in player.bullets:
                i.tick()
        powerup.tick() 
        skipbutton.tick()
        if powerup.hit == True or skipbutton.hit == True:
            powerup.kill()
            skipbutton.kill()
            skipbutton = None
            powerup = None
            level += 1 
            player.dmgMod += 1 
            player.vel += 0
            player.hp = player.maxhp

            exec("setup"+str(level)+"()")
    pg.display.flip()
    runT += clock.tick(60)
    #print(runT)
pg.quit()

    
