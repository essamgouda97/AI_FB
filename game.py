""" 
Created by: Essam Gouda (following Tech with Tim tutorial: https://www.youtube.com/watch?v=MMxFDaIOHsE&list=PLzMcBGfZo4-lwGZWXz5Qgta_YNX3_vLS2)
Description: An AI playing the famous flappy bird. Familiarizing myself with pygame and genetic algorithm NEAT

The idea here is similar to natural selection, each bird has its own NN and the best birds are selected
and breeded and so on, keep repeating until performances are satisfactory.

NEAT conf:
    - Inputs -> Bird.y, top_pipe distance from bird, bottom_pipe distance from bird
    - Outputs -> 1 neuron, either jump or don't jump (binary)
    - Activation func -> TanH (if > 0.5 jump, otherwise don't)
    - Population size -> 100 birds (how many birds per generation)
    - Fitness Function -> evaluating how good these birds are, here we use distance (how far a bird goes)
    - Max generations -> 30

"""

import pygame
import neat
import time
import os
import random
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
            pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
            pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))] #storing bird images in sequence of flying animation

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

#fonts
STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    # constants for bird (can be tweaked and changed)
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25 #how much does the bird tilt
    ROT_VEL = 20 #how much do we rotate on each frame
    ANIMATION_TIME = 5 #how fast will the bird flap its wings

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel =  0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self): #called to make bird fly/jump
        self.vel = -10.5 #negative velocity to go upwards as pygame 0,0 point is top left
        self.tick_count = 0 #keeps track on when we last jumped
        self.height = self.y

    def move(self): #called every single frame to move bird, calculates how much they need to move
        self.tick_count += 1 #a tick happened and a frame went by

        d = self.vel*self.tick_count + 1.5*self.tick_count**2 #displacement = velocity*time, shows number of pixels moved (-ve upwards/+ve downwards)

        if d >= 16: #stop accelration
            d = 16

        if d < 0: #fine tunes upwards more
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50: #checks bird position to adjust tilting (upwards tilting)
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90: #for falling downward tilts
                self.tilt -= self.ROT_VEL 

    
    def draw(self, win):
        self.img_count += 1 #keep track of ticks of showing images


        #ANIMATION of bird on pygame window (might be improved later)
        if self.img_count < self.ANIMATION_TIME: #show the first animation as long as its less then animation time
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:#show second image
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3: #show third image
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4: #get back to second image
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*4 + 1:#reset to starting position
            self.img = self.IMGS[0]
            self.img_count = 0 #reset image count

        if self.tilt <= -80:
            self.img = self.IMGS[1] #shows its nose diving down
            self.img_count = self.ANIMATION_TIME*2

        #rotate image
        rotated_image = pygame.transform.rotate(self.img, self.tilt) #rotate the image
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self): #for collision with pipes
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200 #space between pipes
    VEL = 5 #velocity of pipes moving, note the env is moving not the bird

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0 #where top of pipe is drawn
        self.bottom = 0 #where bottom of pipe is drawn
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True) #flip the top pipe
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False #if the bird passed the pipe, collision
        self.set_height() #define where top and bottom one and how tall it is and where is the gap, randomly defined

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x  -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird): #checks collision of bird collision box and pipe collision box, but using masks instead of normal boxes only (more accurate collision)
        #pixel perfect collision
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        #how far away these masks from each other
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset) #tells overlap point, return None if they don't collide        
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True #collides

        return False

class Base: #base ground in game
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self): #keeps moving the background that they appear connected as if its one continuos image
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
        

def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG, (0,0)) #blit means draw on the window
    
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()
    

def main(genomes, config): #fitness function
    global GEN
    GEN += 1

    nets = [] #NN of birds
    ge = []
    birds = []

    for _, g in genomes: #contain genome ID and object (tuple)
        net = neat.nn.FeedForwardNetwork.create(g, config) #setup NN for each bird
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0 #initially fitness = 0
        ge.append(g)

    
    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0 #score of the pipes passed

    run = True
    while run:
        clock.tick(30) #at most 30 ticks per second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()    
                quit()

        #bird movement
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width(): #if we passed the first pipe check the second one
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1 #every second it stays alive it gets 1 fitness point

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))) #inputs for NN produces output
            
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = [] #remove unwanted pipes
        for pipe in pipes:
            for x, bird in enumerate(birds): #enumerate get position of bird in list
                if pipe.collide(bird):
                    ge[x].fitness -= 1 #removes 1 from fitness score of bird that hits pipe
                    birds.pop(x) #removes item from list
                    nets.pop(x)
                    ge.pop(x)
            
                if not pipe.passed and pipe.x < bird.x: #checks if pipe is passed
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe: #add pipe if pipe is passed
            score += 1
            for g in ge:
                g.fitness += 5 #to encourage birds to pass from pipes
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0: #if it hits the base
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()        
        draw_window(win, birds, pipes, base, score, GEN)



def run(config_path):
    #load up config file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config) #setting population

    #sets output
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50) #fitness function (calls main function 50 times)  and number of generations to run

    #TODO add pickle to save winner

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__) #gives path of current dir
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
