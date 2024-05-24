import random
import pygame as pg
import pathlib


vector=pg.math.Vector2
FPS=30
CELL_SIZE=40


BOARD_COLOR=(100,92,92)
BG_COLOR=(51,25,0)
BOARD_SIZE=BOARD_WIDTH,BOARD_HEIGHT=10,20
BOARD_RESOLUTION=BOARD_WIDTH*CELL_SIZE*1.8,BOARD_HEIGHT*CELL_SIZE
BOARD_RESOLUTION2=BOARD_WIDTH*CELL_SIZE,BOARD_HEIGHT*CELL_SIZE

NEXT_SHAPE_RES=BOARD_WIDTH*CELL_SIZE*1.8-BOARD_WIDTH*CELL_SIZE, BOARD_HEIGHT*CELL_SIZE
NEXT_POSITION=vector(BOARD_WIDTH*1.35, BOARD_HEIGHT/2)
NEXT_POSITIONRECT=vector(BOARD_WIDTH*1.30, BOARD_HEIGHT/2)
INITIALIZE_POSITION = vector(BOARD_WIDTH / 2-1, 1) #TU ZMIANA
MOVEMENTS={"left": vector(-1,0), "right": vector(1, 0), "down": vector(0,1)}
TETROMINOS={
    'T':[(0,0), (-1,0), (1,0), (0,-1)],
    'O':[(0,0), (0,-1), (1,0), (1,-1)],
    'J':[(0,0), (-1,0), (0,-1), (0,-2)],
    'L':[(0,0), (1,0), (0,-1), (0,-2)],
    'I':[(0,0), (0,1), (0,-1), (0,-2)],
    'S':[(0,0), (-1,0), (0,-1), (1,-1)],
    'Z':[(0,0), (1,0), (0,-1), (-1,-1)],
    
}
COLORS=["blue", "pink", "yellow", "green"]
# Ładowanie zdjec menu głownego
background=pg.image.load('images/tetris_background.jpg')
background_scaled=pg.transform.scale(background, BOARD_RESOLUTION)


tetris_logo=pg.image.load('images/Tetris_logo.png')
scaled_logo=pg.transform.scale(tetris_logo, (BOARD_WIDTH*CELL_SIZE, BOARD_HEIGHT/2*CELL_SIZE))
next_shape_img=pg.image.load('images/game.png')
scaled_next_shape=pg.transform.scale(next_shape_img, (BOARD_WIDTH*CELL_SIZE*0.8,BOARD_HEIGHT*CELL_SIZE))


# Ładowanie zdjec klocków
BLOCK_PATH='images/blocks'

def load_images():
    blocks = [item for item in pathlib.Path(BLOCK_PATH).glob('*.png')]
    images = [pg.image.load(block)for block in blocks]
    images = [pg.transform.scale(block, (CELL_SIZE, CELL_SIZE)) for block in images]
    return images


BLOCK_IMAGES=load_images()