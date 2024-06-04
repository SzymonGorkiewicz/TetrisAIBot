import time
from game_settings import *
from Tetris import Tetris



class Game:
    def __init__(self):
        pg.init()
        pg.display.set_caption("Tetris")
        self.window=pg.display.set_mode(BOARD_RESOLUTION)
        self.clock=pg.time.Clock()
        self.tetris=Tetris(self)
        self.user_event = pg.USEREVENT + 0
        self.fast_user_event = pg.USEREVENT + 1
        self.animation = False
        self.fast_animation=False
        self.FAST_ANIM_TIME = 1
        self.timer(self.tetris.animation_time)
        

    def clear_board(self):
        self.tetris.clear_board()

    def timer(self, animation_time):
        pg.time.set_timer(self.user_event, animation_time)
        pg.time.set_timer(self.fast_user_event, self.FAST_ANIM_TIME)

    def draw(self):
        self.window.fill(color=BG_COLOR)
        self.window.fill(color=BOARD_COLOR, rect=(0, 0, *BOARD_RESOLUTION2))
        self.window.blit(scaled_next_shape, (400, 0))
        self.tetris.draw()
        self.window.blit(scaled_logo, (BOARD_WIDTH*CELL_SIZE*0.9, -100))
        self.tetris.printing_score()


    def update(self):
        self.tetris.update()
        self.clock.tick(FPS)


    def restart(self):
        self.tetris.__init__(self)

    def check_events(self):
        self.animation=False
        self.fast_animation=False
        for event in pg.event.get():
            if event.type==pg.QUIT:
                pg.quit()
            elif event.type==pg.KEYDOWN:
                self.tetris.controls(event.key)
                if event.key==pg.K_ESCAPE:
                    pg.quit()
            elif event.type==self.user_event:
                self.animation=True
            elif event.type==self.fast_user_event:
                self.fast_animation=True

    def run(self):
        self.check_events()
        self.draw()
        self.update()
     









                

