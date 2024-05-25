import numpy as np
import torch
import random
from game import Game
from network import NeuralNetwork, Memory
import itertools
import matplotlib.pyplot as plt
from game_settings import *
import time
import copy
from Tetris import get_holes, get_bumpiness, get_aggregate_height, get_complete_lines, get_reward, get_list_of_column_size
from datetime import datetime

LIST_OF_ACTIONS = [(0, 0), (0, 1), (0, 2), (0, 3), 
                           (1, 0), (1, 1), (1, 2), (1, 3), 
                           (2, 0), (2, 1), (2, 2), (2, 3), 
                           (3, 0), (3, 1), (3, 2), (3, 3), 
                           (4, 0), (4, 1), (4, 2), (4, 3), 
                           (5, 0), (5, 1), (5, 2), (5, 3), 
                           (6, 0), (6, 1), (6, 2), (6, 3), 
                           (7, 0), (7, 1), (7, 2), (7, 3), 
                           (8, 0), (8, 1), (8, 2), (8, 3), 
                           (9, 0), (9, 1), (9, 2), (9, 3)]
class Agent():

    def __init__(self, game):
        self.epsilon = 1.0
        self.gamma = 0.95
        self.learning_rate = 0.01
        self.EPISODES = 1000
        self.game = game
        self.ACTIONS = {"rotate": 0,
                        "left": 1,
                        "right": 2,
                        "down": 3,
                        "hold": 4}
        self.policy_net = NeuralNetwork(4,1)
        self.target_net = NeuralNetwork(4,1)
        self.memory = Memory(self.policy_net, self.target_net, self.gamma,self.learning_rate)

    def make_action2(self,action_number):
        
        
        position, rotations = LIST_OF_ACTIONS[action_number]
        start_position = int(INITIALIZE_POSITION.x)
        
        #print(rotations)
        for i in range(rotations):
            self.game.tetris.tetromino.rotate()
        
        direction = position - start_position

        for i in range(abs(direction)):
            if direction<start_position:
                self.game.tetris.tetromino.move(direction="left")
            elif direction> start_position:
                self.game.tetris.tetromino.move(direction="right")
            else:
                pass
        
        self.game.tetris.tetromino.move_down()


    def make_action(self,action_number):
        if action_number == 0:
            self.game.tetris.tetromino.rotate()
        elif action_number == 1:    
            self.game.tetris.tetromino.move(direction="left")
        elif action_number == 2:
            self.game.tetris.tetromino.move(direction="right")
        elif action_number == 3:
            self.game.tetris.tetromino.move_down()
            
        elif action_number == 4:
            pass

    def get_state2(self, board):
        aggr_height = get_aggregate_height(board)
        complete_lines =get_complete_lines(board)
        holes = get_holes(board)
        bumpiness = get_bumpiness(board)
        

        state =[float(aggr_height),
                float(complete_lines),
                float(holes),
                float(bumpiness)]
        #print(state)
        return np.array(state, dtype=np.float32)
    
    def get_state(self):
        #blocks_position = tf.keras.layers.Flatten(self.game.tetris.list_of_tetrominos)
        #block_position1d = [item for sublist in self.game.tetris.list_of_tetrominos for item in sublist]
        block_position1d = [1 if item != 0 else 0 for sublist in self.game.tetris.list_of_tetrominos for item in sublist]
        


        listOfBlocks = {}
        for tetrominoX in self.game.tetris.list_of_tetrominos:
            for singleTetromino in tetrominoX:
                if singleTetromino != 0:
                    tempX = singleTetromino.position.x
                    tempY = singleTetromino.position.y
                    if(tempX not in listOfBlocks):
                        listOfBlocks[tempX] = []
                    listOfBlocks[tempX] += [tempY]

        dictionaryOfColHeight = {}
        listOfMinColHeight = []
        for key,value in listOfBlocks.items():
            dictionaryOfColHeight[key] = min(value)

        for temp in range(0,10):
            if temp in dictionaryOfColHeight:
                listOfMinColHeight.append(dictionaryOfColHeight[float(temp)])
            else:
                listOfMinColHeight.append(20.0)
        


        state = [
            # position of the tetromino blocks
            self.game.tetris.tetromino.blocks[0].position.x,
            self.game.tetris.tetromino.blocks[1].position.x,
            self.game.tetris.tetromino.blocks[2].position.x,
            self.game.tetris.tetromino.blocks[3].position.x,
            self.game.tetris.tetromino.blocks[0].position.y,
            self.game.tetris.tetromino.blocks[1].position.y,
            self.game.tetris.tetromino.blocks[2].position.y,
            self.game.tetris.tetromino.blocks[3].position.y,
            # game board
            #blocks_position
            # game speed
            #speed = 100



        ]
        state = state + listOfMinColHeight
        return np.array(state, dtype=np.float64)
    
    def act(self, actionNumber):
        if random.random() <= self.epsilon:
            print("random")
            return random.randint(0,39)  
        return actionNumber

    def simulate_move(self, action_number, board, tetrominoBlocks):
        position, rotation = LIST_OF_ACTIONS[action_number]

        # for blocks in tetrominoBlocks:
        #     print(blocks.x,blocks.y )

        for temp in range(rotation):
            self.rotate(board,tetrominoBlocks)
        
        self.move(position,tetrominoBlocks,board)
        new_position=[(block.x,block.y + 1) for block in tetrominoBlocks]
        collision = self.checkCollisions(board,new_position)
        while not collision:
            for block in tetrominoBlocks:
                block.y +=1
            new_position=[(block.x,block.y+1) for block in tetrominoBlocks]
            collision = self.checkCollisions(board,new_position)

        for block in tetrominoBlocks:
            board[int(block.y)][int(block.x)] ="1"

        return board


    def checkCollisions(self, board, tetrominoBlocks):
        for block in tetrominoBlocks:
            x,y=int(block[0]), int(block[1])
            if not(0<=x<BOARD_WIDTH and y <BOARD_HEIGHT and (y<0 or not board[y][x])):
                return True
        return False

    def rotate(self, board, tetrominoBlocks):
        rotate_point=tetrominoBlocks[0]
        new_position=[self.rotateBlock(rotate_point,block) for block in tetrominoBlocks]

        if not self.checkCollisions(board,new_position):
            for i, block in enumerate(tetrominoBlocks):
                block.x=new_position[i].x
                block.y=new_position[i].y

    def rotateBlock(self, rotate_point, block):
        point=block-rotate_point
        rotated=point.rotate(90)
        return rotated+rotate_point

    def move(self, position, tetrominoBlocks, board):
        start_position = 4
        direction = position - start_position
        
        sign = -1
        if(direction>0):
            sign = 1

        new_value =[(block.x+sign, block.y) for block in tetrominoBlocks]
        for i in range(abs(direction)):
            new_value =[(block.x+sign, block.y) for block in tetrominoBlocks]
            if not(self.checkCollisions(board,new_value )):
                if direction<start_position:
                    for block in tetrominoBlocks:
                        block.x -=1
                elif direction> start_position:
                    for block in tetrominoBlocks:
                        block.x +=1
                else:
                    pass

    def simulate_aciton(self):
        best_score = -5000
        best_action = None
        
        listOfBlocks= []
        for block in self.game.tetris.tetromino.blocks:
            copiedBlock= copy.deepcopy(block.position)
            listOfBlocks.append(copiedBlock)

        copied_2d_list = [row[::] for row in self.game.tetris.list_of_tetrominos] #copy.deepcopy(self.game.tetris.list_of_tetrominos)#[row[::-1] for row in game.tetris.list_of_tetrominos]

        for action_number in range(len(LIST_OF_ACTIONS)):
            simulated_list = [row[::] for row in copied_2d_list]
            copiedListOfBlocks = copy.deepcopy(listOfBlocks)
            simulated_list = self.simulate_move(action_number, simulated_list, copiedListOfBlocks)
            new_state = self.get_state2(simulated_list)
            scoreTensor = torch.tensor(new_state, dtype=torch.float32)
            score = self.policy_net(scoreTensor)
            value = score.item()
            if value > best_score:
                best_score = value
                best_action = action_number
    
        return best_action



MIN_MEMORY_SIZE = 98
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10_000

UPDATE_FREQ = 100

WEIGHTS = [-0.510066, 0.760666, -0.35663, -0.184483]
def train():
       
        
        game = Game()
        agent = Agent(game)
        plt_scores = []
        plt_loss = []
        plt_average_score = []
        games = 0
        total_score = 0
        save_model = 0
        # for _ in range(100):

        #     game.run()
        #     action = agent.simulate_aciton()
        #     #epsilonAction = agent.act(action)
        #     state = agent.get_state2(game.tetris.list_of_tetrominos)
        #     agent.make_action2(action)
            
        #     #print(state)
        #     #print(state)
        #     #action = agent.act(state=state)
            

        #     #reward, done, score = game.tetris.check_for_reward()
        #     reward, done, score = game.tetris.check_for_reward()
        #     #print(game.tetris.get_aggregate_height())
        #     new_state = agent.get_state2(game.tetris.list_of_tetrominos)
        #     #print(new_state)
        #     #reward = get_reward(WEIGHTS, new_state)
        #     agent.memory.update_memory(state,action,reward,new_state,done)
           
        #     if done:
        #         game.restart()
        #         print("Epoch: ", _)
        
        # game.restart()
        agent.memory.load_checkpoint("savedNN/model_2024-05-24 22-33-23")
        for step in itertools.count():
            if len(agent.memory.memory) < MIN_MEMORY_SIZE:
                return
            game.run()
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            agent.epsilon = epsilon

            state = agent.get_state2(game.tetris.list_of_tetrominos)
            
            action = agent.simulate_aciton()
            #epsilonAction = agent.act(action)
            agent.make_action2(action)
           
            #reward, done, score = game.tetris.check_for_reward()
            
            reward, done, score = game.tetris.check_for_reward()
            
            
            
            new_state = agent.get_state2(game.tetris.list_of_tetrominos)
            #reward = get_reward(WEIGHTS, new_state)
            
            if score > game.tetris.highscore:
                reward += 1000
            if done:
                reward = -100
            # new_state = agent.get_state()
            print("Reward: ", reward)
        
            agent.memory.update_memory(state,action,reward,new_state,done)
            
            sample = agent.memory.sample()
            agent.memory.train_from_memory(sample)

            if step % UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            if done:
                print("Game Number:", games)
                game.restart()
                games +=1
                save_model +=1
                if score > game.tetris.highscore:
                    game.tetris.highscore = score
                total_score +=score
                plt_scores.append(score)
                plt_average_score.append(total_score/games)
                plt_loss.append(agent.memory.loss)

            if save_model % 1000 == 0 and save_model !=0:
                now = datetime.now()
                date_time_str = now.strftime("%Y-%m-%d %H-%M-%S")
                agent.memory.save_checkpoint(f"savedNN/model_{save_model}")
                save_model +=1
            
            if games >= 22000:
                make_plot(plt_scores,games,'Score')
                make_plot(plt_average_score,games,'Average Score')
                make_plot(plt_loss,games,'Loss')
                break


def make_plot(array,games,x,y='Games'):
    plt.figure(figsize=(10, 8))  # Ustawienie większego rozmiaru wykresu
    plt.plot([i for i in range(games)], array)  # Dodanie punktów i wykresu liniowego
    plt.title(f'{x} plot')  # Tytuł wykresu
    plt.xlabel(y)  # Etykieta osi X
    plt.ylabel(x)  # Etykieta osi Y
    plt.grid(True)  # Dodanie siatki
    plt.show()  # Wyświetlenie wykresu







if __name__=="__main__":
    game = Game()
    agent = Agent(game)
    train()
        # game.run()
        # print(get_list_of_column_size(game.tetris.list_of_tetrominos))
    # while True:
    #     game.run()

    #     copiedBlocks= []
    #     for block in game.tetris.tetromino.blocks:
    #         copiedBlock= copy.deepcopy(block.position)
    #         copiedBlocks.append(copiedBlock)
    #     copied_2d_list = [row[::-1] for row in game.tetris.list_of_tetrominos]
    #     for temp in range(40):
    #         agent.simulate_move(temp,copied_2d_list,copiedBlocks)

        #agent.simulate_move(6,game.tetris.list_of_tetrominos,game.tetris.tetromino.blocks)
        #game.run()
# while True:
#     game.run()
#     #agent.make_action2(6)
#     #pg.time.wait(5000)
#     #break
    
            