import cv2
import numpy as np
import random
import CheckingConnectivity
import os
import shutil

##parameter##
numberOfBlocks = 20
width = 25
height = 25
real_pic_width = 1280
real_pic_height = 720
black_area_width   = (height-real_pic_height/(real_pic_width/height))/2
black_area  = int(int((height-720/(1280/height)))*height/2)
gridSize = [width,height,3]
totalGames = 20
start_pos_x = int(black_area_width-1)
end_pos_x = int(width-black_area_width)
#points = np.zeros([totalGames,gridSize[0]*gridSize[1]])
points = []

folderName = 'testImages_box'

if os.path.exists(folderName):
	shutil.rmtree(folderName)

def generate_image(gridSize):
    image = np.zeros(gridSize)
    x, y, z = image.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                image[i][j][k] = 255

    return image 

def set_x_y(image, flag):
    for i in range(int(black_area_width)):
        for j in range(image.shape[1]):  # Loop through all columns
            image[i][j][0] = 0
            image[i][j][1] = 0
            image[i][j][2] = 0
            flag[i*image.shape[1]+j]=1

        for j in range(image.shape[1]):
            image[image.shape[0] - i - 1][j][0] = 0
            image[image.shape[0] - i - 1][j][1] = 0
            image[image.shape[0] - i - 1][j][2] = 0
            flag[(image.shape[0] - i - 1)*image.shape[1]+j]=1

    return image

def generate_block(blocks, numberOfBlocks, xlimit, limit):
    box = 177 - 1 #[6, 1]
    h = 2
    w = 5
    total = 9
    while(total):
        for xa in range(h):
            for ya in range(w):
                box+=1
                blocks.append(box)
                print(box)
            box+=(-5+25)
        box-=(-5+25)        
        total-=1
        if(total%3 != 0):
            box-=22
        else:
            box+=54

import random

def generate_random_blocks():
    blocks = []
    
    # 設定方塊數量上限為9
    numberOfBlocks = random.randint(5, 9)
    box = [] #一個方塊的box_id
    boxes = [] #存取所有box的id值
    while numberOfBlocks:
        h = random.randint(1, 5)
        w = random.randint(1, 5)
        
        box_end = end_pos_x*height - (h-1)*height
        box_id = random.randint(start_pos_x*height, box_end)
        # while(box_id in boxes):
        #     h = random.randint(1, 3)
        #     w = random.randint(1, 3)
        #     box_end = end_pos_x*height - (h-1)*height
        #     box_id = random.randint(start_pos_x*height, box_end)
        #     boxes = []
        #     box=[]
        #     blocks = []
        #     numberOfBlocks=9
        for i in range(h):
            for j in range(w):
                blocks.append(box_id)
                box.append(box_id)
                box_id+=1
            box_id+=(-w+height)
        boxes.append(box)
        box = []
        numberOfBlocks -= 1

    return blocks



    # while(len(blocks)<numberOfBlocks):
    #     x1 = random.randint(0, limit - 1)
    #     x, y = divmod(x1, xlimit)
    #     if(start_pos_x<x<end_pos_x):
    #         blocks.append(x1)

def assignBlocks(grid,xlimit,limit,numberOfBlocks,flag) :
    blocks = []
    # generate_block(blocks, numberOfBlocks, xlimit, limit)
    blocks = generate_random_blocks()
    for pos in blocks:
        # flag[pos] = 1
        x, y = divmod(pos, xlimit)
        for i in range(3):
            grid[x][y][i] = 0 

    # cv2.imshow('a', grid)
    # cv2.waitKey(0)
    while not CheckingConnectivity.Checking(grid[:,:,0]) :
        blocks = []
        # generate_block(blocks, numberOfBlocks, xlimit, limit)
        blocks = generate_random_blocks()
        for pos in blocks:
            # flag[pos] = 1
            x, y = divmod(pos, xlimit)
            for i in range(3):
                grid[x][y][i] = 0 

    for pos in blocks:
        flag[pos] = 1

def set_point(flag, limit):
    print(black_area)
    # breakpoint()
    for i in range(limit) :
        if not flag[i] == 0 :
            #points[t][i] = -100
            points.append(-100)
        elif i== limit-1-black_area :
            points.append(100)
        else :
            points.append(0)

if __name__ == '__main__':
    os.mkdir(folderName)

    for t in range(totalGames) :
        if t%10 == 0 :
            print('%d steps reached'%t)
        a = generate_image(gridSize)            #生成圖片
        x, y, z = a.shape
        limit = x*y
        flag = np.zeros((limit))
        a = set_x_y(a, flag) 
        assignBlocks(a,x,limit,numberOfBlocks,flag)

        set_point(flag, limit)
        # print('a:\n', a[:,:,0])
        # print('flag:\n', flag)
        # print('point:\n', points)
        # breakpoint()
        #end position
        a[end_pos_x][x-1][0] = 255
        a[end_pos_x][x-1][1] = 0
        a[end_pos_x][x-1][2] = 0
        
        for i in range(end_pos_x-start_pos_x) :
            for j in range(x):
                a[i+start_pos_x+1][j][0] = 0
                a[i+start_pos_x+1][j][1] = 255
                a[i+start_pos_x+1][j][2] = 0
                cv2.imwrite(os.path.join(folderName,"image_"+str(t)+"_"+str(gridSize[0]*i+j)+".png"),a)
                if flag[(i+start_pos_x+1)*x+j] == 0 : 
                    a[i+start_pos_x+1][j][0] = 255
                    a[i+start_pos_x+1][j][1] = 255
                    a[i+start_pos_x+1][j][2] = 255
                else :
                    a[i+start_pos_x+1][j][0] = 0
                    a[i+start_pos_x+1][j][1] = 0
                    a[i+start_pos_x+1][j][2] = 0

np.savetxt('testPoints_box.txt',points)