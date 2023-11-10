import cv2
import numpy as np
import random
import CheckingConnectivity
import os
import shutil

##parameter##
numberOfBlocks = 1
width = 25
height = 25
real_pic_width = 1280
real_pic_height = 720
black_area_width   = (height-real_pic_height/(real_pic_width/height))/2
gridSize = [width,height,3]
totalGames = 1
#points = np.zeros([totalGames,gridSize[0]*gridSize[1]])
points = []

folderName = 'Images_yun_test'

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
    print(black_area_width)
    for i in range(int(black_area_width)):
        for j in range(image.shape[1]):  # Loop through all columns
            image[i][j][0] = 0
            image[i][j][1] = 0
            image[i][j][2] = 0
            flag[j*image.shape[1]+i]=1

        for j in range(image.shape[1]):
            image[image.shape[0] - i - 1][j][0] = 0
            image[image.shape[0] - i - 1][j][1] = 0
            image[image.shape[0] - i - 1][j][2] = 0
            flag[j*image.shape[1]+(image.shape[0] - i - 1)]=1

    return image

def generate_block(blocks, numberOfBlocks, xlimit, limit):
    while(len(blocks)<numberOfBlocks):
        x1 = random.randint(0, limit - 1)
        x, y = divmod(x1, xlimit)
        if(start_pos_y<y<end_pos_y):
            print(y, x)
            blocks.append(x1)
    print(len(blocks))
    print('blocks------', blocks)

def assignBlocks(grid,xlimit,limit,numberOfBlocks,flag, start_pos_y, end_pos_y) :
    blocks = []
    generate_block(blocks, numberOfBlocks, xlimit, limit)
    for pos in blocks:
        # flag[pos] = 1
        x, y = divmod(pos, xlimit)
        for i in range(3):
            grid[y][x][i] = 0 
    print("----------------------------------")
    print(grid[:,:,0])

    # start_pos = start_pos_y+1
    # end_pos = (xlimit-1)*xlimit+end_pos_y
    # print(start_pos, xlimit, end_pos_y, end_pos)
    # while not CheckingConnectivity.Checking(grid[:,:,0], start_pos, end_pos) :
    #     # for pos in blocks :
    #     #     x, y = divmod(pos, xlimit)
    #     #     for j in range(3):
    #     #         grid[y][x][j] = 255
    #     blocks = []
    #     generate_block(blocks, numberOfBlocks, xlimit, limit)
    #     for pos in blocks:
    #         # flag[pos] = 1
    #         x, y = divmod(pos, xlimit)
    #         for i in range(3):
    #             grid[y][x][i] = 0 
    #     print('asdf=====================',grid[:,:,0])

    for pos in blocks:
        flag[pos] = 1

def set_point(flag):
    for i in range(limit) :
        if not flag[i] == 0 :
            #points[t][i] = -100
            points.append(-100)
        elif i== limit-1 :
            points.append(100)
        else :
            points.append(0)

if __name__ == '__main__':
    os.mkdir(folderName)

    for t in range(totalGames) :
        if t%10 == 0 :
            print('%d steps reached'%t)
        a = generate_image(gridSize)            #生成圖片
        # a = set_x_y(a, flag)                          #上下都為黑色 > 將長方形照片變為正方形
        x, y, z = a.shape
        print(x, y, z)

        # cv2.imshow('image',a)
        # cv2.waitKey(0)

        limit = x*y
        print('limit: ', limit)
        flag = np.zeros((limit))
        start_pos_y = int(black_area_width-1)
        end_pos_y = int(width-black_area_width)

        
        a = set_x_y(a, flag) 
        assignBlocks(a,x,limit,numberOfBlocks,flag, start_pos_y, end_pos_y)
        print(flag)

        set_point(flag)
        print(points)
        #end position
        a[end_pos_y][x-1][0] = 255
        a[end_pos_y][x-1][1] = 0
        a[end_pos_y][x-1][2] = 0
        
        print(end_pos_y)
        for i in range(end_pos_y-start_pos_y) :
            for j in range(x):
                a[i+start_pos_y+1][j][0] = 0
                a[i+start_pos_y+1][j][1] = 255
                a[i+start_pos_y+1][j][2] = 0
                cv2.imwrite(os.path.join(folderName,"image_"+str(t)+"_"+str(gridSize[0]*i+j)+".png"),a)
                print(j*x+i+start_pos_y+1, limit)
                if flag[j*x+i+start_pos_y+1] == 0 : 
                    a[i+start_pos_y+1][j][0] = 255
                    a[i+start_pos_y+1][j][1] = 255
                    a[i+start_pos_y+1][j][2] = 255
                else :
                    a[i+start_pos_y+1][j][0] = 0
                    a[i+start_pos_y+1][j][1] = 0
                    a[i+start_pos_y+1][j][2] = 0

                cv2.imshow('image',a)
                cv2.waitKey(0)