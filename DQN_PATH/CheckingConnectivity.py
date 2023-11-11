import numpy as np
real_pic_width = 1280
real_pic_height = 720


def Valid(i,limit,flag,value) :
	if i>=0 and i<=limit-1 and value[i] == 255 and flag[i] == 0 :
		return True 
	return False	

# def set_x_y(image, flag, black_area_width):
#     print(black_area_width)
#     for i in range(int(black_area_width)):
#         for j in range(image.shape[1]):  # Loop through all columns
#             image[i][j] = 0
#             flag[i*image.shape[1]+j]=1

#         for j in range(image.shape[1]):
#             image[image.shape[0] - i - 1][j] = 0
#             flag[(image.shape[0] - i - 1)*image.shape[1]+j]=1

#     return image

def Checking(graph) :
	stack = []
	x,y = graph.shape
	graph1 = np.reshape(graph,-1)
	limit = len(graph1)
	#====
	black_area_width   = int((y-real_pic_height/(real_pic_width/y))/2)
	start = x*black_area_width
	end = limit - start - 1
	#===
	flag = np.zeros([len(graph1)])

	stack.append(start)
	flag[start] = 1
	while (len(stack)>0 and not flag[limit-1] == 1) :
		element = stack.pop()
		#print element
		'''
		if Valid(element-1,limit,flag,graph1) and not element%x == 0 :
			stack.append(element-1)
			flag[element-1] = 1
		'''	
		if Valid(element+1,limit,flag,graph1) and not (element+1)%x == 0:
			stack.append(element+1)
			flag[element+1] = 1
		'''	
		if Valid(element-x,limit,flag,graph1) :
			stack.append(element-x)
			flag[element-x] = 1
		'''	
		if Valid(element+x,limit,flag,graph1) :
			stack.append(element+x)
			flag[element+x] = 1

	if flag[end] == 1 :
		return True
	return False			

						


# a = np.zeros([25,25])
# for i in range(25):
# 	for j in range(25):
# 		a[i][j] = 255

# #====
# print('b', a)
# flag = np.zeros(625)
# black_area_width   = (25-real_pic_height/(real_pic_width/25))/2
# print(black_area_width)
# b = set_x_y(a, flag, black_area_width)
# print('b', b)
# print(flag)
# #===

# a[0][1] = 0
# a[1][1] = 0
# a[1][2] = 0
# print(a)

# if Checking(a) :
# 	print('True')
# else :
# 	print('False')
	