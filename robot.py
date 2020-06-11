import pygame
import sys
import random
import math
import numpy as np
from ga import *
vec = pygame.math.Vector2
from math import *

# TODO
# 旋转方向与实际情况相反？
# 视野角度仅在0～180度间适用

pygame.init()

SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
BACKGROUND_COLOR = (0,0,0)

screen = pygame.display.set_mode(SIZE)
robots = []
lights = []

# 机器人配置
VMAX = 15           # 最大速度
D = 1               # 轮子直径     
ANGLE = 0          # 水平眼睛抬起角度 
VISION_ANGLE = 2*(90-ANGLE) # 视野角度  
mp4 = False         # 是否截图
RUNNING_TIME = 500 # 运行时间

# 颜色
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARKGRAY = (40, 40, 40)

# for bp algorithm
angles = []         # 两只眼睛与光源夹角
velocities = []     # 速度

# ANN 结构
input_size = 4
hidden_size = 20
output_size = 2
std = 1e-2

# rand函数根据给定维度生成[0,1)之间的数据，shape(input_size, hidden_size)
W1 = std * np.random.randn(input_size, hidden_size)   #(2,10)
b1 = np.zeros((1, hidden_size))                       #(1,10)
W2 = std * np.random.randn(hidden_size,output_size)   #(10,1)
b2 = np.zeros((1, output_size))                       #(1,2)
params = {}
params["W1"] = W1
params["b1"] = b1
params["W2"] = W2
params["b2"] = b2

IDX = 99
filename = "model2/genetic0"
loaded_ga_instance = load(filename=filename)
loaded_ga_instance.best_dna()
ann_params = loaded_ga_instance.best_dnas[IDX]

# ann_params = [0.29206694,-1.12764626,-0.25338742,0.0093111,1.1474796,-1.14158496
# ,-0.11047037,1.78318212,-1.4273091,-0.3511147,0.41424884,0.22776319
# ,1.22959633,-0.65407822,1.388445,-0.08088593,1.51906805,-0.74863552
# ,-0.82540428,0.17280692,-1.90248669,-1.18705694,-0.4395787,3.27884332
# ,-1.04985322,0.81555624,-1.14116555,-1.02818322,-2.55422391,-1.14891684
# ,1.693369,-2.93727087,0.34217733,-1.11540733,-0.78988005,-1.11390283
# ,-1.41857688,0.53719164,0.59365983,1.2693424,-0.42483805,0.00600158
# ,1.77860656,0.46959559,-0.09628464,-0.4467237,1.23855698,0.68736386
# ,0.43556836,1.36378201,-0.12235901,-0.03596828,0.72665431,-0.68072703
# ,-0.84562548,2.14618309,-0.44587769,-0.51216261,-0.6204513,-1.5243027
# ,-0.9382298,-1.50967747,-1.1793049,-1.21229243,0.63961258,-0.67151312
# ,-1.43373659,-0.623599,0.71080416,-1.49853935,-0.70069586,-0.83718268
# ,-0.03507381,0.72105619,-0.98810112,-0.8265761,-0.4712849,-0.1487148
# ,0.25708944,0.12968569,-1.38768253,-0.51082379,-0.61426624,-0.86504677
# ,0.25724603,-0.52528635,-1.99293083,-0.63521342,-0.35146199,0.21463902
# ,-0.38939612,0.02018344,-0.59229256,-0.03919557,-0.89061698,-0.49900935
# ,-1.86247203,-2.32737008,-0.76925637,-0.34648897,0.15579547,-0.54310937]

def relu(x):    
    return np.maximum(0, x)


# ANN output
def predict(X,params):
    z1 = np.dot(X,params["W1"]) + params["b1"]              
    a1 = relu(z1)                 
    z2 = np.dot(a1,params["W2"]) + params["b2"]
    exp_z2 = np.exp(z2)
    a2 = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True) 
    return a2

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.zeros(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(relu(z2), 0, values=np.zeros(m), axis=1)
    z3 = np.array(a2 * theta2.T) 
    z3 -= z3.max(axis=-1, keepdims=True)  # 防止溢出
    exp_z3 = np.array(np.exp(z3))
    h = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)
    return h*VMAX



# ann_params = (np.random.random(size=hidden_size * (input_size + 1) + output_size * (hidden_size + 1)) - 0.5) * 0.25
theta1 = np.matrix(np.reshape(ann_params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(ann_params[hidden_size * (input_size + 1):], (output_size, (hidden_size + 1))))

# 随机初始化方向
def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    r = np.sqrt((x*x).sum())
    return x / r

# 加载光源图片
class Light(object):
    def __init__(self,pos,img_src,id):
        self.image_light = pygame.image.load(img_src)
        # rect_light = image_light.get_rect()
        self.pos = np.array(pos)
        self.id = id
    def draw(self):
        screen.blit(self.image_light,self.pos)


# 绘制光源图像
# class Light:
#     def __init__(self,pos,rc,rr,id):
#         self.color = rc
#         self.radius = rr
#         self.pos = pos
#         self.id = id

#     def draw(self):
#         pygame.draw.circle(screen,self.color,self.pos,self.radius)


# 追光机器人
class Robot(object):
    # 机器人属性
    def __init__(self, pos=[10.0, 10.0], velocity=[0, 0]):
        self.imageSrc = pygame.image.load("open.png")
        self.rect = self.imageSrc.get_rect()
        self.image = self.imageSrc
        self.velocity = vec(velocity)
        self.vl = 0         # 初始化左轮速度
        self.vr = 0         # 初始化右轮速度
        self.angle = 0      # 初始化角度 竖直向上为0度
        self.pos = np.array(pos)
        self.rect = self.rect.move(pos[0], pos[1])  # 初始化位置
        self.step = 1
    # 当前运动方向(单位向量)
    def direction(self):
        vel = np.linalg.norm(self.velocity)
        return self.velocity/vel

    # 视野
    def vision(self,light):
        # 中心坐标
        self.pos = self.rect.center
        # 左眼坐标
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        # 右眼坐标
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
        # 中心到左/右眼方向(单位向量) 
        l =  np.array([x1,y1]-np.array(self.pos))
        r =  np.array([x2,y2]-np.array(self.pos))
        norm_l = l/np.linalg.norm(l)
        norm_r = r/np.linalg.norm(r)
        # 中心到光源方向(单位向量) 
        arr_light = np.array(np.array(light.pos)-np.array(self.pos))
        norm_light = arr_light/np.linalg.norm(arr_light)
        # 分别求出  中心到光源方向 与 中心到左/右眼方向 向量夹角(角度制) arccos(a*b/(|a|*|b|))
        l_n_light = degrees(np.arccos( np.dot( norm_l,norm_light)/( np.linalg.norm(norm_l)*np.linalg.norm(norm_light)) ) )
        r_n_light = degrees(np.arccos(np.dot(norm_r,norm_light)/(np.linalg.norm(norm_r)*np.linalg.norm(norm_light))))
        # 如果向量间的夹角均小于视野角度 说明在视野范围内
        l_and_r = l_n_light + r_n_light
        if l_and_r>=VISION_ANGLE-5 and l_and_r<=VISION_ANGLE+5:
            return True
        else:
            return False
        # return (l_n_light <= VISION_ANGLE and r_n_light <= VISION_ANGLE)

    def move_by_ann(self,lights):
         # 左/右眼坐标
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
       
        self.pos = self.rect.center
        arr_light_one = np.array(np.array(lights[0].pos)-np.array(self.pos))
        arr_light_two = np.array(np.array(lights[1].pos)-np.array(self.pos))

        # 机器人中心到光源距离
        distance_two = np.linalg.norm(arr_light_two)
        distance_one = np.linalg.norm(arr_light_one)
        # 在视野范围内 未到达光源
        if((self.vision(lights[0]) or self.vision(lights[1])) and (distance_one >= 35 and distance_two >= 35)):
            arrSensor = np.array([x2-x1,y2-y1])
            arr1_one =  np.mat([lights[0].pos[0] - x1,lights[0].pos[1] - y1])
            arr2_one =  np.mat([lights[0].pos[0] - x2,lights[0].pos[1] - y2])
            # print(arrSensor.shape,arr1.shape,arr2.shape)
            sl_one = float(arrSensor*arr1_one.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr1_one))
            sr_one = float(arrSensor*arr2_one.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr2_one))

            arr1_two =  np.mat([lights[1].pos[0] - x1,lights[1].pos[1] - y1])
            arr2_two =  np.mat([lights[1].pos[0] - x2,lights[1].pos[1] - y2])
            # print(arrSensor.shape,arr1.shape,arr2.shape)
            sl_two = float(arrSensor*arr1_two.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr1_two))
            sr_two = float(arrSensor*arr2_two.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr2_two))
            # angle_l = degrees(acos(sl))
            # angle_r = degrees(acos(sr))
            # angles.append(str(sl)+" "+str(sr))
            # angles.append(str(angle_l)+" "+str(angle_r))

            # based on rules
            # self.vl = (VMAX * 0.5 * (1-sl))
            # self.vr = (VMAX * 0.5 * (1+sr))
            # print(np.matrix([sl,sr]))
            # Move by ann
            # if light.id == 0:
            ann_vr,ann_vl = np.array(forward_propagate(np.matrix([sl_one,sr_one,sl_two,sr_two]),theta1,theta2))[0]
            self.vr = ann_vr
            self.vl = ann_vl
            print(self.vl,self.vr)
            self.angle -= (self.vr - self.vl)/D
            tmp = (self.vr - self.vl)/D
            dir = vec(self.direction()[0],self.direction()[1]).rotate(tmp)
            self.velocity = (self.vl + self.vr)/2 * dir
            self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)
                
        elif ((self.vision(lights[0]) or self.vision(lights[1])) and (distance_one < 35 or distance_two < 35)):
            print("---------")
            print(distance_one,distance_two)
            x = random.randrange(30, WINDOW_WIDTH - 30)
            y = random.randrange(30, WINDOW_HEIGHT - 30)
            light_pos = np.array([x,y])
            if distance_two<35:
                lights[1].pos = light_pos
                x = random.randrange(30, WINDOW_WIDTH - 30)
                y = random.randrange(30, WINDOW_HEIGHT - 30)
                light_pos = np.array([x,y])
                lights[0].pos = light_pos
            else:
                lights[0].pos = light_pos
        # 不在视野范围内 
        else:
            self.vl = 0
            self.vr = VMAX
            
            self.angle -= (self.vr - self.vl)/D
            # self.angle -= 1
            ag = (self.vr - self.vl)/D
            dir = vec(self.direction()[0],self.direction()[1]).rotate(ag)
            # print("out of vision!")
    
            self.velocity = (self.vl + self.vr)/2 * dir
            # print("", dir)

            self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)


    def move(self,lights):

        # 左/右眼坐标
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
       
        
        # 对于多个光源
        for light in lights: 
            self.pos = self.rect.center
            arr_light = np.array(np.array(light.pos)-np.array(self.pos))
            # 机器人中心到光源距离
            distance = np.linalg.norm(arr_light)
            # 在视野范围内 未到达光源
            if(self.vision(light) and distance >= 35):
                arrSensor = np.array([x2-x1,y2-y1])
                arr1 =  np.mat([light.pos[0] - x1,light.pos[1] - y1])
                arr2 =  np.mat([light.pos[0] - x2,light.pos[1] - y2])
                # print(arrSensor.shape,arr1.shape,arr2.shape)
                sl = float(arrSensor*arr1.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr1))
                sr = float(arrSensor*arr2.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr2))
                # angle_l = degrees(acos(sl))
                # angle_r = degrees(acos(sr))
                # angles.append(str(sl)+" "+str(sr))
                # angles.append(str(angle_l)+" "+str(angle_r))

                # based on rules
                # self.vl = (VMAX * 0.5 * (1-sl))
                # self.vr = (VMAX * 0.5 * (1+sr))
                # print(np.matrix([sl,sr]))
                # Move by ann
                # if light.id == 0:
                ann_vr,ann_vl = np.array(forward_propagate(np.matrix([sl,sr,light.id]),theta1,theta2))[0]
                self.vr = ann_vr
                self.vl = ann_vl
                print(self.vl,self.vr)
                self.angle -= (self.vr - self.vl)/D
                tmp = (self.vr - self.vl)/D
                dir = vec(self.direction()[0],self.direction()[1]).rotate(tmp)
                self.velocity = (self.vl + self.vr)/2 * dir
                self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)
                    
            elif (self.vision(light) and distance < 35):
                x = random.randrange(30, WINDOW_WIDTH - 30)
                y = random.randrange(30, WINDOW_HEIGHT - 30)
                light_pos = np.array([x,y])
                light.pos = light_pos
            # 不在视野范围内 
            else:
                self.vl = 0
                self.vr = VMAX
                
                self.angle -= (self.vr - self.vl)/D
                # self.angle -= 1
                ag = (self.vr - self.vl)/D
                dir = vec(self.direction()[0],self.direction()[1]).rotate(ag)
                # print("out of vision!")
        
                self.velocity = (self.vl + self.vr)/2 * dir
                # print("", dir)

                self.rect = self.rect.move(self.velocity[0]*self.step, self.velocity[1]*self.step)

        # velocities.append(str(self.velocity[0])+" "+str(self.velocity[1]))

    def draw(self):
        screen.blit(self.image, self.rect)

    # 机器人中心到眼睛向量
    def draw_vectors(self):
        self.pos = self.rect.center
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2

        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2
        scale = 100
        l =  np.array([x1,y1]-np.array(self.pos))
        r =  np.array([x2,y2]-np.array(self.pos))
        norm_l = l/np.linalg.norm(l)
        norm_r = r/np.linalg.norm(r)
        # left
        pygame.draw.line(screen, BLACK, self.pos, [x1,y1] + norm_l*scale, 5)
        # right
        pygame.draw.line(screen, RED, self.pos, [x2,y2] + norm_r*scale, 5)
      
    # 小车旋转
    def rotate(self):
        self.image = pygame.transform.rotate(self.imageSrc, self.angle)
        if math.fabs(self.angle) >= 360:
            self.angle -= 360
        
        self.rect = self.image.get_rect(center = self.rect.center)  # 中心矫正


def init():
    for i in range(0, 1):
        x = random.randrange(50, WINDOW_WIDTH - 50)
        y = random.randrange(50, WINDOW_HEIGHT - 50)
        robot_pos = np.array([x,y])
        # y_speed = random.randrange(-10, -5)
        y_speed = -VMAX
        robot_velocity = vec([0,y_speed])
        robot = Robot(robot_pos,robot_velocity)
        robots.append(robot)

    for j in range(0,2):
        x = random.randrange(30, WINDOW_WIDTH - 30)
        y = random.randrange(30, WINDOW_HEIGHT - 30)
        light_pos = np.array([x,y])
        # light = Light(light_pos) 
        if j == 0:
            light = Light(light_pos,"yao.png",j)
        else:
            light = Light(light_pos,"light.png",j) 
        lights.append(light)


 # Save file
def save_file():
    with open("input.txt","a+",encoding="utf-8") as f:
        for i in angles:
            f.writelines(i+'\n')
  
    with open("output.txt","a+",encoding="utf-8") as f2:
        for i in velocities:
            f2.writelines(i+'\n')
    f.close()
    f2.close()

clock = pygame.time.Clock()
init()
# light = Light([300,100],RED,10) 
index = 0
done = False
while not done:

    index+=1
    if mp4 == True:
        filename = 'animation/'+'capture_'+str(index)+'.jpeg'
        pygame.image.save(screen, filename)

    if index >= RUNNING_TIME:  # 控制运行时间
        done = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    x = random.randrange(30, WINDOW_WIDTH - 30)
                    y = random.randrange(30, WINDOW_HEIGHT - 30)
                    light_pos = np.array([x,y])
                    light = Light(light_pos,RED,10)
                    # light = Light(light_pos) 
              
    screen.fill(BACKGROUND_COLOR)

    # while tracking
    # light = Light(pygame.mouse.get_pos(),RED,20)
    # lights = [light]
    for light in lights:
        light.draw()
    # 将旋转后的图象，渲染到新矩形里
    for item in robots:
        item.rotate()
        # item.move(lights)
        item.move_by_ann(lights)
        item.draw()
        # item.draw_vectors()
    
    # myfont = pygame.font.SysFont("arial",20)
    # text_dist = myfont.render("Dist = "+str(distance), True, (0,255,0))
    # screen.blit(text_dist, (WINDOW_WIDTH-150, 0))
    pygame.display.update()
    # 控制帧数<=100
    clock.tick(60)

# save_file()