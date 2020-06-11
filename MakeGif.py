import imageio
outfilename = "my.gif" # 转化的GIF图片名称
filenames = []
for i in range(1,500):
    filename = 'animation/'+'capture_'+str(i)+'.jpeg'
    filenames.append(filename)
frames = []
for image_name in filenames:
    im = imageio.imread(image_name)           # 读取方式上存在略微区别，由于是直接读取数据，并不需要后续处理
    frames.append(im)
imageio.mimsave(outfilename, frames, 'GIF', duration=0.05) # 生成方式也差不多
