torch.setdefaulttensortype('torch.FloatTensor')

require 'opticalflow'
require 'image'

local im1 = image.scale(image.load('data/000000004.png'), 320, 180, 'bilinear')
local im2 = image.scale(image.load('data/000000005.png'), 320, 180, 'bilinear')
--local im2 = image.rotate(im1, 0.1)

image.display{image=computeOpticalFlowOpenCV(im1, im2, nil, 'farnebach'),legend='farnebach'}
a = torch.Timer()
--image.display{image=computeOpticalFlowOpenCV(im1, im2, nil, 'block'), legend='block'}
print(a:time())

--image.display{image=computeOpticalFlowFastBM(im1, im2, true), legend='fastBM norm'}
flow = computeOpticalFlowFastBM(im1, im2, true)
image.display(flow)
flow:add(-8)
norm = (flow[1]:cmul(flow[1])+flow[2]:cmul(flow[2])):sqrt()
image.display{image=norm}