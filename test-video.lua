torch.setdefaulttensortype('torch.FloatTensor')

require 'opticalflow'
require 'image'
require 'matching'

local hwin = 14
local wwin = 25
local filters, matcher = opticalFlowFastBM(hwin, wwin)

function loadImg(i)
   return image.rgb2y(image.scale(image.load(string.format("data/%09d.jpg", i)),
				  320, 180, 'bilinear'))
end

local i_img = 1
local im1 = loadImg(i_img)
local im1filtered = filters:forward(im1):clone()
local win, win2
local timer = torch.Timer()

while true do
   i_img = i_img + 1
   local im2 = loadImg(i_img)
   timer:reset()
   local im2filtered = filters:forward(im2)
   print("tcc filters : ", timer:time()['real'])
   local flow = matcher:forward{im1filtered, im2filtered}
   MedianFilter(flow, 5)
   print("toc match   : ", timer:time()['real'])
   flow = flow:real()
   flow[1]:add(-math.floor((hwin-1)/2))
   flow[2]:add(-math.floor((wwin-1)/2))
   local norm = (flow[1]:cmul(flow[1])+flow[2]:cmul(flow[2])):sqrt()
   win = image.display{image=norm, win=win}
   win2 = image.display{image=im2, win=win2}
   im1filtered = im2filtered:clone()
   print("---")
end