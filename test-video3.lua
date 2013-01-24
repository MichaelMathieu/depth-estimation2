torch.setdefaulttensortype('torch.FloatTensor')


require 'camera'
require 'opticalflow'
require 'image'
require 'matching'
require 'prettydisplay'
require 'filtering'
require 'planning'
require 'ardrone'
require 'io'
require 'xlua'


local hwin = 10
local wwin = 16

local filters0 = torch.LongTensor{
   {0, 4, 8, 8, 12, 0, 16, 8},
   {0, 8, 8, 12, 12, 4, 16, 12},
   {8, 0, 12, 12, 4, 4, 16, 8},
   {0, 0, 4, 8, 8, 0, 16, 4},
   {8, 4, 12, 16, 4, 8, 16, 12},
   {4, 0, 8, 12, 0, 4, 12, 8},
   {0, 12, 8, 16, 12, 8, 16, 16},
   {4, 4, 8, 16, 0, 8, 12, 12},
   {0, 4, 8, 12, 0, 0, 4, 16},
   {0, 4, 4, 12, 8, 4, 16, 8},
   {4, 0, 12, 16, 0, 4, 16, 12},
   {4, 0, 8, 4, 0, 0, 4, 4},
   {4, 4, 12, 12, 4, 0, 8, 16},
   {4, 4, 8, 8, 0, 4, 4, 8},
   {0, 8, 4, 16, 8, 8, 16, 12},
   {8, 0, 12, 4, 4, 0, 8, 4},
   {4, 8, 8, 12, 0, 8, 4, 12},
   {4, 12, 8, 16, 0, 12, 4, 16},
   {8, 4, 12, 8, 4, 4, 8, 8},
   {8, 4, 16, 12, 8, 0, 12, 16},
   {8, 8, 12, 12, 4, 8, 8, 12},
   {8, 12, 12, 16, 4, 12, 8, 16},
   {12, 0, 16, 8, 0, 4, 8, 8},
   {0, 4, 12, 8, 0, 0, 4, 12},
   {12, 0, 16, 4, 8, 0, 12, 4},
   {12, 4, 16, 12, 0, 8, 8, 12},
   {4, 0, 12, 4, 0, 0, 8, 4},
   {0, 8, 12, 12, 0, 4, 4, 16},
   {12, 4, 16, 8, 8, 4, 12, 8},
   {12, 8, 16, 12, 8, 8, 12, 12},
   {12, 12, 16, 16, 8, 12, 12, 16},
   {4, 4, 12, 8, 0, 4, 8, 8},
}


--[[
local filters0= torch.LongTensor{{ 0, 0, 8, 8,   8, 8,16,16},
				 { 8, 0,16, 8,   0, 8, 8,16},
				 { 0, 0, 8,16,   8, 0,16,16},
				 { 0, 0,16, 8,   0, 8,16,16},

				 { 4, 4, 8, 8,   8, 8,12,12},
				 { 8, 4,12, 8,   4, 8, 8,12},
				 { 4, 0, 8,16,   8, 0,12,16},
				 { 0, 4,16, 8,   0, 8,16,12},

				 { 0, 0, 4,16,   4, 0, 8,16},
				 { 8, 0,12,16,  12, 0,16,16},
				 { 0, 0,16, 4,   0, 4,16, 8},
				 { 0, 8,16,12,   0,12,16,16},
			      }
--]]

local filters = nn.Sequential()
--filters:add(nn.SpatialContrastiveNormalization(n_chans,image.gaussian1D(k_norm)))
filters:add(nn.BlockFilter(filters0))
local filters2 = filters:clone()
--TODO depends on the size of the filters
local matcher = nn.BinaryMatching(hwin, wwin, 7, 8, 7, 8)
local matcher2 = nn.BinaryMatching(hwin, wwin, 7, 8, 7, 8)


function loadImgFile(i)
   return image.rgb2y(image.scale(image.load(string.format("data/%09d.jpg", i)),
				  320, 180, 'bilinear'))
end

--cam = image.Camera{idx=1, fps=30, width=320, height=240}
--io.write("Creating Ardrone ... ")
--ar = ardrone.new('/home/linaro/work/torch/packages/fifo_ardrone_cmd')
--print("done")
local fg
local fr
iim = 0
local p = xlua.Profiler()

function loadImgCam(i)
   p:start('getFrame')
   fr = ardrone.getframe(fr)
   p:lap('getFrame')
   --fr = torch.Tensor(180, 320)
   p:start('scaleFrame')
   local returnedFrame = image.scale(fr,320,180)
   p:lap('scaleFrame')
   --p:start('saveFrame')
   --image.save(string.format("imgs/%07d.jpg", iim), fr)
   --p:lap('saveFrame')
   --p:start('resizeFrame')
   --returnedFrame = fr:resize(1,180,320)
   --p:lap('resizeFrame')
   iim = iim+1
   return returnedFrame
   --fg = fg or new_framegrabber(240, 320)
   --fr = cam:forward()
   --win2 = image.display{image=fr, win=win2}
   --image.save(string.format("imgs/%07d.jpg", iim), fr)
   --return image.rgb2y(fr)
   --return image.rgb2y(fg:grab())
end

--io.write("Sync drone ... ")
--ardrone.command(ar, 0, 0, 0, 0)
--print("done")

os.execute('sleep 2')

io.write("Init Video ... ")
ardrone.initvideo()
print("done")

--io.write("Take off...")
--ardrone.takeoff(ar, 1)
--print("done")

os.execute('sleep 2')

--loadImg = loadImgFile
loadImg = loadImgCam

local i_img = 1
local im1 = loadImg(i_img)
local im12 = image.scale(im1, im1:size(3)/2, im1:size(2)/2)
local im1filtered = filters:forward(im1):clone()
local im1filtered2 = filters2:forward(im12):clone()
local win, win2, win3, win4, win5
local timer = torch.Timer()
local flow = torch.ByteTensor()
local flowreal = torch.Tensor()
local flowrealdisp = torch.Tensor()

local meantfilter = 0
local nfilter = 0
local totaltime = 0

while true do
   p:start('total')
   i_img = i_img + 1
   p:start('loadImage')
   local im2 = loadImg(i_img)
   p:lap('loadImage')
   p:start('scaleImage')
   local im22 = image.scale(im2, im1:size(3)/2, im1:size(2)/2)
   p:lap('scaleImage')
   timer:reset()
   p:start('filterIm1')
   local im2filtered = filters:forward(im2)
   p:lap('filterIm1')
   p:start('filterIm2')
   local im2filtered2 = filters2:forward(im22)
   p:lap('filterIm2')
   meantfilter = nfilter/(nfilter+1)*meantfilter + timer:time()['real']/(nfilter+1)
   nfilter = nfilter + 1
   print("toc filters : ", timer:time()['real'])
   print("mean   filters : ", meantfilter)
   local flow1, score1 = matcher:forward{im1filtered, im2filtered}
   local flow2, score2 = matcher2:forward{im1filtered2, im2filtered2}
   print("toc match   : ", timer:time()['real'])
   --local tmed = torch.Timer()
   MergeFlow(flow1, score1, flow2, score2, flow,
	     math.floor((hwin-1)/2), math.floor((wwin-1)/2), 1,1)
   --print("merge                                 ", tmed:time()['real'])
   MedianFilter(flow, 3)
   print("toc median   : ", timer:time()['real'])
   totaltime = totaltime + timer:time()['real']
   print("  ===== FPS : ",1/(timer:time()['real']), " =====")
   flowreal:resize(flow:size())
   flowreal:copy(flow)
   flowreal[1]:add(-math.floor((hwin-1)/2)*2)
   flowreal[2]:add(-math.floor((wwin-1)/2)*2)
   flowrealdisp:resizeAs(flowreal)
   flowrealdisp:copy(flowreal)
   local norm = (flowreal[1]:cmul(flowreal[1])+flowreal[2]:cmul(flowreal[2])):sqrt()
   norm = norm[{{1, norm:size(1)/2},{}}]
   local x1,y1,x2,y2 = planningOld(norm)

   local target = (x2+x1)/2-im2[1]:size(2)/2
   target = 0.25*math.max(-1,math.min(1,target/200))
   print(target)
   --ardrone.command(ar, 0.1, 0, target, 0)
   p:start('display')
   local im2color = torch.Tensor(3, im2:size(2), im2:size(3))
   im2color[1]:copy(im2[1])
   im2color[2]:copy(im2[1])
   im2color[3]:copy(im2[1])
   im2color[2][{{y1+1,y2},{x1+1,x2}}]:fill(0)
   win = image.display{image=im2color, win=win}
   p:lap('display')
   
   im1filtered = im2filtered:clone()
   im1filtered2 = im2filtered2:clone()
   p:lap('total')
   p:printAll{}
   print("---")
end
