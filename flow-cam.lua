torch.setdefaulttensortype('torch.FloatTensor')

require 'opticalflow'
require 'image'
require 'matching'
require 'prettydisplay'
require 'framegrabber'
require 'filtering'
require 'planning'

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

local fg
function loadImgCam(i)
   fg = fg or new_framegrabber(240, 320, 0)
   return image.rgb2y(fg:grab())
end


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
   --sys.execute('sleep 1')
   i_img = i_img + 1
   local im2 = loadImg(i_img)
   local im22 = image.scale(im2, im1:size(3)/2, im1:size(2)/2)
   timer:reset()
   local im2filtered = filters:forward(im2)
   local im2filtered2 = filters2:forward(im22)
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
   local x1,y1,x2,y2 = planningOld(norm)
   --norm[{{y1+1,y2},{x1+1,x2}}]:add(10)
   local score2b = image.scale(score2:real(), score1:size(2),
			       score1:size(1), 'simple')
   local select = score1:real():gt(score2b)
   local maxflow = math.max(math.ceil((hwin-1)/2)*2,math.ceil((wwin-1)/2)*2)
   im2c = torch.Tensor(3, im2:size(2), im2:size(3))
   for i = 1,3 do
      im2c[i]:copy(im2[1])
   end
   im2c[{2,{y1+1,y2},{x1+1,x2}}]:zero()
   local todisp=prettydisplay({{im2c,norm,flow1},
					  {flowrealdisp,flow2},
					  {score1:real(), score2b:real(),select}},
					 {{{},{},{}},
					  {{-maxflow,maxflow},{}},
					  {{0,10},{0,10},{}}})
   win=image.display{image=todisp,  win=win}
   --image.save(string.format('output_video/%07d.jpg', i_img), todisp)
   --[[win = image.display{image=norm, win=win, min=0, max=20}
   win2 = image.display{image=im2, win=win2}
   win3 = image.display{image=flow1, win=win3}
   win4 = image.display{image=flow2, win=win4}
   win5 = image.display{image=flow, win=win5}--]]
   im1filtered = im2filtered:clone()
   im1filtered2 = im2filtered2:clone()
   print("---")
end
