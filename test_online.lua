torch.setdefaulttensortype('torch.FloatTensor')

require 'framegrabber'
require 'opticalflow'

local fg = new_framegrabber(240, 320)

while true do
   fg:recenter()
   local frame1 = fg:grab()
   local flow = nil
   for i = 1,10 do
      local frame2 = fg:grab()
      flow = computeOpticalFlow(frame1, frame2, flow)
      win1 = image.display{image={frame1, frame2}, win=win1}
      win2 = image.display{image=flow, win=win2, min=-20,max=20}
      frame1=frame2
   end
end