require 'image'
require 'camera'
require 'sfm2'

function recenter(fg)
   fg.R0 = torch.mm(mongoose.getRotmat(fg.mg), fg.P)
end

function new_framegrabber(h, w, camidx, use_mongoose)
   local cam = image.Camera{idx = camidx, width = w, height = h, nbuffers = 1, fps = 30}
   local dist = torch.FloatTensor({-0.0142,-0.0045,-0.0011,0.0056,-0.5707})
   local K = torch.FloatTensor({{818.3184,        0, 333.2290},
				{       0, 818.4109, 230.9768},
				{       0,        0,        1}})
   local fg = {cam=cam, K=K, dist=dist}
   if use_mongoose then
      require 'mongoose'
      local mg = mongoose.new('/dev/ttyUSB0')
      while mongoose.getRotmat(mg) == nil do
	 mongoose.fetchData(mg)
      end
      local P = torch.FloatTensor({{1,0,0},
				   {0,0,1},
				   {0,1,0}})
      K:mul(w/640)
      K[3][3]=1
      fg = {cam=cam, mg=mg, P=P, K=K, dist=dist}
   
      function fg:recenter()
	 mongoose.fetchData(self.mg)
	 print(self.P)
	 print(mongoose.getRotmat(self.mg):float())
	 self.R0 = torch.mm(mongoose.getRotmat(self.mg):float(), self.P)
      end
      fg:recenter()
      
      local R = torch.FloatTensor(3,3)
      function fg:grab()
	 mongoose.fetchData(self.mg)
	 R:copy(mongoose.getRotmat(self.mg))
	 local frame = self.cam:forward()
	 mongoose.fetchData(self.mg)
	 --R:copy(mongoose.getRotmat(self.mg))
	 R:add(mongoose.getRotmat(self.mg):float()):mul(0.5)
	 local R = torch.mm(sfm2.inverse(torch.mm(R, P)), self.R0)
	 frame = sfm2.undistortImage(frame, self.K, self.dist)
	 frame = sfm2.removeEgoMotion(frame, self.K, R, 'bilinear')
	 return frame
      end
   else
      function fg:recenter()
      end
      function fg:grab()
	 local frame = self.cam:forward()
	 frame = sfm2.undistortImage(frame, self.K, self.dist)
	 return frame
      end
   end
      
   return fg
end

function framegrabber_testme()
   local fg = new_framegrabber(480, 640)
   local k = 0
   while true do
      for i = 1,10 do
	 local frame = fg:grab()
	 image.save(string.format('output2/%07d.jpg', k), frame)
	 k = k + 1
	 win=image.display{image=frame, win=win}
      end
      --fg:recenter()
   end
end