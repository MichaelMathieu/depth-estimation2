require 'libfiltering'
require 'nn'
require 'image'
require 'math'

local BlockFilter, parent = torch.class('nn.BlockFilter', 'nn.Module')

function BlockFilter:__init(filters)
   parent:__init(self)
   self.output = torch.LongTensor()
   self.integral = torch.FloatTensor()
   self.filters = filters
   self.hmax = math.max((filters[{{},4}]-filters[{{},2}]):max(),
			(filters[{{},8}]-filters[{{},6}]):max())
   self.wmax = math.max((filters[{{},3}]-filters[{{},1}]):max(),
			(filters[{{},7}]-filters[{{},5}]):max())
end

function BlockFilter:updateOutput(input)
   input = input:squeeze()
   if input:nDimension() ~= 2 then
      error("BlockFilter : input must be h x w (1 channel)")
   end
   self.output:resize(input:size(1)-self.hmax, input:size(2)-self.wmax,1):zero()
   self.integral:resize(input:size(1)+1, input:size(2)+1)
   if input:stride(2) ~= 1 then
      input = input:newContiguous()
   end
   if self.integral:stride(2) ~= 1 then
      self.integral = self.integral:newContiguous()
   end
   libfiltering.integralImage(input, self.integral)
   libfiltering.filterImage(self.integral, self.filters, self.output,
			    self.hmax, self.wmax)
   return self.output
end

function IntegralImage()
   local iimage = {}
   iimage.iimage = torch.FloatTensor()
   function iimage:compute(input)
      self.iimage:resize(input:size(1)+1, input:size(2)+1)
      libfiltering.integralImage(input, self.iimage)
      return self.iimage
   end
   function iimage:isum(x1, y1, x2, y2) --x1,y1 included, x2,y2 excluded
      return self.iimage[y1+1][x1+1] + self.iimage[y2+1][x2+1] - self.iimage[y1+1][x2+1] - self.iimage[y2+1][x1+1]
   end
   return iimage
end

function filtering_testme()
   torch.setdefaulttensortype('torch.FloatTensor')
   local a = image.lena()[1]

   local filters = torch.LongTensor{{ 0, 0, 8, 8,   8, 8,16,16},
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
   local filter = nn.BlockFilter(filters)
   filter:forward(a)
end
