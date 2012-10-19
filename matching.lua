require 'libmatching'
require 'math'
require 'nn'

local Binarizer, parent1 = torch.class('nn.Binarizer', 'nn.Module')
local BinaryMatching, parent2 = torch.class('nn.BinaryMatching', 'nn.Module')

function Binarizer:__init(threshold)
   parent1:__init(self)
   self.output = torch.LongTensor()
   self.threshold = threshold
end

function Binarizer:updateOutput(input)
   local k = math.ceil(input:size(1)/64) --TODO doesn't work on 32bits
   self.output:resize(input:size(2), input:size(3), k):zero()
   libmatching.binarize(input, self.output, self.threshold)
   return self.output
end

function BinaryMatching:__init(hwin, wwin)
   parent2:__init(self)
   self.hwin = hwin
   self.wwin = wwin
   self.lwin = math.floor((wwin-1)/2)
   self.rwin = math.ceil ((wwin-1)/2)
   self.twin = math.floor((hwin-1)/2)
   self.bwin = math.ceil ((hwin-1)/2)
   self.output = torch.ByteTensor()
end

function BinaryMatching:updateOutput(input)
   local input1 = input[1][{{self.twin+1, -self.bwin-1},
			    {self.rwin+1,-self.lwin-1},{}}]
   self.output:resize(2, input1:size(1), input1:size(2))
   libmatching.binaryMatching(input1, input[2], self.output, self.hwin, self.wwin)
   return self.output
end

function MedianFilter(input, k)
   k = k or 3
   libmatching.medianFilter(input, k)
   return input
end
