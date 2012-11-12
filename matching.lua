require 'libmatching'
require 'math'
require 'nn'

local Binarizer, parent1 = torch.class('nn.Binarizer', 'nn.Module')
local BinaryMatching, parent2 = torch.class('nn.BinaryMatching', 'nn.Module')

function Binarizer:__init(threshold)
   parent1:__init(self)
   self.output = torch.LongTensor()
   self.threshold = threshold
   self.wordsize = libmatching.sizeofLong()*8
   self.useNeon = libmatching.useNeon()
end

function Binarizer:updateOutput(input)
   local k
   k = math.ceil(2*input:size(1)/self.wordsize)
   self.output:resize(input:size(2), input:size(3), k):zero()
   libmatching.binarize(input, self.output, self.threshold)
   return self.output
end

function BinaryMatching:__init(hwin, wwin, tpad, bpad, lpad, rpad)
   parent2:__init(self)
   self.hwin = hwin
   self.wwin = wwin
   self.twin = math.ceil(hwin/2)-1
   self.lwin = math.ceil(wwin/2)-1
   self.bwin = math.floor(hwin/2)
   self.rwin = math.floor(wwin/2)
   self.lpad = (lpad or 0)
   self.tpad = (tpad or 0)
   self.bpad = (bpad or 0)
   self.rpad = (rpad or 0)
   self.hpad = self.tpad+self.bpad
   self.wpad = self.rpad+self.lpad
   self.output = torch.ByteTensor()
   self.outputscore = torch.LongTensor()
end

function BinaryMatching:updateOutput(input)
   self.output:resize(2, input[1]:size(1)+self.hpad,
		      input[1]:size(2)+self.wpad):zero()
   --TODO zero suboptimal
   local input1 = input[1][{{self.twin+1, -self.bwin-1},
			    {self.lwin+1, -self.rwin-1}, {}}]
   self.outputscore:resize(input[1]:size(1)+self.hpad+self.twin,
			   input[1]:size(2)+self.wpad+self.lwin):fill(-1)
   local output1 = self.output:narrow(2, self.tpad+self.twin+1, input1:size(1))
                              :narrow(3, self.lpad+self.lwin+1, input1:size(2))
   local outputscore1 =
      self.outputscore:narrow(1, self.tpad+self.twin+1, input1:size(1))
                      :narrow(2, self.lpad+self.lwin+1, input1:size(2))
   libmatching.binaryMatching(input1, input[2], output1, outputscore1,
			      self.hwin, self.wwin)
   return self.output, self.outputscore
end

function MedianFilter(input, k)
   k = k or 3
   libmatching.medianFilter(input, k)
   return input
end

function MergeFlow(input1, input1score, input2, input2score, output,
		   hhwin, hwwin, n_filters1, n_filters2)
   output:resizeAs(input1)
   if n_filters1 ~= 1 then
      input2score:mul(n_filters1)
   end
   if n_filters2 ~= 1 then
      input1score:mul(n_filters2)
   end
   libmatching.merge(input1, input1score, input2, input2score, output,
		     hhwin, hwwin)
   return output
end
