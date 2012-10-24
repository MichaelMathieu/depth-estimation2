require 'math'

local function height(im)
   if im:nDimension() == 3 then
      return im:size(2)
   elseif im:nDimension() == 2 then
      return im:size(1)
   else
      error('prettydisplay.height : image must be 2D or 3D')
   end
end

local function width(im)
   if im:nDimension() == 3 then
      if im:size(1) == 3 then
	 return im:size(3)
      else
	 local output = 0
	 for i = 1,im:size(1) do
	    output = output + width(im[i])
	 end
	 return output
      end
   elseif im:nDimension() == 2 then
      return im:size(2)
   else
      error('prettydisplay.width : image must be 2D or 3D')
   end
end

local function printin(src, dst, y, x, ranges)
   local h = height(src)
   local w = width(src)
   src = src:real()

   if ranges[1] ~= "no" then
      vmin = ranges[1] or src:min()
      vmax = ranges[2] or src:max()
      local vrange = vmax-vmin
      if vrange ~= 0 then
	 src:add(-vmin)
	 src:div(vrange)
      end
   end

   if src:nDimension() == 3 then
      if src:size(1) ~= 3 then
	 local wlocal = 0
	 for i = 1,src:size(1) do
	    printin(src[i], dst, y, x+wlocal, {"no"})
	    wlocal = wlocal + width(src[i])
	 end
      else
	 dst[{{},{y, y+h-1},{x,x+w-1}}]:copy(src)
      end
   elseif src:nDimension() == 2 then
      dst[{1,{y, y+h-1},{x,x+w-1}}]:copy(src)
      dst[{2,{y, y+h-1},{x,x+w-1}}]:copy(src)
      dst[{3,{y, y+h-1},{x,x+w-1}}]:copy(src)
   else
      error('prettydisplay.printin : image must be 2D or 3D')
   end
end


function prettydisplay(input, ranges)
   local h = 0
   local w = 0
   local hs = {}
   for i = 1,#input do
      local line = input[i]
      local hlocal = 0
      local wlocal = 0
      for j = 1,#line do
	 hlocal = math.max(hlocal, height(line[j]))
	 wlocal = wlocal + width(line[j])
      end
      hs[i] = h
      h = h + hlocal
      w = math.max(w, wlocal)
   end
   local todisp = torch.Tensor(3, h, w):zero()
   for i = 1,#input do
      local line = input[i]
      local lineranges = ranges[i]
      local wlocal = 0
      for j = 1,#line do
	 printin(line[j], todisp, hs[i]+1, wlocal+1, lineranges[j])
	 wlocal = wlocal + width(line[j])
      end
   end
   return todisp
end