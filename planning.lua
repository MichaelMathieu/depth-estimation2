require 'filtering'

function planningStep(idepthmap, x1, y1, x2, y2)
   local xm2 = (x1+x2)/2
   local ym2 = (y1+y2)/2
   local xm1 = (x1+xm2)/2
   local ym1 = (y1+ym2)/2
   local xm3 = (xm2+x2)/2
   local ym3 = (ym2+y2)/2
   local vals = torch.Tensor({
      idepthmap:isum(x1 , y1 , xm2, ym2),
      idepthmap:isum(xm1, y1 , xm3, ym2),
      idepthmap:isum(xm2, y1 , x2 , ym2),
      idepthmap:isum(x1 , ym1, xm2, ym3),
      idepthmap:isum(xm1, ym1, xm3, ym3),
      idepthmap:isum(xm2, ym1, x2 , ym3),
      idepthmap:isum(x1 , ym2, xm2, y2 ),
      idepthmap:isum(xm1, ym2, xm3, y2 ),
      idepthmap:isum(xm2, ym2, x2 , y2 ),
   })
   _, idx = vals:min(1)
   idx = idx[1]
   if idx == 1 then
      return x1, y1, xm2, ym2
   elseif idx == 2 then
      return xm1, y1, xm3, ym2
   elseif idx == 3 then
      return xm2, y1, x2, ym2
   elseif idx == 4 then
      return x1, ym1, xm2, ym3
   elseif idx == 5 then
      return xm1, ym1, xm3, ym3
   elseif idx == 6 then
      return xm2, ym1, x2, ym3
   elseif idx == 7 then
      return x1, ym2, xm2, y2
   elseif idx == 8 then
      return xm1, ym2, xm3, y2
   elseif idx == 9 then
      return xm2, ym2, x2, y2
   end
end

function planningOld(depthmap)
   local idepthmap = IntegralImage()
   idepthmap:compute(depthmap)
   local x1 = 30
   local y1 = 30
   local x2 = depthmap:size(2)-30
   local y2 = depthmap:size(1)-30
   for i = 1,2 do
      x1, y1, x2, y2 = planningStep(idepthmap, x1, y1, x2, y2)
   end
   return x1, y1, x2, y2
end

oldm = 10000000
function planningStep2(idepthmap, x1, y1, x2, y2)
   local xm2 = (x1+x2)/2
   local ym2 = (y1+y2)/2
   local xm1 = (x1+xm2)/2
   local ym1 = (y1+ym2)/2
   local xm3 = (xm2+x2)/2
   local ym3 = (ym2+y2)/2
   local vals = torch.Tensor({
      idepthmap:isum(x1 , y1 , xm2, ym2),
      idepthmap:isum(xm1, y1 , xm3, ym2),
      idepthmap:isum(xm2, y1 , x2 , ym2),
      idepthmap:isum(x1 , ym1, xm2, ym3),
      idepthmap:isum(xm1, ym1, xm3, ym3),
      idepthmap:isum(xm2, ym1, x2 , ym3),
      idepthmap:isum(x1 , ym2, xm2, y2 ),
      idepthmap:isum(xm1, ym2, xm3, y2 ),
      idepthmap:isum(xm2, ym2, x2 , y2 ),
   })
   m, idx = vals:min(1)
   m = m[1]
   if m < 1.2*oldm then
      return oldx1, oldy1, oldx2, oldy2
   else
      idx = idx[1]
      oldm = m
      if idx == 1 then
	 return x1, y1, xm2, ym2
      elseif idx == 2 then
	 return xm1, y1, xm3, ym2
      elseif idx == 3 then
	 return xm2, y1, x2, ym2
      elseif idx == 4 then
	 return x1, ym1, xm2, ym3
      elseif idx == 5 then
	 return xm1, ym1, xm3, ym3
      elseif idx == 6 then
	 return xm2, ym1, x2, ym3
      elseif idx == 7 then
	 return x1, ym2, xm2, y2
      elseif idx == 8 then
	 return xm1, ym2, xm3, y2
      elseif idx == 9 then
	 return xm2, ym2, x2, y2
      end
   end
end

function planning(depthmap)
   local idepthmap = IntegralImage()
   idepthmap:compute(depthmap)
   local x1 = 30
   local y1 = 30
   local x2 = depthmap:size(2)-30
   local y2 = depthmap:size(1)-30
   x1, y1, x2, y2 = planningStep(idepthmap, x1, y1, x2, y2)
   oldx1 = x1
   oldx2 = x2
   oldy1 = y1
   oldy2 = y2
   
   return x1, y1, x2, y2
end