require 'nn'
require 'image'
require 'xlua'

op = xlua.OptionParser('%prog [options]')
opt,args = op:parse()

local im = image.load(args[1])
im = im:squeeze()
if im:nDimension() == 3 then
   im = image.rgb2y(im)
end
local out = nn.SpatialContrastiveNormalization(1,image.gaussian1D(17)):forward(im)
image.save('tmp.png', out)