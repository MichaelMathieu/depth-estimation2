require 'matching'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local im = image.load('lena.png')
local im2 = image.load('lena2.png')
local bin = torch.LongTensor(im:size(2), im:size(3)):zero()
local bin2 = torch.LongTensor(im:size(2), im:size(3)):zero()

matching.binarize(im, bin, 0.5)
matching.binarize(im2, bin2, 0.5)
bin1b = bin[{{25,475},{25,475}}]

local match = torch.LongTensor(2, im:size(2), im:size(3))
matching.binaryMatching(bin1b, bin2, match, 50, 50)

image.display({bin:float(), bin2:float()})
image.display(match)