require 'math'
require 'image'

function wavelet1d(n, rho, phi)
   local ret = torch.Tensor(n)
   for i = 1,n do
      ret[i] = math.sin(i*2*math.pi/n*rho+phi)
   end
   ret:cmul(image.gaussian1D(n))
   return ret:div(ret:norm())
end

function wavelet2d(n, rho, phi, theta)
   local ret = torch.Tensor(n, n)
   local ct = math.cos(theta)
   local st = math.sin(theta)
   for i = 1,n do
      for j = 1,n do
	 local x = ct*i-st*j
	 --local y = st*i+ct*j
	 ret[i][j] = math.sin(x*2*math.pi/n*rho+phi)
      end
   end
   ret:cmul(image.gaussian(n))
   return ret:div(ret:norm())
end

function wavelet1d_testme()
   require 'gnuplot'
   gnuplot.plot({'0',wavelet1d(420, 6,0)},
		{'1',wavelet1d(420, 6,math.pi/4)},
		{'2',wavelet1d(420, 6,math.pi/2)},
		{'3',wavelet1d(420, 6,3*math.pi/4)})
end

function wavelet2d_testme()
   image.display(wavelet2d(17, 6, 0, 1.2))
end