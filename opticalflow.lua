require 'nn'
require 'nnx'
require 'wavelet'
require 'matching'

function computeOpticalFlowOpenCV(im1, im2, previous, method)
   require 'opencv24'
   method = method or 'farnebach'
   local flow
   if method == 'farnebach' then
      flow = opencv24.DenseOpticalFlow{im1=im1, im2=im2, iterations=10,
				       pyr_scale=0.7,levels = 20,
				       winsize=51,poly_n=7,poly_sigma=1.5,
				       flowguess=previous}
   else
      flow = opencv24.DenseOpticalFlow{im1=im1, im2=im2,mode='block',
				       flowguess=previous,
				       winsize=5,maxrange=20}
   end
   return flow
end

function opticalFlowFastBM(hwin, wwin, filtersp)
   local lwin = math.floor((wwin-1)/2)
   local rwin = math.ceil ((wwin-1)/2)
   local twin = math.floor((hwin-1)/2)
   local bwin = math.ceil ((hwin-1)/2)
   local k_norm = 17
   local n_chans = 1

   function filter(n_freq, n_theta, n_phases, k_filter, kmax)
      local ret = nn.Sequential()
      local n_filters = n_freq*n_theta*n_phases
      ret:add(nn.SpatialConvolution(n_chans, n_filters, k_filter, k_filter))
      local k1 = math.floor((kmax-k_filter)/2)
      local k2 = math.ceil((kmax-k_filter)/2)
      ret:add(nn.SpatialPadding(-k1, -k1, -k2, -k2))
      --local k1 = math.floor((k_filter-1)/2)
      --local k2 = math.ceil ((k_filter-1)/2)
      --ret:add(nn.SpatialPadding(k1, k2, k1, k2))
      local fil = ret.modules[1]
      fil.bias:zero()
      local i = 1
      for i_freq = 1,n_freq do
	 for i_theta = 1,n_theta do
	    for i_phase = 1,n_phases do
	       local freq = math.pow(2,i_freq-1)
	       local theta = (i_theta-1)*math.pi/n_theta
	       local phase = (i_phase-1)*math.pi/n_phases
	       for k = 1,n_chans do
		  fil.weight[i][k]:copy(wavelet2d(k_filter, freq, phase, theta))
	       end
	       i = i + 1
	    end
	 end
      end
      --image.display(fil.weight)
      return ret, n_filters
   end

   local filters = nn.Sequential()
   filters:add(nn.SpatialContrastiveNormalization(n_chans, image.gaussian1D(k_norm)))
   local filtergroups = nn.ConcatTable()
   filters:add(filtergroups)
   local n_filters = 0
   for i = 1,#filtersp do
      local localfilter,n = filter(filtersp[i][1],filtersp[i][2], filtersp[i][3],
				 filtersp[i][4], filtersp[#filtersp][4])
      filtergroups:add(localfilter)
      n_filters = n_filters + n
   end
   filters:add(nn.JoinTable(1))
   filters:add(nn.Binarizer(0.1))

   local hpad = filtersp[#filtersp][4]
   local wpad = filtersp[#filtersp][4]
   local matcher = nn.BinaryMatching(hwin, wwin,
				     math.floor((hpad-1)/2),math.ceil((hpad-1)/2),
				     math.floor((wpad-1)/2),math.ceil((wpad-1)/2))

   return filters, matcher, n_filters
end









--============================================================================--
--==========                          OLD                           ==========--
--============================================================================--

function computeOpticalFlowFastBM(im1, im2, normalize)
   local lwin = 8
   local rwin = 8
   local twin = 8
   local bwin = 8
   local hwin = twin+bwin+1
   local wwin = lwin+rwin+1
   local k_norm = 17
   local n_chans = 1
   function filter(n_freq, n_theta, n_phases, k_filter, kmax)
      local ret = nn.Sequential()
      local k1 = math.floor((kmax-k_filter)/2)
      local k2 = math.ceil((kmax-k_filter)/2)
      local n_filters = n_freq*n_theta*n_phases
      ret:add(nn.SpatialConvolution(n_chans, n_filters, k_filter, k_filter))
      ret:add(nn.SpatialPadding(-k1, -k1, -k2, -k2))
      local fil = ret.modules[1]
      fil.bias:zero()
      local i = 1
      for i_freq = 1,n_freq do
	 for i_theta = 1,n_theta do
	    for i_phase = 1,n_phases do
	       local freq = math.pow(2,i_freq-1)
	       local theta = (i_theta-1)*math.pi/n_theta
	       local phase = (i_phase-1)*2*math.pi/n_phases
	       for k = 1,n_chans do
		  fil.weight[i][k]:copy(wavelet2d(k_filter, freq, phase, theta))
	       end
	       i = i + 1
	    end
	 end
      end
      image.display(fil.weight)
      return ret
   end
   local filters0 = {}
   local filtersp = {
      --{2, 8, 4, 16},
      --{1, 4, 4, 32},
      {2,4,4,16},
      --{1,2,2,32}
   }
   for i = 1,#filtersp do
      table.insert(filters0, filter(filtersp[i][1],filtersp[i][2],
				    filtersp[i][3], filtersp[i][4],
				    filtersp[#filtersp][4]))
   end
   function filtergroup(filtertable)
      local ret = nn.Sequential()
      local par = nn.ConcatTable()
      for i = 1,#filtertable do
	 par:add(filtertable[i])
      end
      ret:add(par)
      ret:add(nn.JoinTable(1))
      return ret
   end
   local filters = filtergroup(filters0)

   local filter1 = nn.Sequential()
   if normalize then
      filter1:add(nn.SpatialContrastiveNormalization(n_chans,
						     image.gaussian1D(k_norm)))
   end
   filter1:add(filters)

   local filter2 = nn.Sequential()
   if normalize then
      filter2:add(nn.SpatialContrastiveNormalization(n_chans,
						     image.gaussian1D(k_norm)))
   end
   filter2:add(filters:clone("weight", "bias", "gradWeight", "gradBias"))

   local matcher

   if full then
      matcher = nn.SpatialMatching(hwin, wwin, false)
   else
      matcher = nn.BinaryMatching(hwin, wwin)
      filter1:add(nn.Binarizer(0))
      filter2:add(nn.Binarizer(0))
   end

   local im1filtered, im2filtered
   local timer
   if n_chans == 1 then
      im1filtered = filter1:forward(image.rgb2y(im1))
      timer = torch.Timer()
      im2filtered = filter2:forward(image.rgb2y(im2))
   else
      im1filtered = filter1:forward(im1)
      timer = torch.Timer()
      im2filtered = filter2:forward(im2)
   end
   print("toc filter", timer:time()['real'])

   if full then
      local output = matcher:forward{im1filtered, im2filtered}
      output = output:resize(output:size(1), output:size(2), hwin*wwin);
      local _, idx = output:min(3)
      idx = idx:add(-1):squeeze():real()
      image.display(idx)
      local yflow = (idx/wwin):floor()
      local xflow = idx-yflow*wwin - lwin
      yflow = yflow - twin
      --return {yflow, xflow}
   else
      local flow = matcher:forward{im1filtered, im2filtered}
      print("toc flow", timer:time()['real'])
      --return flow:real()
   end

   -- AGAIN, now it's hot

   if n_chans == 1 then
      im1filtered = filter1:forward(image.rgb2y(im1))
      timer = torch.Timer()
      im2filtered = filter2:forward(image.rgb2y(im2))
   else
      im1filtered = filter1:forward(im1)
      timer = torch.Timer()
      im2filtered = filter2:forward(im2)
   end
   print("toc filter", timer:time()['real'])

   if full then
      local output = matcher:forward{im1filtered, im2filtered}
      output = output:resize(output:size(1), output:size(2), hwin*wwin);
      local _, idx = output:min(3)
      idx = idx:add(-1):squeeze():real()
      image.display(idx)
      local yflow = (idx/wwin):floor()
      local xflow = idx-yflow*wwin - lwin
      yflow = yflow - twin
      return {yflow, xflow}
   else
      local flow = matcher:forward{im1filtered, im2filtered}
      print("toc flow", timer:time()['real'])
      MedianFilter(flow, 5)
      print("toc median", timer:time()['real'])
      return flow:real()
   end

end

