import torch


temporal_psize = 2
in_chans = 3
out_chans = 1024
psize = 16

conv_kernel = [temporal_psize, psize, psize]
conv_proj = torch.nn.Conv3d(in_chans, out_chans, kernel_size=conv_kernel, stride=conv_kernel, bias=True)

inp = torch.load('/mnt/local_storage/mm-tensor.pt')['pixel_values'].cuda()
conv_proj = conv_proj.cuda().bfloat16()
inp = inp.view(-1, in_chans, temporal_psize, psize, psize)
out = conv_proj(inp)