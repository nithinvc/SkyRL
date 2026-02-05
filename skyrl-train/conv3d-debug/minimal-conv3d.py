import torch


temporal_psize = 2
in_chans = 3
out_chans = 1024
psize = 16

conv_kernel = [temporal_psize, psize, psize]
conv_proj = torch.nn.Conv3d(in_chans, out_chans, kernel_size=conv_kernel, stride=conv_kernel, bias=True)

inp = torch.load('/mnt/local_storage/mm-tensor.pt')['pixel_values'].cuda()
print('loaded input')
conv_proj = conv_proj.cuda().bfloat16()
inp = inp.view(-1, in_chans, temporal_psize, psize, psize)
print('viewed input')
import time
start = time.perf_counter()
out = conv_proj(inp)
end = time.perf_counter()

torch.cuda.synchronize()
print('synchronized')
print(f'forward pass time: {end - start}')