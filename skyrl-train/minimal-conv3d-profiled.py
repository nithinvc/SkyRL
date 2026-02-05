import torch
from torch.profiler import profile, record_function, ProfilerActivity

temporal_psize = 2
in_chans = 3
out_chans = 1024
psize = 16

conv_kernel = [temporal_psize, psize, psize]
conv_proj = torch.nn.Conv3d(in_chans, out_chans, kernel_size=conv_kernel, stride=conv_kernel, bias=True)
print('created conv_proj')

inp = torch.load('/mnt/local_storage/mm-tensor.pt')['pixel_values'].cuda()
conv_proj = conv_proj.cuda().bfloat16()
inp = inp.view(-1, in_chans, temporal_psize, psize, psize)
print('viewed input')
# Warmup run
with torch.no_grad():
    _ = conv_proj(inp)
    torch.cuda.synchronize()
print('warmup done')
# Profile the Conv3d operation
print('profiling')
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True,
    # with_flops=True,
) as prof:
    print('in profile block')
    with record_function("conv3d_forward"):
        out = conv_proj(inp)
        torch.cuda.synchronize()
print('forward pass done')
# Print summary
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# print("\n" + "="*80 + "\n")
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))

# Export profiles
prof.export_chrome_trace("./profiles/conv3d_trace.json")
print('exported chrome trace')
prof.export_stacks("./profiles/conv3d_stacks.txt", "self_cuda_time_total")

print("\nProfiles exported to ./profiles/")
print("  - conv3d_trace.json (Chrome trace, open with chrome://tracing)")
print("  - conv3d_stacks.txt (Stack traces)")
