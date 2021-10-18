import torch

FIFTY_MIL_CYCLES=500
# FIFTY_MIL_CYCLES=5000
# FIFTY_MIL_CYCLES=50000
# FIFTY_MIL_CYCLES=500000
# FIFTY_MIL_CYCLES=5000000
# FIFTY_MIL_CYCLES=50000000 # this is the default value and should take about 50 ms

def _test_copy_sync_current_stream(x, y):
    x_plus_one = x + 1
    s0 = torch.cuda.Stream(device=x.device)
    s1 = torch.cuda.Stream(device=y.device)
    s2 = torch.cuda.Stream(device=x.device)
    s3 = torch.cuda.Stream(device=y.device)

    # same dst stream different src streams
    with torch.cuda.stream(s0):
        torch.cuda._sleep(FIFTY_MIL_CYCLES)
        with torch.cuda.stream(s1):
            y.copy_(x_plus_one)

    with torch.cuda.stream(s2), torch.cuda.stream(s1):
        y.copy_(x)

    s1.synchronize()
    # The copy() is synchronized on the current streams of both src and dst.
    # In the above test, the _sleep() op on s0 will not block the copy() on
    # s2, but both copies are synchronized on s1 in the dst device. Hence,
    # x is copied to y after x_plus_one is copied to y. If x and y are on
    # the same device, both copy() ops are synchronized on s1.
    # self.assertEqual(y, x)

    # same src stream different dst streams
    with torch.cuda.stream(s1):
        torch.cuda._sleep(FIFTY_MIL_CYCLES)
        with torch.cuda.stream(s0):
            y.copy_(x_plus_one)

    with torch.cuda.stream(s3), torch.cuda.stream(s0):
        y.copy_(x)

    s0.synchronize()
    # Similarly, both copy() ops are synchronized on s0.
    # self.assertEqual(y, x)


d0 = torch.device('cuda:0')
x0 = torch.zeros(5, 5, device=d0)

d1 = torch.device('cuda:1')
x1 = torch.zeros(5, 5, device=d1)
_test_copy_sync_current_stream(x0, x1)

# x2 = torch.zeros(5, 5, device=d0)
# _test_copy_sync_current_stream(x0, x2)
