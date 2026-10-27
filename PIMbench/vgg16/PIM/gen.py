# usage: python3 gen.py cin cout h w kh kw stride pad lo hi seed
import sys, torch, torch.nn.functional as F
cin,cout,h,w,kh,kw,s,p,lo,hi,seed = map(int, sys.argv[1:12])
g = torch.Generator().manual_seed(seed)
x  = torch.randint(lo, hi, (1,cin,h,w), generator=g)
wt = torch.randint(lo, hi, (cout,cin,kh,kw), generator=g)
y  = F.conv2d(x.double(), wt.double(), stride=s, padding=p).round().long()
assert y.abs().max() < 2**31, "output too large for int"
def save(t, fn):
    open(fn, 'w').write(' '.join(map(str, t.flatten().tolist())))
save(x,  'test_input.txt')
save(wt, 'test_weight.txt')
save(y,  'test_output_expected.txt')
open('test_meta.txt', 'w').write(f"{cin} {cout} {h} {w} {kh} {kw} {s} {p}\n")