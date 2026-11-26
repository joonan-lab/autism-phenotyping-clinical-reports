"""Extract sentence CLS states from a memory-mapped token-state array."""
from pathlib import Path
import argparse
import sys
import numpy as np
_REPO=next(p for p in Path(__file__).resolve().parents if (p/'analysis_paths.py').is_file())
sys.path.insert(0,str(_REPO))
import analysis_paths as paths

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=paths.INTER/'lhs_embs_np.npy')
    ap.add_argument('--output',type=Path,default=paths.INTER/'sentence_cls.npy')
    args=ap.parse_args()
    if not args.input.is_file() or args.input.stat().st_size==0:raise FileNotFoundError(f'Token-state array is missing or empty: {args.input}')
    source=np.load(args.input,mmap_mode='r',allow_pickle=False)
    if source.ndim!=4:raise ValueError('Expected report × sentence × token × hidden-size array')
    shape=(source.shape[0],source.shape[1],source.shape[3])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_name(args.output.name+'.partial')
    target=np.lib.format.open_memmap(partial,mode='w+',dtype=source.dtype,shape=shape)
    try:
        for i in range(shape[0]):
            target[i]=source[i,:,0,:]
            if (i+1)%100==0:print(f'Extracted {i+1}/{shape[0]} reports',flush=True)
        target.flush();del target
        partial.replace(args.output)
    except BaseException:
        del target
        partial.unlink(missing_ok=True)
        raise
    print(f'Wrote {shape} {source.dtype} CLS states to {args.output}')
if __name__=='__main__':main()
