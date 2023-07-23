import glob
import os
import subprocess


def batch_train(in_dir,out_dir):
    blocks_paths=glob.glob(os.path.join(in_dir,"*"))
    for block_path in blocks_paths:
        block_id=int(os.path.basename(block_path))
        if block_id<=0:
            continue
        cmd=["python","scripts/train.py","mct_mipnerf",
             "--data="+block_path,
               "--output-dir="+out_dir,
               "--timestamp=0",
               "--max-num-iterations=30000"]
        print(cmd)
        subprocess.call(cmd)



batch_train(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks',r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks64')
