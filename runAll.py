import sys
import random
import subprocess
import os
script = sys.argv[1]


for _ in range(10000):
   P = random.choice([0,1,2,4,6,8])
   fold = random.randint(0,9)
   targetScript = script
   if P == 0:
     targetScript = script.replace(".py", "_L0.py")
   elif P == 1:
     targetScript = script.replace(".py", "_L1.py")
   targetFile = f"losses/{targetScript}_{P}_{fold}_10.0_180.txt"
   if os.path.exists(targetFile):
       continue
   print(f"Running {targetScript} with P={P}, fold={fold}")
   subprocess.call(["/Users/mhahn/anaconda3/envs/python39/bin/python3", targetScript, str(P), str(fold), "10.0", "180"])

