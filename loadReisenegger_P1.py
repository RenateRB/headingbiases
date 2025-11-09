import torch
import glob
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import ToDevice

files = sorted(glob.glob("data/*P1.mat"))
# ego=0
# allo circle=1
 

from scipy.io import loadmat

target = []
response = []
condition = []
centralOrientationPerSubject = []

for f in files:
 annots = loadmat(f)
 sample = MakeFloatTensor(annots["heading"])
 responses = MakeFloatTensor(annots["response"])
 conditions = MakeFloatTensor(annots["condition"])
 target.append(sample)
 response.append(responses)
 condition.append(conditions)
 centralOrientation = sample.mean()
 centralOrientationPerSubject.append(centralOrientation)
target = torch.stack(target, dim=0)
response = torch.stack(response, dim=0)
condition = torch.stack(condition, dim=0)
centralOrientationPerSubject = torch.stack(centralOrientationPerSubject, dim=0)
#print(f"CentralOrientationPerSubject: {centralOrientationPerSubject}. Size: {centralOrientationPerSubject.size()}")
centralOrientationPerTrial = centralOrientationPerSubject.view(-1, 1).expand(-1, target.size()[1])
#print(f"centralOrientationPerTrial: {centralOrientationPerTrial}. Size: {centralOrientationPerTrial.size()}")
subject = MakeFloatTensor(list(range(target.size()[0]))).view(-1, 1).expand(-1, target.size()[1])
#print(f"subject: {subject}. Size: {subject.size()}")
target = target.view(-1).contiguous()
response = response.view(-1).contiguous()
condition = condition.view(-1).contiguous()
centralOrientationPerTrial = centralOrientationPerTrial.contiguous().view(-1).contiguous()
Subject = subject.contiguous().view(-1).contiguous()
#print(f"Subject: {Subject}. Size: {Subject.size()}")

observations_x = target
observations_y = response



sample=target

