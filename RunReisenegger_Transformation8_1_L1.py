# This file originated as
#  cp RunReisenegger_FreePrior_DifFreeResources_byCentralOrientation_Simplified_Shift_Co1_Prior_SepEnc_Debug.py RunReisenegger_Transformation1.py

import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import torch
from torch.optim.lr_scheduler import ExponentialLR
from l1Estimator import L1Estimator
from loadReisenegger_P2 import *
from matplotlib import rc
from util import MakeFloatTensor
from util import MakeLongTensor
from util import MakeZeros
from util import computeCenteredMean
from util import computeCircularMean
from util import computeCircularMeanWeighted
from util import computeCircularSD
from util import computeCircularSDWeighted
from util import makeGridIndicesCircular
from util import mean
from util import product
from util import savePlot
from util import toFactor

__file__ = __file__.split("/")[-1]
rc('font', **{'family':'Arial'})

OPTIMIZER_VERBOSE = False

P = int(sys.argv[1])
assert P == 1
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
SHOW_PLOT = (len(sys.argv) < 6) or (sys.argv[5] == "SHOW_PLOT")
DEVICE = 'cuda'
#assert GRID == 180

FOURIER_BASIS_SIZE = 50 #int(sys.argv[5])
assert FOURIER_BASIS_SIZE in [30, 50, 80]
#FILE = f"logs/CROSSVALID/{__file__.replace('_VIZ', '')}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt" #'C:\\Users\\Tate\\Documents\\unifying-theory-biases-main\\code\\Reisenegger\\logs\\CROSSVALID\\RunReisenegger_FreePrior_DifFreeResources.py_2_0_10.0_180.txt'

# Helper Functions dependent on the device

##############################################
# ADD CONDITION AND CENTRAL ORIENTATION



# Store observations
assert (observations_x == sample).all()
assert (observations_y == response).all()

observations_x = observations_x + 180
observations_y = observations_y + 180
sample = sample + 180
response = response + 180


# Part: Partition data into folds. As described in the paper,
# this is done within each subject.
N_FOLDS = 10
assert FOLD_HERE < N_FOLDS
randomGenerator = random.Random(10)

Fold = 0*Subject
N_SUBJECTS = int(max(Subject))+1
COs = torch.unique(centralOrientationPerTrial)
CO_size = COs.size()
N_CO = CO_size[0]

#for i in range(int(min(Subject)), int(max(Subject))+1):
for i in range(int(min(Subject)), N_SUBJECTS):
    trials = [j for j in range(Subject.size()[0]) if Subject[j] == i]
    randomGenerator.shuffle(trials)
    foldSize = int(len(trials)/N_FOLDS)
    for k in range(N_FOLDS):
        Fold[trials[k*foldSize:(k+1)*foldSize]] = k

##############################################
# Part: Set up the discretized grid
MIN_GRID = 0
MAX_GRID = 360

CIRCULAR = True
INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS = GRID/(MAX_GRID-MIN_GRID)

grid = MakeFloatTensor([x/GRID * (MAX_GRID-MIN_GRID) for x in range(GRID)]) + MIN_GRID
grid_indices = MakeFloatTensor([x for x in range(GRID)])
grid, grid_indices_here = makeGridIndicesCircular(GRID, MIN_GRID, MAX_GRID)
assert grid_indices_here.max() >= GRID, grid_indices_here.max()

# Part: Project observed stimuli onto grid
xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)

x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

##############################################
# Part: Specify `similarity` or `difference` functions.

STIMULUS_SPACE_VOLUME = MAX_GRID-MIN_GRID
SENSORY_SPACE_VOLUME = 2*math.pi

# Part: Specify `similariy` or `difference` functions.
## These are negative squared distances (for interval spaces) or
## trigonometric functions (for circular spaces), with
## some extra factors for numerical purposes.
## Exponentiating a `similarity` function and normalizing
## is equivalent to the Gaussian / von Mises density.
## The purpose of specifying these as `closeness` or `distance`,
## rather than simply calling squared or trigonometric
## functions is to  flexibly reuse the same model code for
## both interval and circular spaces.
def SQUARED_STIMULUS_DIFFERENCE(x):
    return torch.sin(math.pi*x/180)
def SQUARED_STIMULUS_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    stimulus space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    return torch.cos(math.pi*x/180)
def SQUARED_SENSORY_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    sensory space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    return torch.cos(x)
def SQUARED_SENSORY_DIFFERENCE(x):
    return torch.sin(x)

#############################################################
# Part: Configure the appropriate estimator for minimizing the loss function
assert P == 1

# Part: Import/define the appropriate estimator for minimizing the loss function
L1Estimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE)


def computeBias(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, centralOrientation=None, sigma_stimulus=None, sigma2_stimulus=0, condition_=None, folds=None, lossReduce='mean'):
 #print(f"Current CO: {COs[centralOrientation]}")
 motor_variance = torch.exp(- parameters["log_motor_var"][centralOrientation, condition_])
 # Part: Obtain the sensory noise variance.
 sigma2 = 2*torch.sigmoid(sigma_logit) #maybe change 2 for 4?
#  print(f"sigma2: {sigma2}")
 # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 if True:
  # Part: Select data for the relevant fold
  folds = MakeLongTensor(folds)
  if subject is not None:
    print("Subject is not None!!!")
    MASK = torch.logical_and(condition==condition_, torch.logical_and((Fold.unsqueeze(0) == folds.unsqueeze(1)).any(dim=0), Subject==subject))
    stimulus = stimulus_[MASK]
    responses = responses_[MASK]
  else:
    MASK = torch.logical_and(condition==condition_, torch.logical_and((Fold.unsqueeze(0) == folds.unsqueeze(1)).any(dim=0), centralOrientationPerTrial==COs[centralOrientation]))
    #print(f"MASK: {MASK}")
    stimulus = stimulus_[MASK]
    responses = responses_[MASK]
  assert stimulus.view(-1).size()[0] > 0

  
  # Part: Compute sensory likelihoods. Across both interval and
  ## circular stimulus spaces, this amounts to exponentiaring a
  ## `similarity`
  sensory_likelihoods = torch.softmax(((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)))/(sigma2)) + volumeElement.unsqueeze(1).log(), dim=0)
  # if torch.isnan(sensory_likelihoods).any():
  #    #print(f"NaNs in sensory_likelihoods in line 146. sensory_likelihoods: {sensory_likelihoods}")
  #    print(f"F: {F}.")
  #    print(f"VolumeElement: {volumeElement}. ")
     
     #print(f"SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)): {SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1))}")
  # Part: If stimulus noise is nonzero, convolve the likelihood with the
  ## stimulus noise.
  if sigma2_stimulus == 0:
    likelihoods = sensory_likelihoods
  else:
    ## On this dataset, this is zero, so the
    ## code block will not be used.
    assert False
    likelihoods = torch.matmul(sensory_likelihoods, stimulus_likelihoods)

  ## Compute posterior using Bayes' rule. As described in the paper, the posterior is computed
  ## in the discretized stimulus space.
  posterior = prior.unsqueeze(1) * likelihoods.t()

  posterior = posterior / posterior.sum(dim=0, keepdim=True)
  # if torch.isnan(posterior).any():
  #    print(f"NaNs in posterior in line 165.")

  ## Compute the estimator for each m in the discretized sensory space.
  bayesianEstimate = L1Estimator.apply(grid_indices_here, posterior)

  # now we a round of mapping
  sigma2_t = 10+100*torch.sigmoid(init_parameters["sigma2_t"]) #maybe change 2 for 4?
 #  print(f"sigma2: {sigma2}")
  # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)

  mapping_likelihoods = torch.softmax(-(360/GRID*bayesianEstimate.unsqueeze(0) - grid.unsqueeze(1)).pow(2) / (sigma2_t), dim=0)
#  print(mapping_likelihoods)
  # now a second transfer
  sigma2_t2 = 2*torch.sigmoid(init_parameters["sigma2_t2"]) #maybe change 2 for 4?
  F_t = torch.cat([MakeZeros(1), torch.cumsum(torch.softmax(init_parameters["f_t"], dim=0), dim=0)], dim=0)
  transfer_likelihoods = torch.softmax(-(F_t[:-1].unsqueeze(0) - F_t[:-1].unsqueeze(1)).pow(2) / (sigma2_t2), dim=0)
  print("f_t", torch.softmax(init_parameters["f_t"], dim=0))

  likelihood_including_trafo = torch.matmul(transfer_likelihoods, torch.matmul(mapping_likelihoods, likelihoods))


  ## Compute posterior using Bayes' rule. As described in the paper, the posterior is computed
  ## in the discretized stimulus space.
  posterior_second = prior.unsqueeze(1) * likelihood_including_trafo.t()

  posterior_second = posterior_second / posterior_second.sum(dim=0, keepdim=True)
  # if torch.isnan(posterior).any():
  #    print(f"NaNs in posterior in line 165.")

  bayesianEstimateSecond = L1Estimator.apply(grid_indices_here, posterior_second)
#  print(bayesianEstimateSecond, "hatTheta")

  ## Compute the motor likelihood
  ## `error' refers to the stimulus similarity between the estimator assigned to each m and
  ## the observations found in the dataset.
  ## The Gaussian or von Mises motor likelihood is obtained by exponentiating and normalizing
  error = (SQUARED_STIMULUS_SIMILARITY(360/GRID*bayesianEstimateSecond.unsqueeze(0) - responses.unsqueeze(1)))



  ## The log normalizing constants, for each m in the discretized sensory space
  log_normalizing_constant = torch.logsumexp((SQUARED_STIMULUS_SIMILARITY(grid))/motor_variance, dim=0) + math.log(2 * math.pi / GRID)
  ## The log motor likelihoods, for each pair of sensory encoding m and observed human response
  log_motor_likelihoods = (error/motor_variance) - log_normalizing_constant
  ## Obtaining the motor likelihood by exponentiating.
  motor_likelihoods = torch.exp(log_motor_likelihoods)
  ## Obtain the guessing rate, parameterized via the (inverse) logit transform as described in SI Appendix
  # Mixture of estimation and uniform response
  uniform_part = torch.sigmoid(parameters["mixture_logit"][centralOrientation])
  ## The full likelihood then consists of a mixture of the motor likelihood calculated before, and the uniform
  ## distribution on the full space.
  motor_likelihoods = (1-uniform_part) * motor_likelihoods + (uniform_part / (2*math.pi) + 0*motor_likelihoods)

  

  print(likelihood_including_trafo)

  overall_likelihood = torch.matmul(motor_likelihoods, likelihood_including_trafo)
#  print(mapping_likelihoods)
  

  # Now the loss is obtained by marginalizing out m from the motor likelihood
  if lossReduce == 'mean':
    assert False
    loss = -torch.gather(input=torch.matmul(motor_likelihoods, likelihoods),dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().mean()
  elif lossReduce == 'sum':
    loss = -torch.gather(input=overall_likelihood,dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().sum()
  else:
    assert False

  ## If computePredictions==True, compute the bias and variability of the estimate
  if computePredictions:
     bayesianEstimate_byStimulus = bayesianEstimateSecond.unsqueeze(1)/INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS
#     print(bayesianEstimateSecond.size(), bayesianEstimate.size(), overall_likelihood.size(), likelihoods.size())
     bayesianEstimate_avg_byStimulus = computeCircularMeanWeighted(bayesianEstimate_byStimulus, likelihood_including_trafo)
     bayesianEstimate_sd_byStimulus = computeCircularSDWeighted(bayesianEstimate_byStimulus, likelihood_including_trafo)
     bayesianEstimate_sd_byStimulus = torch.sqrt(bayesianEstimate_sd_byStimulus.pow(2) + motor_variance * 3282.806)
     #bayesianEstimate_sd_byStimulus = (bayesianEstimate_sd_byStimulus.pow(2) + motor_variance * math.pow(180/math.pi,2)).sqrt()

     bayesianEstimate_avg_byStimulus = torch.where((bayesianEstimate_avg_byStimulus-grid).abs()<180, bayesianEstimate_avg_byStimulus, torch.where(bayesianEstimate_avg_byStimulus > 180, bayesianEstimate_avg_byStimulus-360, bayesianEstimate_avg_byStimulus+360))
     assert float(((bayesianEstimate_avg_byStimulus-grid).abs()).max()) <= 180, float(((bayesianEstimate_avg_byStimulus-grid).abs()).max())
     posteriorMaxima = grid[posterior.argmax(dim=0)]
     posteriorMaxima = computeCircularMeanWeighted(posteriorMaxima.unsqueeze(1), likelihoods)
     encodingBias = computeCircularMeanWeighted(grid.unsqueeze(1), likelihoods)
     attraction = (posteriorMaxima-encodingBias)
     attraction1 = attraction
     attraction2 = attraction+360
     attraction3 = attraction-360
     attraction = torch.where(attraction1.abs() < 180, attraction1, torch.where(attraction2.abs() < 180, attraction2, attraction3))
     encodingBias = encodingBias-grid
     encodingBias1 = encodingBias
     encodingBias2 = encodingBias+360
     encodingBias3 = encodingBias-360
     encodingBias = torch.where(encodingBias1.abs() < 180, encodingBias1, torch.where(encodingBias2.abs() < 180, encodingBias2, encodingBias3))
  else:
     bayesianEstimate_avg_byStimulus = None
     bayesianEstimate_sd_byStimulus = None
     attraction = None
     encodingBias = None

  if float(loss) != float(loss):
      print("NAN!!!!")
      quit()
  return loss, bayesianEstimate_avg_byStimulus, bayesianEstimate_sd_byStimulus, attraction, encodingBias

def retrieveObservations(x, Subject_, Condition_, meanMethod = "circular"):
     assert Subject_ is None
     y_set = []
     sd_set = []
     for x in x_set:
        y_here = observations_y[torch.logical_and(condition==Condition_, xValues == x)]
       # y_set.append(float(y_here.mean() - grid[x]))
      #  sd_set.append(float(math.sqrt(float(y_here.pow(2).mean() - y_here.mean().pow(2)))))
     #y_set = MakeFloatTensor(y_set).cpu()
     #sd_set = MakeFloatTensor(sd_set).cpu()
     
        Mean1 = computeCircularMean(y_here)
        Mean2 = computeCenteredMean(y_here, grid[x])
        if abs(Mean1-grid[x]) > 180 and Mean1 > 180:
            Mean1 = Mean1-360
        elif abs(Mean1-grid[x]) > 180 and Mean1 < 180:
            Mean1 = Mean1+360
        if Condition_ > 1 and abs(Mean1-Mean2) > 180:
            print(Condition_)
            print(y_here)
            print("Warning: circular and centered means are very different", Mean1, Mean2, grid[x], y_here)
        if meanMethod == "circular":
            Mean = Mean1
        elif meanMethod == "centered":
            Mean = Mean2
        else:
            assert False

        bias = Mean - grid[x]
        if abs(bias) > 180:
            bias = bias+360
        y_set.append(bias)
        sd_set.append(computeCircularSD(y_here))
     return y_set, sd_set

trigonometric_basis = torch.stack([SQUARED_STIMULUS_SIMILARITY(i*grid) for i in range(1, FOURIER_BASIS_SIZE+1)] + [SQUARED_STIMULUS_DIFFERENCE(i*grid) for i in range(1, FOURIER_BASIS_SIZE+1)], dim=0)
trigonometric_basis = trigonometric_basis * (GRID / (2*trigonometric_basis.pow(2).sum(dim=1, keepdim=True)))
fourierMultiplier = MakeFloatTensor(list(range(1,FOURIER_BASIS_SIZE+1)) + list(range(1,FOURIER_BASIS_SIZE+1)))

#print(fourierMultiplier)
#print(trigonometric_basis.pow(2).sum(dim=1))
if False:
  figure, axis = plt.subplots(2*FOURIER_BASIS_SIZE, 1, figsize=(50, 50))
  for i in range(trigonometric_basis.size()[0]):
     axis[i].plot(grid.cpu(), trigonometric_basis[i].cpu())
  savePlot(f"figures/{__file__}_BASIS_{GRID}_{FOURIER_BASIS_SIZE}.pdf")
  # for debugging savePlot(f"C:\\Users\\Tate\\Documents\\Experiments\\unifying-theory-biases-main\\code\\Reisenegger\\figures\\RunReisenegger_FreePrior_DifFreeResources_byCentralOrientation.py_BASIS_{GRID}_{FOURIER_BASIS_SIZE}.pdf")
  plt.close()

def model(grid):
  lossesBy500 = []
  crossLossesBy500 = []
  noImprovement = 0
  global optim, learning_rate
  lossAverageOver500 = 0
  crossValidLossAverageOver500 = 0
  ELBOAverageOver500 = 0
  for iteration in range(10000000):
   parameters = init_parameters
   ## In each iteration, recompute
   ## - the resource allocation (called `volume' due to a geometric interpretation)
   ## - the prior

   #volume = 2 * math.pi * torch.nn.functional.softmax(parameters["volume"], dim=0)
   #prior = torch.nn.functional.softmax(parameters["prior"], dim=0)
   loss = 0
   grid_cpu = grid.cpu()
   if iteration % 1000 == 0:
     figure, axis = plt.subplots(N_CO+1, 8, figsize=(20,20))
     #axis[0,0].scatter(grid_cpu, volume.detach().cpu())
     #axis[0,0].plot([grid_cpu[0], grid_cpu[-1]], [0,0])
     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   ## Separate train and test/heldout partitions of the data
   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

  # priorByCORandomAdjustment = torch.normal(MakeZeros(N_CO,2*FOURIER_BASIS_SIZE), MakeZeros(N_CO,2*FOURIER_BASIS_SIZE)+1) * (init_parameters["SIGMA_priorByCO"]+.1)

 #  priorByCOFourierTimesWeightInclMean = priorByCORandomAdjustment # + init_parameters["priorByCO"]

#   priorByCO = torch.matmul(priorByCOFourierTimesWeightInclMean / fourierMultiplier.unsqueeze(0), trigonometric_basis)

   logQPrior = 0 # * (-(priorByCORandomAdjustment / (init_parameters["SIGMA_priorByCO"]+.1) ).pow(2)/2 - (init_parameters["SIGMA_priorByCO"]+.1).log())

  # prior_logits_effect_weight = parameters["prior_logits_effect_weight"]+1
 #  log_prior_logits_effect_weight = torch.log(parameters["prior_logits_effect_weight"]+1)
#   regularizer4 = prior_logits_effect_weight * (priorByCOFourierTimesWeightInclMean).pow(2).sum() - 0.5 * (N_CO*2*FOURIER_BASIS_SIZE) * log_prior_logits_effect_weight

   ## Iterate over the conditions and possibly subjects, if parameters are fitted separately.
   ## In this dataset, all parameters are fitted across subjects.
   for CONDITION in [1]:
    for CO in range(N_CO):
     volume = SENSORY_SPACE_VOLUME * torch.nn.functional.softmax(parameters["volume"][CO,CONDITION],dim=0).detach()
     prior = torch.nn.functional.softmax(0*parameters["prior"] + init_parameters["priorByCO"][CO,CONDITION], dim=0).detach()
     ## Run the model at its current parameter values.
     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction, encodingBias = computeBias(xValues, init_parameters["sigma_logit"][CO,CONDITION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%500 == 0), condition_=CONDITION, folds=trainFolds, lossReduce='sum', centralOrientation=CO)
     loss += loss_model

     if iteration % 1000 == 0:
       ## Visualization
       y_here = observations_y[torch.logical_and(centralOrientationPerTrial == COs[CO], condition==CONDITION)]
       x_here = observations_x[torch.logical_and(centralOrientationPerTrial == COs[CO], condition==CONDITION)]

       MASK = (SQUARED_STIMULUS_SIMILARITY(COs[CO]+180-grid) > .5)

       volumeExpected = (2-SQUARED_STIMULUS_DIFFERENCE(2*grid).abs())
       volumeExpected = SENSORY_SPACE_VOLUME * volumeExpected / volumeExpected.sum()

       def wrap180(x):
          return x
#          return ((x + 180) % 360) - 180

       grid_centered = wrap180(grid)
       print(grid_centered)
       x_here_centered = wrap180(x_here)
       #y_here_centered = wrap180(y_here)
       CO_centered = wrap180(COs[CO])

       axis[CO,0].plot(grid_centered[0:179].cpu(), volumeExpected[0:179].cpu())
       axis[CO,0].plot(grid_centered[180:].cpu(), volumeExpected[180:].cpu())
       #axis[CO,0].scatter(grid_centered[MASK].cpu(), volume[MASK].detach().cpu())
       axis[CO,0].scatter(grid_centered.cpu(), volume.detach().cpu())
       #axis[CO,0].plot([grid_cpu[0], grid_cpu[-1]], [0,0])

       #priorExpected1 = torch.softmax(15*SQUARED_STIMULUS_SIMILARITY(grid-COs[CO]-32),dim=0)
       #priorExpected2 = torch.softmax(15*SQUARED_STIMULUS_SIMILARITY(grid-COs[CO]+32),dim=0)
       #priorExpected = .2*(priorExpected1 + priorExpected2)
       priorExpected = torch.softmax(3*SQUARED_STIMULUS_SIMILARITY(grid-COs[CO]-180),dim=0)

       #axis[CO,1].plot(grid_centered[MASK].cpu(), priorExpected[MASK].detach().cpu())
       #axis[CO,1].scatter(grid_centered[MASK].cpu(), prior[MASK].detach().cpu())
#       axis[CO,1].plot(grid_centered[0:179].cpu(), priorExpected[0:179].detach().cpu())
 #      axis[CO,1].plot(grid_centered[180:].cpu(), priorExpected[180:].detach().cpu())
       axis[CO,1].plot(grid_centered.cpu(), priorExpected.detach().cpu())
       axis[CO,1].scatter(grid_centered.cpu(), prior.detach().cpu())
       axis[CO,1].plot([grid_centered[0].cpu(), grid_centered[-1].cpu()], [0,0])
       #axis[CO,2].scatter(grid_centered.cpu(), (bayesianEstimate_model-grid).detach().cpu())
       axis[CO,2].scatter(grid_centered[MASK].cpu(), (bayesianEstimate_model-grid)[MASK].detach().cpu())
       axis[N_CO,2+CONDITION].scatter(grid_centered[MASK].cpu(), (bayesianEstimate_model-grid)[MASK].detach().cpu())

       axis[3,0].scatter(grid.cpu(), torch.softmax(init_parameters["f_t"], dim=0).detach().cpu())
#       axis[3,0].
       print("f_t for plot", torch.softmax(init_parameters["f_t"], dim=0))
       #if iteration > 1:
         #quit()
       #axis[CO,3].scatter(grid_centered.cpu(), (bayesianEstimate_sd_byStimulus_model).detach().cpu())
       axis[CO,3].scatter(grid_centered[MASK].cpu(), (bayesianEstimate_sd_byStimulus_model)[MASK].detach().cpu())
       #axis[CO,3].plot([grid_centered[0].cpu(), grid_centered[-1].cpu()], [0,0])
#       axis[CO,4].scatter(grid_centered.cpu(), (attraction).detach().cpu())
       axis[CO,4].scatter(grid_centered[MASK].cpu(), (attraction[MASK]).detach().cpu())
       _, bayesianEstimate_2_4_repulsion, _, _, _ = computeBias(xValues, init_parameters["sigma_logit"][CO,CONDITION], 1/GRID+MakeZeros(GRID), volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%500 == 0), sigma_stimulus=0, sigma2_stimulus=0, condition_=CONDITION, folds=trainFolds, lossReduce='sum', centralOrientation=CO)
       #axis[CO,5].scatter(grid_centered.cpu(), (bayesianEstimate_2_4_repulsion-grid).detach().cpu())
       axis[CO,5].scatter(grid_centered[MASK].cpu(), (bayesianEstimate_2_4_repulsion-grid)[MASK].detach().cpu())

       kappa = 15
       kernel = torch.exp(kappa*SQUARED_STIMULUS_SIMILARITY(x_here.view(-1,1)-grid.view(1,-1))) / (2*math.pi*np.i0(kappa))
       kernel = kernel / kernel.sum(dim=0).max()

       bias = y_here - x_here
       bias = torch.where(bias > 180, bias-360, torch.where(bias < -180, bias+360, bias))
       y_smoothed = computeCircularMeanWeighted(bias.unsqueeze(1), kernel)
       y_smoothed = torch.where(y_smoothed > 180, y_smoothed-360, torch.where(y_smoothed < -180, y_smoothed+360, y_smoothed))



       axis[CO][6+CONDITION].scatter(grid_centered.cpu()[MASK], y_smoothed.cpu()[MASK])
       axis[CO][6+CONDITION].scatter(x_here_centered.cpu(), bias.cpu(), s=0.1, alpha=0.2)
       for w in [2,4,5,6,7]:
          axis[CO][w].set_ylim(-80, 80)
       axis[N_CO][6+CONDITION].scatter(grid_centered.cpu()[MASK], y_smoothed.cpu()[MASK])
       axis[N_CO][6+CONDITION].scatter(x_here_centered.cpu(), bias.cpu(), s=0.1, alpha=0.2)
       for w in [2,3,4,5,6,7]:
          axis[N_CO][w].set_ylim(-80, 80)

       bound1, bound2 = COs[CO]-60, COs[CO]+60
       for w in range(8):
         bound1, bound2 = CO_centered - 60 + 180, CO_centered + 60 + 180
         axis[CO,w].plot([bound1, bound2], [0,0])
         axis[CO,w].scatter([CO_centered.cpu()], [0], color="purple", s=10)
         axis[CO,w].set_xlim(0,360)
       axis[3,0].set_ylim(0, 0.01)
   if iteration % 1000 == 0:

     axis[0,0].set_title("Resources")
     axis[0,1].set_title("Prior")
     axis[0,2].set_title("Bias")
     axis[0,3].set_title("Variability")
     axis[0,4].set_title("Attraction")
     axis[0,5].set_title("Repulsion")
     axis[0,6].set_title("Bias (0)")
     axis[0,7].set_title("Bias (1)")

     print("Saving plot...")

     savePlot(f"figures/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}_{FOURIER_BASIS_SIZE}.pdf")
     plt.close()

     crossValidLoss = 0
     regularizer_total = MakeZeros(2)  # Initialize regularizer_total for two conditions
     for CONDITION in range(2):
      if not torch.any(condition == CONDITION):
          continue
      for CO in range(N_CO):
       volume = SENSORY_SPACE_VOLUME * torch.nn.functional.softmax(parameters["volume"][CO,CONDITION], dim=0).detach()
       prior = torch.nn.functional.softmax(0*parameters["prior"] + init_parameters["priorByCO"][CO,CONDITION], dim=0).detach()
       #shift_by = GRID-int((grid-COs[CO]).abs().argmin())
       print(f"COs[CO]: {COs[CO]}")
       #prior = torch.cat([prior_overall[shift_by:], prior_overall[:shift_by]], dim=0)

       loss_2_4, bayesianEstimate_2_4, bayesianEstimate_sd_byStimulus_2_4, attraction, _ = computeBias(xValues, init_parameters["sigma_logit"][CO,CONDITION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), sigma_stimulus=0, sigma2_stimulus=0, condition_=CONDITION, folds=testFolds, lossReduce='sum', centralOrientation=CO)
       crossValidLoss += loss_2_4

#   regularizer1 = ((init_parameters["volume"][:,:,1:] - init_parameters["volume"][:,:,:-1]).pow(2).sum() + (init_parameters["volume"][:,:,0] - init_parameters["volume"][:,:,-1]).pow(2).sum())/GRID
#   priorLogits = init_parameters["prior"].unsqueeze(0)
#   regularizer2 = ((priorLogits[:,1:] - priorLogits[:,:-1]).pow(2).sum() + (priorLogits[:,0] - priorLogits[:,-1]).pow(2).sum())/GRID
#   regularizer3 = ((init_parameters["priorByCO"][:,:,1:] - init_parameters["priorByCO"][:,:,:-1]).pow(2).sum() + (init_parameters["priorByCO"][:,:,0] - init_parameters["priorByCO"][:,:,-1]).pow(2).sum())/GRID
   regularizer4 = ((init_parameters["f_t"][1:] - init_parameters["f_t"][:-1]).pow(2).sum() + (init_parameters["f_t"][0] - init_parameters["f_t"][-1]).pow(2).sum())/GRID
 #  regularizer_total = regularizer1 + regularizer2 + regularizer3 + 
   regularizer_total = regularizer4



   ELBO = loss # + regularizer4 + torch.nansum(logQPrior) #logQVolume.sum() + + regularizer3
   print(f"Iteration: {iteration}, loss: {loss}") #, regularizer4: {regularizer4}, logQPrior: {torch.nansum(logQPrior)}") #, logQVolume.sum(), regularizer3,
   loss = ELBO

   loss = loss * (1/observations_y.size()[0])
   loss = loss + REG_WEIGHT * regularizer_total.sum()

   optim.zero_grad()
   loss.backward() #retain_graph=True)

   maximumGradNorm = []
   largestGradNorm = 0
   ## For monitoring purposes, calculate the size of the gradients
   for w in init_parameters:
     if init_parameters[w].grad is not None:
      maximumGradNorm.append(w)
      gradNormMax = float(init_parameters[w].grad.abs().max())
      maximumGradNorm.append(gradNormMax)
      largestGradNorm = max(largestGradNorm, float(gradNormMax))
      ## Optionally, in order to use SignGD, can now replace each
      ## gradient by an indicator of its sign.
      init_parameters[w].grad.data = torch.sign(init_parameters[w].grad.data)
   if iteration % 10 == 0:
     print(largestGradNorm, maximumGradNorm)


  #  print(init_parameters)
   #torch.nn.utils.clip_grad_norm_(init_parameters["sigma_logit"], max_norm=0.1, norm_type='inf') #clamp magnitude of the parameter updates   
   optim.step()
   if iteration % 10 == 0:
     print(lossesBy500, noImprovement)
     print(crossLossesBy500)
     print(iteration, loss, init_parameters["sigma_logit"], torch.sigmoid(init_parameters["mixture_logit"]), init_parameters["log_motor_var"])
   ELBOAverageOver500 = ELBOAverageOver500 + float(ELBO)/500
   lossAverageOver500 = lossAverageOver500 + float(loss)/500
   crossValidLossAverageOver500 = crossValidLossAverageOver500 + float(crossValidLoss)/500

   #if iteration % 500 == 0:
   #  crossValidLoss = 0
   #  for condition_ in [0,1]:
   #     volume = 2 * math.pi * torch.nn.functional.softmax(parameters["volume"][condition_], dim=0)
   #     loss_, bayesianEstimate, bayesianEstimate_sd, attraction, encodingBias = computeBias(xValues, init_parameters["sigma_logit"][condition_], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, computePredictions=(iteration % 500 == 0), parameters=parameters, condition_=condition_, folds=testFolds, lossReduce='sum')
   #     crossValidLoss += loss_

#   print(lossesBy500)
   if iteration % 500 == 0 and iteration > 0:
       lossesBy500.append(float(loss))
       crossLossesBy500.append(float(crossValidLoss))
       #with open(f"C:\\Users\\Tate\\Documents\\unifying-theory-biases-main\\code\\Reisenegger\\losses\\RunReisenegger_FreePrior_DifFreeResources_CosineLoss.py_0-0_9_10.0_360.txt", "w") as outFile: # for debugging
       if crossLossesBy500[-1] <= min(crossLossesBy500):
          with open(f"losses/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt", "w") as outFile:
              print(float(crossValidLoss), file=outFile)
          #with open(f"C:\\Users\\Tate\\Documents\\unifying-theory-biases-main\\code\\Reisenegger\\logs\\RunReisenegger_FreePrior_DifFreeResources_CosineLoss.py_0-0_9_10.0_360.txt", "w") as outFile: # for debugging
          with open(f"logs/CROSSVALID/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt", "w") as outFile:
              print(float(loss), "CrossValid", float(crossValidLoss), "CrossValidLossesBy500", " ".join([str(q) for q in crossLossesBy500]), file=outFile)
              print(iteration, "LossesBy500", " ".join([str(q) for q in lossesBy500]), file=outFile)
              for z, y in init_parameters.items():
                  print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
       if len(crossLossesBy500) > 1 and crossLossesBy500[-1] >= crossLossesBy500[-2]-1e-5:
         learning_rate *= 0.8
         optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate)
       if len(crossLossesBy500) > 1 and crossLossesBy500[-1] >= min(crossLossesBy500[:-1])-1e-5:
         noImprovement += 1
       else:
         noImprovement = 0
       if noImprovement >= 5:
           print("Stopping")
           break
       ELBOAverageOver500 = 0
       lossAverageOver500 = 0
       crossValidLossAverageOver500 = 0
############################3

# Project the stimuli onto the discrete grid

for P1 in [P]: #, 2, 4, 6, 8, 10]:

  ##############################################
# Initialize the model
# Part: Initialize the model
  init_parameters = {}
  init_parameters["log_motor_var"] = MakeZeros(N_CO, 2)
  init_parameters["sigma_logit"] = -1 + MakeZeros(N_CO, 2)
  init_parameters["mixture_logit"] = MakeFloatTensor(N_CO*[-1])
  init_parameters["f_t"] = MakeZeros(GRID)
  init_parameters["sigma2_t"] = MakeZeros(1)
  init_parameters["sigma2_t2"] = MakeZeros(1)
  init_parameters["prior"] = MakeZeros(GRID)
  init_parameters["volume"] = MakeZeros(N_CO,2,GRID) #Different for each condition
  init_parameters["priorByCO"] = MakeZeros(N_CO,2,GRID)
 # for CO in range(3):
#     init_parameters["priorByCO"][CO] = 0.5*(3*SQUARED_STIMULUS_SIMILARITY(grid-COs[CO]-180))

  init_parameters["prior_logits_effect_weight"] = MakeZeros(1)
  #init_parameters["volumeBySubject"] = MakeZeros(2,N_SUBJECTS,2*FOURIER_BASIS_SIZE) #Different for each condition
  #init_parameters["volume_logits_effect_weight"] = MakeZeros(1)

#  init_parameters["SIGMA_priorByCO"] = MakeZeros(N_CO,2*FOURIER_BASIS_SIZE)
  #init_parameters["SIGMA_volumeBySubject"] = MakeZeros(2,N_SUBJECTS,2*FOURIER_BASIS_SIZE) #Different for each condition

  from util import loadParameters
  loadParameters(init_parameters, f"logs/CROSSVALID/RunReisenegger_FreePrior_DifFreeResources_byCentralOrientation_Simplified_Shift_Co0_Prior_SepEnc_Debug_L1.py_1_0_{REG_WEIGHT}_180.txt")
  init_parameters["volume"][:,1] = init_parameters["volume"][:,0]
  init_parameters["priorByCO"][:,1] = init_parameters["priorByCO"][:,0]
 # print(init_parameters["volume"])
#  quit()

  for _, y in init_parameters.items():
    y.requires_grad = True

# Initialize optimizer.
# The learning rate is a user-specified parameter.
  learning_rate = 0.1
  optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate)

  ##############################################
  #assert P >= 2
# Import the appropriate estimator for minimizing the loss function
  SCALE = 50

# Run the model
  #For condition 0
  #if P == 0:
   # KERNEL_WIDTH = 0.05
    #MAPCircularEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, KERNEL_WIDTH=KERNEL_WIDTH, SCALE=SCALE, MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID)
  
  #elif P>0:
  #CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY, SCALE=SCALE)
#LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY, SCALE=SCALE)
   
  #For condition 1
  #if P1 == 0:
  #  KERNEL_WIDTH = 0.05
  #  MAPCircularEstimator1.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, KERNEL_WIDTH=KERNEL_WIDTH, SCALE=SCALE, MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID)
  
  #elif P1>0:
 # CosineEstimator1.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P1=P1, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY, SCALE=SCALE)
  
  model(grid)
