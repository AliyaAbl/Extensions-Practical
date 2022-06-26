import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sklearn.cluster
import sklearn.decomposition
import torch.nn as nn
import torch
from torch import optim
from torch import relu as relu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def make_GMM(dim, N, var,plot,mu_r_1=None,mu_r_2=None, linsep=False): 
  '''
  This Function generates the gaussian mixtrue models. Set plot = True to inspect the first four dimensions visually. 
  input:  dim     = dimension D
          N       = number of samples 
          var     = standard deviation (sigma) for all clusters 
          mu_r_1  = scaling facor for the distance of the cluster centers to the origin, default=None
          mu_r_2  = scaling facor for the distance of the cluster centers to the origin, default=None
  output: X       = data points of shape [N, dim]
          Y       = labels of shape [N]
          mus     = means of the 4 GMs of shape [4, dim]
  '''
  if mu_r_1==None:
    mu_r_1 = math.sqrt(dim)
  if mu_r_2 == None:
    mu_r_2 = math.sqrt(dim)

  
  # Cluster means of the 4 GMs in the first two dimensions. 
  # If mu_r is set to none, then the cluster centers will be (0,±1) and (±1, 0).

  mu1 = [0,                         mu_r_2/math.sqrt(dim)]
  mu2 = [0,                         (-1)*mu_r_2/math.sqrt(dim)]
  mu3 = [mu_r_1/math.sqrt(dim),     0]
  mu4 = [(-1)*mu_r_1/math.sqrt(dim),0]

  # Cluster means of the 4 GMs for the other D - 2 dimensions set to zero.
  if dim>2:
    mu1 = np.append(mu1, np.zeros((dim-2), dtype=int))
    mu2 = np.append(mu2, np.zeros((dim-2), dtype=int))
    mu3 = np.append(mu3, np.zeros((dim-2), dtype=int))
    mu4 = np.append(mu4, np.zeros((dim-2), dtype=int))

  # Shared diagonal coariance matrix.
  
  cov = np.eye(dim)* (var**2) 

  # Sampled datapoints from the 4 multivariate gaussians.

  cluster1 = np.random.multivariate_normal(mu1, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster2 = np.random.multivariate_normal(mu2, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster3 = np.random.multivariate_normal(mu3, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster4 = np.random.multivariate_normal(mu4, cov, size = int(N/4) , check_valid='warn', tol=1e-8)

  if linsep == False:
    # Labels for the 4 GMs according to the 2 clusters of an XOR distribution. 
    label1 = np.ones(int(N/4), dtype=int)*(-1)
    label2 = np.ones(int(N/4), dtype=int)*(1)
    label3 = np.ones(int(N/4), dtype=int)*(-1)
    label4 = np.ones(int(N/4), dtype=int)*(1)

  if linsep == True:
    # Labels for the 4 GMs according to the 2 clusters of an XOR distribution. 
    label1 = np.ones(int(N/4), dtype=int)*(-1)
    label2 = np.ones(int(N/4), dtype=int)*(-1)
    label3 = np.ones(int(N/4), dtype=int)*(1)
    label4 = np.ones(int(N/4), dtype=int)*(1)

  if plot==True:
    
    # This part visualizes the first four dimensions of the data. 
    
    plt.scatter(cluster1[:,0],cluster1[:,1] , color='red')
    plt.scatter(cluster2[:,0],cluster2[:,1] , color='blue')
    plt.scatter(cluster3[:,0],cluster3[:,1] , color='red')
    plt.scatter(cluster4[:,0],cluster4[:,1] , color='blue')
    plt.title('Input Space')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.gca().set_xticks([])
    plt.xticks([])
    plt.gca().set_yticks([])
    plt.yticks([])
    plt.xlim([-6,6])
    plt.ylim([-6,6])
    plt.show()

  return np.vstack((cluster1, cluster2, cluster3, cluster4)), np.hstack((label1, label2, label3, label4)) , np.vstack((mu1, mu2, mu3, mu4))

def oracle(X, mu, linsep = False):
  """
  This function implements the 'oracle' which is defined as a network "with knowledge of the means of 
  the mixture that assigns to each input the label of the nearest mean". 
  
  Input:  X       = data points of shape [N, dim]
          mu      = means of the 4 GMs of shape [4, dim]
  Output: labels  = assigned cluster to each datapoints of shape [N]
  """ 
  oracle = sklearn.cluster.KMeans(n_clusters=4, init=mu, n_init=1).fit(X)
  labels = oracle.labels_
  ind1 = np.where(labels==0)[0]
  ind2 = np.where(labels==1)[0]
  ind3 = np.where(labels==2)[0]
  ind4 = np.where(labels==3)[0]

  cluster1 = np.hstack((ind1, ind2))
  cluster2 = np.hstack((ind3, ind4))

  if linsep == True:
    labels[labels==0] = -1
    labels[labels==1] = -1
    labels[labels==2] = 1
    labels[labels==3] = 1
  
  if linsep == False:
    labels[labels==0] = -1
    labels[labels==1] = 1
    labels[labels==2] = -1
    labels[labels==3] = 1

  return labels 

def make_splits(X, Y):
  '''
  input:  X       = data points of shape [N, dim]
          Y       = labels of shape      [N]
  output: X_train = 2/3 of the datapoints used for trainig of shape     [2N/3, dim]
          X_val   = 1/3 of the datapoints used for validation of shape  [N/3, dim]
          Y_train = 2/3 of the labels used for trainig of shape         [2N/3]
          Y_val   = 1/3 of the labels used for validation of shape      [N/3]
  '''
  N = np.shape(Y)[0]
  indices = np.arange(N)
  np.random.shuffle(indices)

  X = X[indices]
  Y = Y[indices]
  X_train = X[0:int(N*0.66),:]
  X_val   = X[int(N*0.66):,:]
  Y_train = Y[0:int(N*0.66)]
  Y_val   = Y[int(N*0.66):]

  return torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(Y_train), torch.from_numpy(Y_val)

def plot_input_feature_spaces(dim, sigma, mu_r_1, mu_r_2):
  N   = 500
  X, Y, m     = make_GMM(dim , N , var = sigma, plot=True, mu_r_1=mu_r_1, mu_r_2=mu_r_2)
  F           = torch.randn((dim,dim*10))
  X_trafo     = transform_RF(X=X,F=F)
  fig         = plt.figure(figsize = (10, 7))
  plt.rcParams.update({'font.size': 16})
  ax          = fig.add_subplot(projection='3d')
  ax.scatter(X_trafo[:int(N/2),0],X_trafo[:int(N/2),1],X_trafo[:int(N/2),2],color='red')
  ax.scatter(X_trafo[int(N/2):int(N),0],X_trafo[int(N/2):int(N),1],X_trafo[int(N/2):int(N),2],color='blue')
  frame1      = plt.gca()
  frame1.axes.xaxis.set_ticklabels([])
  frame1.axes.yaxis.set_ticklabels([])
  frame1.axes.zaxis.set_ticklabels([])
  ax.set_xlabel('z1')
  ax.set_ylabel('z2')
  ax.set_zlabel('z3')
  plt.title('Feature Space of RF')
  plt.show()

# NN
class Student(nn.Module):
  """
  This is the 2-layerd neuronal network with K hidden neurons and 1 output neuron, used thoughtout this report. 
  """
  def __init__(self,K,N,weight_std_initial_layer=1):
      """ 
      Input:  K                         = number of hidden neurons
              N                         = number of samples 
              weight_std_initial_layer  = standard deviation for the weight initialization of the first
      """
      print("Creating a Student with InputDimension: %d, K: %d"%(N,K) )
      super(Student, self).__init__()
      
      self.N=int(N)
      self.g=nn.ReLU()
      self.K=int(K)
      self.loss=nn.MSELoss(reduction='mean')
      # Definition of the 2 layers 
      self.fc1 = nn.Linear(int(N), K)
      self.fc2 = nn.Linear(K, 1)

      ##For Figure 1 reproduction   
      #torch.nn.init.xavier_uniform_(self.fc2.weight)
      #torch.nn.init.xavier_uniform_(self.fc1.weight)
      nn.init.normal_(self.fc1.weight)
      nn.init.normal_(self.fc2.weight)

      ##For figure 4 reproduction
      #nn.init.normal_(self.fc1.weight,std=weight_std_initial_layer)
      #nn.init.normal_(self.fc2.weight,std=weight_std_initial_layer)


  def forward(self, x):
      # This is the input to the hidden layer. 
      x=self.fc1(x) /math.sqrt(self.N)
      x=self.g(x)
      x = self.fc2(x)
      return x

def HalfMSE(output, target): 
    loss = (0.5)*torch.mean((output - target)**2)
    return loss

def linear(x):
  return x

def centered_relu(x,var):
    a = math.sqrt(var)/math.sqrt(2*math.pi)
    return torch.relu(x)-a

# RF
def transform_RF(X,F):
    """
    This function tansforms the datapoints X into a feature space of P>>dim, with the 
    transform-matrix F. 
    Input:  X       = data points of shape [N, dim]
            F       = transformation matrix of shape [dim, P]
    Output: X_trafo = transformed datapoints in the feature space of shape [N, P]
    """
    D, P = F.shape
    X    = torch.from_numpy(X)
    X    = X.float()
    F   /= F.norm(dim=0).repeat(D, 1)
    F   *= math.sqrt(D)
    X_trafo = centered_relu((X@F) / math.sqrt(D),0)
    return X_trafo

class Student_RF(nn.Module):
  """
  This is the second layer for the Random Features, which takes the projected datapoints 
  and predcits the cluster labels via a linear model. 
  """
  def __init__(self,K,N,bias=False):
    """ 
    Input:  K                         = number of hidden neurons
            N                         = number of input dimensions
    """
    print("Creating a Student with InputDimension: %d, K: %d"%(N,K) )
    super(Student_RF, self).__init__()
    
    self.P= int(N)
    self.g=linear
    self.K=1
    self.loss=nn.MSELoss(reduction='mean')
    self.fc1 = nn.Linear(self.P, int(K))
    nn.init.normal_(self.fc1.weight,std=0.01)

  def forward(self, x):
    x = self.g(self.fc1(x)/math.sqrt(self.P))
    return x

# PCA
def PCA(X, Y):
    # X     [N, dim]
    # U     [4, dim]
    # X_pca [N, 4]
    # Y_pca [N, 1]
    pca = sklearn.decomposition.PCA(n_components=4).fit(X)
    U   = pca.components_
    X_pca = np.matmul(X, U.T)
    Y_pca = Y
    return X_pca, Y_pca

"""Figure 6 iterating over dim"""

######### Defining Parameters ########################################
######################################################################
K           = 10
std_weights = 1e-2
sigma       = math.sqrt(0.05)
N           = 50000
num_t       = 10*3
t1          = np.logspace(1, 1.5, num= int(num_t/3))
t2          = np.logspace(1.5,2,  num= int(num_t/3))
t3          = np.logspace(2.5 , 3,  num= int(num_t/3))
t           = np.round(np.append(t1, np.append(t2, t3)), 5)
lr          = 0.1
reg         = 0.0
dim         = N/np.floor(t)       
P           = 2*dim

def iterate_over_time(N, P, K, dim, lr, reg, sigma, SNR):


  ######### iterate over time (t = N/D) ################################
  ######################################################################
  for i in range(0, len(dim)):

    ######### initilize the NN layers for RF, NT, NT KRR and 2LNN ########
    ######################################################################
    student_rf    = Student_RF(N = P[i], K =1)
    params_RF     = []
    params_RF    += [{'params': student_rf.fc1.parameters(),'lr': lr,'weight_decay':reg}]
    optimizer_RF  = optim.SGD(params_RF, lr=lr, weight_decay=reg)
    criterion_RF  = student_rf.loss 

    student_2lnn  = Student(K = K, N = dim[i])
    params_2LNN   = []
    params_2LNN  += [{'params': student_2lnn.fc1.parameters()}]
    params_2LNN  += [{'params': student_2lnn.fc2.parameters(),'lr': lr,'weight_decay':reg}]
    optimizer_2LNN = optim.SGD(params_2LNN, lr=lr, weight_decay=reg)
    criterion_2LNN = student_2lnn.loss 

    student_pca   = Student_RF(N = 4, K =1)
    params_PCA    = []
    params_PCA   += [{'params': student_pca.fc1.parameters(),'lr': lr,'weight_decay':reg}]
    optimizer_PCA = optim.SGD(params_PCA, lr=lr, weight_decay=reg)
    criterion_PCA = student_rf.loss 

    mse_RF   = np.zeros(N)
    mse_2LNN = np.zeros(N)
    mse_PCA  = np.zeros(N)
    print("For length N = {}".format(N))


    if SNR == "low":
      mu_r_1 = math.sqrt(dim[i])*1
      mu_r_2 = math.sqrt(dim[i])*1
    if SNR == "high":
      mu_r_1 = math.sqrt(dim[i])*5
      mu_r_2 = math.sqrt(dim[i])*5
    if SNR == "mixed":
      mu_r_1 = math.sqrt(dim[i])*2
      mu_r_2 = dim[i]/20

    print("Run number:{}".format(i))
    X, Y, m  = make_GMM(dim = int(dim[i]), N  = N, var = sigma, plot=False, mu_r_1=mu_r_1, mu_r_2=mu_r_2, linsep = True)

    # RF
    F     = torch.randn((int(dim[i]), int(P[i]))) 
    X_RF  = transform_RF(X, F)
    X_RF  = (X_RF).numpy()
    X_train_RF, X_val_RF, Y_train_RF, Y_val_RF         = make_splits(X_RF, Y)
    
    X_val_RF = (X_val_RF).float()
    Y_val_RF = (Y_val_RF).float()

    X_train_2LNN, X_val_2LNN, Y_train_2LNN, Y_val_2LNN = make_splits(X, Y)
    X_val_2LNN = (X_val_2LNN).float()
    Y_val_2LNN = (Y_val_2LNN).float()

    X_PCA, Y_PCA        = PCA(X,Y)
    X_train_PCA, X_val_PCA, Y_train_PCA, Y_val_PCA     = make_splits(X_PCA, Y_PCA)
    X_val_PCA = (X_val_PCA).float()
    Y_val_PCA = (Y_val_PCA).float()


    ######### Training the 2LNN and RF with online SGD on MSE ############
    ######################################################################
    student_rf.train() 
    student_2lnn.train()
    student_pca.train()

    for j in range(X_train_RF.shape[0]): 
      targets_RF    = (Y_train_RF[j]).float() 
      inputs_RF     = (X_train_RF[j,:]).float()
      student_rf.zero_grad()
      preds_RF      = student_rf(inputs_RF)
      loss_RF       = criterion_RF(preds_RF, targets_RF)

      targets_2LNN  = (Y_train_2LNN[j]).float() 
      inputs_2LNN   = (X_train_2LNN[j,:]).float()
      student_2lnn.zero_grad()
      preds_2LNN    = student_2lnn(inputs_2LNN)
      loss_2LNN     = criterion_2LNN(preds_2LNN, targets_2LNN)

      targets_PCA    = (Y_train_PCA[j]).float() 
      inputs_PCA     = (X_train_PCA[j,:]).float()
      student_pca.zero_grad()
      preds_PCA      = student_pca(inputs_PCA)
      loss_PCA       = criterion_PCA(preds_PCA, targets_PCA)

      if j% 500 ==0: #print train loss every 100 steps
        print("Train loss RF: {}--- Train loss 2LNN:{}".format(loss_RF,loss_2LNN))
      loss_RF.backward()
      torch.nn.utils.clip_grad_norm_(student_rf.parameters(), 10.0)
      optimizer_RF.step()

      loss_2LNN.backward()
      torch.nn.utils.clip_grad_norm_(student_2lnn.parameters(), 10.0)
      optimizer_2LNN.step()

      loss_PCA.backward()
      torch.nn.utils.clip_grad_norm_(student_pca.parameters(), 10.0)
      optimizer_PCA.step()

    ######### Evaluating the 2LNN and RF with online SGD on MSE ###########
    ######################################################################
    student_rf.eval()
    student_2lnn.eval() 
    student_pca.eval()
    with torch.no_grad():
      preds_RF    = student_rf(X_val_RF)
      preds_RF    = preds_RF[:,0]
      eg_RF       = criterion_RF(preds_RF, Y_val_RF)
      mse_RF[i]   = eg_RF
      _eg_RF      = eg_RF.cpu().detach().numpy()

      preds_2LNN  = student_2lnn(X_val_2LNN)
      preds_2LNN  = preds_2LNN[:,0]
      eg_2LNN     =  criterion_2LNN(preds_2LNN, Y_val_2LNN)
      mse_2LNN[i] = eg_2LNN
      _eg_2LNN    = eg_2LNN.cpu().detach().numpy()

      preds_PCA   = student_pca(X_val_PCA)
      preds_PCA   = preds_PCA[:,0]
      eg_PCA      = criterion_PCA(preds_PCA, Y_val_PCA)
      mse_PCA[i]  = eg_PCA
      _eg_PCA     = eg_PCA.cpu().detach().numpy()

      t_          = N/dim[i]
      print("Test Data: MSE RF: {}; MSE 2LNN:{}; Test Data: MSE PCA: {}; t:{}".format(np.round(_eg_RF, 3),np.round(_eg_2LNN, 3),np.round(_eg_PCA, 3),int(t_)))
      print("---------------------------------------------------------")
  return mse_RF, mse_2LNN, mse_PCA

############# low SNR ################################################

mse_RF_200_low, mse_2LNN_200_low  = iterate_over_time(N, P, K, dim, lr, reg, sigma, SNR = "low")

############# high SNR ###############################################

mse_RF_200_high, mse_2LNN_200_high  = iterate_over_time(N, P, K, dim, lr, reg, sigma, SNR = "high")
############# mixed SNR ##############################################

mse_RF_200_mixed, mse_2LNN_200_mixed  = iterate_over_time(N, P, K, dim, lr, reg, sigma, SNR = "mixed")




