import numpy as np

class neuroNetwork(object):
  def __init__( self, nnStruct, r=0.1 ):
    """
    nnStruct: a list
    eg.
    neuroNetwork( [2,6,1] )
    get a 2-6-1 neuro network (2 input, 1 ouput, 6 node in the middle hidden layer)
  
    r:
    the initial links weight will be random numbers in range {-r,+r}
    """
    self.struct = nnStruct
    self.nodes = []
    self.links = [None] # None is a place holder, links[i] are links from layer i-1 to i
    for layerNodes in nnStruct:
      tmp = np.zeros(layerNodes+1)
      tmp[0] = 1.   # the "constant term" node
      self.nodes.append(tmp)
    
    for layerID in range(1, len(nnStruct) ):
      self.links.append( (np.random.rand( nnStruct[layerID] , nnStruct[layerID-1]+1 )*2.-1.)*r )

  def eval( self, input ):
    """
    get the prediction form NN with the input

    input: np array of correct size
    """
    if len(input)!=self.struct[0] :
      print "Wrong input for NN:" , input
      return None

    self.nodes[0][1:] = input
    for layerID in range(1, len(self.struct)):
      self.nodes[layerID][1:] = self.links[layerID].dot(self.nodes[layerID-1])
      self.nodes[layerID][1:] = np.tanh(self.nodes[layerID][1:])

    return self.nodes[-1][1:]

  def train(self, input, answer, eta = 0.1):
    """
    train the NN
    input  : the input, an array
    answer : the correct answer for such input, can be an array
    eta    : the step size to be used in the gradient decend
    """
    output = self.eval(input)
    delta = []
    for i in self.struct:
      delta.append(np.zeros(i))

    delta[-1] = -2.*( answer - output ) * (1.-pow(output,2))
    for layerID in range(len(self.struct)-2,0,-1):
      delta[layerID] = self.links[layerID+1][:,1:].transpose().dot(delta[layerID+1]) * (1.-pow(self.nodes[layerID][1:],2))
    
    for layerID in range(1,len(self.struct)):
      self.links[layerID] -= eta * np.outer(delta[layerID], self.nodes[layerID-1] )

  
