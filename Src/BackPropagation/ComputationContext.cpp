#include "ComputationContext.h"
#include "DebugLib.h"

using namespace std;

/**
  Constructor for BackPropagator class.

  @param  FCN   A reference to a FullyConnectedNetwork object to be associated with this BackPropagator.

**/
ComputationContext::ComputationContext (
  vector<unsigned int>  Layout
  )
{
  InitActivation (Layout);

  InitNodeDelta (Layout);

  Loss = 0.0;
}

/**
  Destructor for ComputationContext class.

**/
ComputationContext::~ComputationContext ()
{
  //
  // Nothing to do here for now.
  //
}

/**
  Initialize node values of each layer to zero.

  @param[in]  Layout  The layout of the network is using.

**/
void
ComputationContext::InitActivation (
  const vector<unsigned int>  &Layout
  )
{
  if (Layout.size() < 2) {
    throw invalid_argument ("Layout of a network should at least have 2 layers.");
  }

  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerNodes(Layout[Index], 1);
    Activation.push_back(LayerNodes);
  }
}

/**
  Get the activation matrix of a specific layer.

  @param  Layer  An unsigned integer representing the layer index.

  @return A matrix representing the activation values of the specified layer.

  @throw std::runtime_error if the Layer index is out of range.

**/
matrix
ComputationContext::GetActivationByLayer (
  unsigned int  Layer
  ) const
{
  if (Layer >= Activation.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Activation.size());
    throw std::runtime_error("Error: Layer index out of range in GetActivationByLayer().");
  }

  return Activation[Layer];
}

/**

**/
void
ComputationContext::SetActivationByLayer (
  unsigned int Layer,
  matrix       ActivationOfLayer
  )
{
  if (Layer >= (unsigned int)Activation.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Activation.size());
    throw std::runtime_error("Error: Layer index out of range in SetActivationByLayer().");
  }

  Activation[Layer] = ActivationOfLayer;
}

/**
  Print the activation values of a specific layer.

  @param  Layer   An unsigned integer representing the layer index.

  @throw std::runtime_error if the Layer index is out of range.

**/
void
ComputationContext::PrintActivationInLayer (
  unsigned int  Layer
  )
{
  if (Layer >= (unsigned int)Activation.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Activation.size());
    throw std::runtime_error("Error: Layer index out of range in PrintNodesInLayer().");
  }

  cout << "Nodes in layer" << Layer << ":" << endl;
  Activation[Layer].show();
}

/**
  Initialize delta values of each layer to zero.

**/
void
ComputationContext::InitNodeDelta (
  vector<unsigned int>  &Layout
  )
{
  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerDelta (Layout[Index], 1);
    NodeDelta.push_back (LayerDelta);
  }
}

/**
  Get the node delta matrix of a specific layer.

  @param  Layer  An unsigned integer representing the layer index.

  @return A matrix representing the node delta values of the specified layer.

  @throw std::runtime_error if the Layer index is out of range.

**/
matrix
ComputationContext::GetNodeDeltaByLayer (
  unsigned int  Layer
  ) const
{
  if (Layer >= NodeDelta.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << NodeDelta.size());
    throw std::runtime_error("Error: Layer index out of range in GetNodeDeltaByLayer().");
  }

  return NodeDelta[Layer];
}

/**
  Set the delta value of a specific node in a specific layer.

  @param  Layer   An unsigned integer representing the layer index.
  @param  Number  An unsigned integer representing the node index within the layer.
  @param  Delta   A double representing the delta value to be set for the specified node.

  @throw std::runtime_error if the Layer or Number index is out of range.

**/
void ComputationContext::SetNodeDeltaByLayer (
  unsigned int Layer,
  matrix       NodeDeltaOfLayer
  )
{
  if (Layer >= (unsigned int)NodeDelta.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << NodeDelta.size());
    throw std::runtime_error("Error: Layer index out of range in SetNodeDelta().");
  }

  NodeDelta[Layer] = NodeDeltaOfLayer;
}

/**
  [**Only for internal debug use**]
  Print the delta values of all nodes in each layer.

**/
void
ComputationContext::PrintNodeDelta(
  void
  )
{
  for(int Index = 0; Index < (int)NodeDelta.size(); Index++) {
    cout << "Delta in layer " << Index<< ":" <<endl;
    NodeDelta[Index].show();
  }
}