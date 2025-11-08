#include "BackPropagator.h"
#include "DebugLib.h"

using namespace std;

BackPropagator::BackPropagator (
  FullyConnectedNetwork &FCN
  ) : Network (FCN)
{
  InitNodeDelta ();
  InitDeltaWeights ();
}

/**
  Initialize delta values of each layer to zero.

**/
void
BackPropagator::InitNodeDelta ()
{
  vector<unsigned int> Layout = Network.GetLayout ();

  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerDelta (Layout[Index], 1);
    NodeDelta.push_back (LayerDelta);
  }
}

/**
  Set the delta value of a specific node in a specific layer.

  @param  Layer   An unsigned integer representing the layer index.
  @param  Number  An unsigned integer representing the node index within the layer.
  @param  Delta   A double representing the delta value to be set for the specified node.

  @throw std::runtime_error if the Layer or Number index is out of range.

**/
void BackPropagator::SetNodeDelta (
  unsigned int Layer,
  unsigned int Number,
  double       Delta
  )
{
  vector<unsigned int> Layout = Network.GetLayout ();

  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in SetNodeDelta().");
  }
  if (Number >= (unsigned int)Layout[Layer]) {
    DEBUG_LOG ("Number: " << Number << ", Nodes in layer: " << Layout[Layer]);
    throw std::runtime_error("Error: Node number index out of range in SetNodeDelta().");
  }

  NodeDelta[Layer].SetValue (Number, 0, Delta);
}

void BackPropagator::PrintNodeDelta()
{
  for(int Index = 0; Index < (int)NodeDelta.size(); Index++) {
    cout << "Delta in layer " << Index<< ":" <<endl;
    NodeDelta[Index].show();
  }
}

/**
  Initialize delta weights between each layers to zero.

**/
void BackPropagator::InitDeltaWeights ()
{
  vector<unsigned int> Layout = Network.GetLayout ();

  for(int Index = 0; Index < (int)Layout.size() - 1; Index++) {
    matrix LayerDeltaWeights (Layout[Index + 1], Layout[Index]);
    DeltaWeights.push_back (LayerDeltaWeights);
  }
}