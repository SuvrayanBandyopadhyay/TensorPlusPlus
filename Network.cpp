#include"Network.h"
using namespace TPP;
using namespace std;

void Network::addLayer(Layer &l) 
{
	
	//Add the new layer
	layers.push_back(&l);
}

Tensor Network::output(Tensor input) 
{
	for (int i = 0; i < layers.size(); i++) 
	{
		input = layers[i]->output(input);
	}
	return input;
}

//Backpropagate
void Network::backpropagate(Tensor fb) 
{
	for (int i = layers.size() - 1; i >= 0; i--) 
	{
		fb = layers[i]->backpropagate(fb);
	}
}

//Update
void Network::update()
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i]->update(alpha);
	}
}

