#include"Layer.h"
using namespace std;
using namespace TPP;

//DENSE LAYER
//Initialization
Dense::Dense(unsigned int in, unsigned int out, time_t seed)
{
	input_size = in;
	output_size = out;

	W = RandomTensor({ in,out }, seed);
	B = RandomTensor({ 1,out }, seed+1);

	dW = Tensor({ in,out }, 0);
	dB = Tensor({ 1,out }, 0);
}

Tensor Dense::output(Tensor in) 
{
	input = in;
	Tensor output = (in * W) + B;
	return output;
}

//Backpropagation
void Dense::backpropagate(Tensor feedback) 
{
	dW += input.transpose() * feedback;
	dB += feedback;
}

//Update
void Dense::update(long double alpha) 
{
	W = W - alpha*dW;
	B = B - alpha * dB;

	dW = Tensor({ input_size,output_size }, 0);
	dB = Tensor({ 1,output_size }, 0);


}

