#pragma once
#include<iostream>
#include<vector>
#include<type_traits>
#include<typeinfo>
#include<initializer_list>
#include<string>
#include<algorithm>
#include"Tensor.h"


namespace TPP 
{
	//Base class for layer
	class Layer
	{
	public:
		virtual ~Layer() {};
		//Get output
		virtual Tensor output(Tensor input) =0;

		//Calculate gradients
		virtual void backpropagate(Tensor feedback) = 0;

		//Update gradients
		virtual void update(long double lr) = 0;
	};

	//Derived class for a Dense Layer
	class Dense:public Layer
	{
	private:
		//Input (Used in backpropagation)
		Tensor input;

		//Gradients
		Tensor dW;
		Tensor dB;

		//Weights and Bias
		Tensor W;
		Tensor B;

		//Input and output size
		unsigned int input_size;
		unsigned int output_size;

	public:
		//Constructor
		Dense(unsigned int in, unsigned int out,time_t seed);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		void backpropagate(Tensor feedback) override;

		//Update function
		//Todo replace with actual optimizations
		void update(long double alpha) override;



	};
	
}
