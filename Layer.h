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
		virtual Tensor backpropagate(Tensor feedback) = 0;

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
		Dense(unsigned int in, unsigned int out, time_t seed, int min = -1,int max = 1);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;

		//Update function
		//Todo replace with actual optimizations
		void update(long double alpha) override;



	};
	



	//Activation layers
	//RELU
	class RELU :public Layer
	{
	private:
		Tensor input;
	public:
		RELU() = default;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};

	};
	//LEAKY RELU
	class LEAKY_RELU :public Layer
	{
	private:
		Tensor input;
		long double alpha;
	public:
		LEAKY_RELU(long double a) ;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};

	};
	//SIGMOID
	class SIGMOID :public Layer
	{
	private:
		Tensor input;
	public:
		SIGMOID() = default;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};

	};
	//Tanh 
	class TANH :public Layer
	{
	private:
		Tensor input;
		long double alpha;
	public:

		TANH() = default;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};

	};
	//Softmax
	class SOFTMAX :public Layer
	{
	private:
		Tensor input;
	public:

		SOFTMAX() = default;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};
	};
	//Flatten
	class FLATTEN :public Layer
	{
	private:
		Tensor input;
	public:

		FLATTEN() = default;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};
	};
	//Convolutional Layer
	class CONV:public Layer
	{
	private:
		unsigned int _stride;
		std::vector<Tensor> filter;
		std::vector<Tensor> bias;
		Tensor input;
	public:

		CONV(unsigned int n, unsigned int fsize, std::vector<size_t>input_shape,time_t seed, unsigned int stride = 1, long double min = -1, long double  max = 1);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(long double lr)override {};
	};
}
