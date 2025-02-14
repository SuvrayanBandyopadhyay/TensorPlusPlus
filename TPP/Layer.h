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
	protected:
		std::vector<size_t>_input_shape;
		
	public:
		//Number of times backpropagations last update
		float backprop_count = 0;
		virtual ~Layer() {};
		//Get output
		virtual Tensor output(Tensor input) =0;

		//Calculate gradients
		virtual Tensor backpropagate(Tensor feedback) =0;

		//Update gradients
		virtual void update(float lr);

		//Get the shape of the output
		virtual std::vector<size_t> outputShape() = 0;
	};

	//Derived class for a DENSE Layer
	class DENSE:public Layer
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
		DENSE(unsigned int in, unsigned int out, time_t seed, int min = -1,int max = 1);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;

		//Update function
		//Todo replace with actual optimizations
		void update(float alpha) override;

		std::vector<size_t> outputShape() override;

	};
	



	//Activation layers
	//RELU
	class RELU :public Layer
	{
	private:
		Tensor input;
	public:
		RELU(std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};

		std::vector<size_t> outputShape() override;

	};
	//LEAKY RELU
	class LEAKY_RELU :public Layer
	{
	private:
		Tensor input;
		float alpha;
	public:
		LEAKY_RELU(std::vector<size_t>input_shape,float a) ;

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};

		//Output shape
		std::vector<size_t> outputShape() override;

	};
	//SIGMOID
	class SIGMOID :public Layer
	{
	private:
		Tensor input;
	public:
		SIGMOID(std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};
		//Output shape
		std::vector<size_t> outputShape() override;

	};
	//Tanh 
	class TANH :public Layer
	{
	private:
		Tensor input;
		float alpha;
	public:

		TANH(std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};

		//Output shape
		std::vector<size_t> outputShape() override;

	};
	//Softmax
	class SOFTMAX :public Layer
	{
	private:
		Tensor input;
	public:

		SOFTMAX(std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};

		//Output shape
		std::vector<size_t> outputShape() override;
	};
	//Flatten
	class FLATTEN :public Layer
	{
	private:
		Tensor input;
	public:

		FLATTEN(std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override {};
		//Output shape
		std::vector<size_t> outputShape() override;
	};
	//Convolutional Layer
	class CONV:public Layer
	{
	private:
		unsigned int _stride;
		unsigned int _fsize;
		std::vector<Tensor> filter;
		std::vector<Tensor> bias;
		std::vector<size_t>_input_shape;

		//Gradients
		std::vector<Tensor> dfilter;
		std::vector<Tensor> dbias;
		//Gradient resets (used to re-initialize gradients)
		std::vector<Tensor>rfilter;
		std::vector<Tensor>rbias;
		Tensor input;

	public:

		CONV(unsigned int n, unsigned int fsize, std::vector<size_t>input_shape,time_t seed, unsigned int stride = 1, float min = -1, float  max = 1);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr)override;
		//Output shape
		std::vector<size_t> outputShape() override;
	};

	//MAXPOOLING layer
	class MAXPOOLING :public Layer
	{
	private:

		std::vector<size_t> _pool_shape;
		std::vector<size_t>_input_shape;
		std::vector<size_t>_output_shape;
		
		Tensor input;

	public:

		MAXPOOLING(unsigned int fsize, std::vector<size_t>input_shape);

		//Output function
		Tensor output(Tensor in) override;

		//Backpropagate function
		Tensor backpropagate(Tensor feedback) override;
		//Update gradients
		void update(float lr) {};
		//Output shape
		std::vector<size_t> outputShape() override;
	};

	
}
