#pragma once
#include"Layer.h"

namespace TPP 
{
	class Network 
	{
	private:
		std::vector<std::unique_ptr<Layer>>layers;
	public:
		long double alpha = 0.1;
		void addLayer(std::unique_ptr<Layer>l);

		//Output of the layers
		Tensor output(Tensor input);

		//Backpropagation of all the layers
		void backpropagate(Tensor fb);
		//Updating the gradients
		void update();
		//Get the last shape
		std::vector<size_t> getOutputShape();
	};
}