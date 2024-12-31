#include"Layer.h"
using namespace std;
using namespace TPP;

//DENSE LAYER
//Initialization
Dense::Dense(unsigned int in, unsigned int out, time_t seed,int min,int max)
{
	input_size = in;
	output_size = out;

	W = RandomTensor({ in,out }, seed,min,max);
	B = RandomTensor({ 1,out }, seed+1,min,max);

	dW = Tensor({ in,out }, 0);
	dB = Tensor({ 1,out }, 0);
}

//Output
Tensor Dense::output(Tensor in) 
{
	input = in;
	Tensor output = (in * W) + B;
	return output;
}

//Backpropagation
Tensor Dense::backpropagate(Tensor feedback) 
{
	dW += input.transpose() * feedback;
	dB += feedback;

	return feedback * W.transpose();
}

//Update
void Dense::update(long double alpha) 
{
	W = W - alpha*dW;
	B = B - alpha * dB;

	dW = Tensor({ input_size,output_size }, 0);
	dB = Tensor({ 1,output_size }, 0);


}

//Activation functions
//Relu
//Output
Tensor RELU::output(Tensor in) 
{
	input = in;
	//The output tensor
	vector<long double>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++) 
	{
		outdata[i] = max(outdata[i], (long double)0.0);
	}
	return Tensor(in.shape(), outdata);
	
}
//Backpropagate
Tensor RELU::backpropagate(Tensor feedback) 
{
	vector<long double> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++) 
	{
		if (diffdata[i] > 0) 
		{
			diffdata[i] = 1;
		}
		else
		{
			diffdata[i] = 0;
		}
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}


//Leaky RELU
LEAKY_RELU::LEAKY_RELU(long double a) 
{
	alpha = a;
}
//Output
Tensor LEAKY_RELU::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<long double>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++)
	{
		if (outdata[i] <= 0) 
		{
			outdata[i] = alpha * outdata[i];
		}
	}
	return Tensor(in.shape(), outdata);

}
//Backpropagate
Tensor LEAKY_RELU::backpropagate(Tensor feedback)
{
	vector<long double> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++)
	{
		if (diffdata[i] > 0)
		{
			diffdata[i] = 1;
		}
		else
		{
			diffdata[i] = alpha;
		}
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}

//SIGMOID
//Output
Tensor SIGMOID::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<long double>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++)
	{
		outdata[i] = 1 / (1 + exp(-outdata[i]));
	}
	return Tensor(in.shape(), outdata);

}
//Backpropagate
Tensor SIGMOID::backpropagate(Tensor feedback)
{
	vector<long double> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++)
	{
		long double e = exp(-diffdata[i]);
		diffdata[i] = e / pow((1 + e), 2);
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}

//Tanh 
//Output
Tensor TANH::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<long double>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++)
	{
		outdata[i] = tanh(outdata[i]);//tanh 
	}
	return Tensor(in.shape(), outdata);

}
//Backpropagate
Tensor TANH::backpropagate(Tensor feedback)
{
	vector<long double> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++)
	{
		diffdata[i] = 1 - pow(tanh(diffdata[i]),2);//1-tanh^2 x
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}

//Softmax 
//Output
Tensor SOFTMAX::output(Tensor in)
{
	input = in;
	long double denom = 0;
	vector<long double>outdata = in.data();
	//Get the value of the constant summation e^c
	for (int i = 0; i < outdata.size(); i++) 
	{
		outdata[i] = exp(outdata[i]);
		denom += outdata[i];
	}
	//The output tensor
	for (int i = 0; i < outdata.size(); i++)
	{
		outdata[i] /= denom;
	}
	return Tensor(in.shape(), outdata);

}
//Softmax
Tensor SOFTMAX::backpropagate(Tensor feedback)
{
	long double denom = 0;
	vector<long double>diffdata = input.data();
	//Get the value of the constant summation e^c
	for (int i = 0; i < diffdata.size(); i++)
	{
		diffdata[i] = exp(diffdata[i]);
		denom += diffdata[i];
	}
	//The derivative tensor
	for (int i = 0; i < diffdata.size(); i++)
	{
		long double c = denom - diffdata[i];
		diffdata[i] *= c / pow(denom,2);//We want c e^x/(e^x + c) where c is e^x2 + e^x3 + .....
	}
	Tensor diff = Tensor(input.shape(), diffdata);
	//Return hadamard product
	return (diff % feedback);
}

//Flatten
//Output
Tensor FLATTEN::output(Tensor in)
{
	input = in;
	return input.flatten();

}
//Flatten
Tensor FLATTEN::backpropagate(Tensor feedback)
{
	return feedback.reshape(input.shape());
}

//CONV
TPP::CONV::CONV(unsigned int n, unsigned int fsize, std::vector<size_t>input_shape, time_t seed, unsigned int stride, long double min, long double max)
{
	_stride = stride;
	unsigned int resultx = ((input_shape[input_shape.size() - 1]-fsize)/stride) + 1;
	unsigned int resulty = ((input_shape[input_shape.size() - 2] - fsize) / stride) + 1;

	//The filter shape
	vector<size_t>filter_shape = input_shape;
	filter_shape[input_shape.size() - 2] = fsize;
	filter_shape[input_shape.size() - 1] = fsize;
	//The bias shape
	vector<size_t>bias_shape = input_shape;
	bias_shape[input_shape.size() - 1] = resultx;
	bias_shape[input_shape.size() - 2] = resulty;



	for (int i = 0; i < n; i++) 
	{
		Tensor f = RandomTensor(filter_shape, seed+i,min,max);
		Tensor b = RandomTensor(bias_shape, seed+i, min, max);

		//Push back the filters and bias;
		filter.push_back(f);
		bias.push_back(b);
	}

}
//Output
Tensor CONV::output(Tensor in)
{
	input = in;

	//Resultant tensor
	vector<size_t>fshape = bias[0].shape();
	fshape.insert(fshape.begin(), bias.size());
	Tensor result(fshape, 0);

	for (unsigned int i = 0; i < filter.size(); i++) 
	{
		
		Tensor r = input.convMult(filter[i], _stride)+bias[i];

		result.set({ i }, r);
	}
	return result;
	


}
//Flatten
Tensor CONV::backpropagate(Tensor feedback)
{
	return feedback.reshape(input.shape());
}