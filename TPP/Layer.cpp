#include"Layer.h"
using namespace std;
using namespace TPP;

#define EPSILON 0.01

//Layer
void Layer::update(float lr)
{
	backprop_count = 0;
}

//DENSE LAYER
//Initialization
DENSE::DENSE(unsigned int in, unsigned int out, time_t seed,int min,int max)
{
	input_size = in;
	output_size = out;
	_input_shape = { 1,out };

	W = RandomTensor({ in,out }, seed,min,max);
	B = RandomTensor({ 1,out }, seed+1,min,max);

	dW = Tensor({ in,out }, 0);
	dB = Tensor({ 1,out }, 0);
}

//Output
Tensor DENSE::output(Tensor in) 
{
	input = in;
	Tensor output = (in * W) + B;
	return output;
}

//Backpropagation
Tensor DENSE::backpropagate(Tensor feedback) 
{
	backprop_count++;
	dW += input.transpose() * feedback;
	dB += feedback;

	return feedback * W.transpose();
}

vector<size_t> DENSE::outputShape()
{
	return B.shape();
}

//Update
void DENSE::update(float alpha) 
{
	if (backprop_count > 0)
	{
		W = W - alpha * dW * (1.0 / backprop_count);
		B = B - alpha * dB * (1.0 / backprop_count);
		backprop_count = 0;

		dW = Tensor({ input_size,output_size }, 0);
		dB = Tensor({ 1,output_size }, 0);
	}

}

//Activation functions
//Relu
RELU::RELU(vector<size_t>input_shape)
{
	_input_shape = input_shape;
}
//Output
Tensor RELU::output(Tensor in) 
{
	input = in;
	//The output tensor
	vector<float>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++) 
	{
		outdata[i] = max(outdata[i], (float)0.0);
	}
	return Tensor(in.shape(), outdata);
	
}
//Backpropagate
Tensor RELU::backpropagate(Tensor feedback) 
{
	backprop_count++;
	vector<float> diffdata = input.data();
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
//output shape
vector<size_t>RELU::outputShape()
{
	return _input_shape;
}

//Leaky RELU
LEAKY_RELU::LEAKY_RELU(vector<size_t>input_shape,float a) 
{
	alpha = a;
	_input_shape = input_shape;
}
//Output
Tensor LEAKY_RELU::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<float>outdata = in.data();
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
	backprop_count++;
	vector<float> diffdata = input.data();
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

//Output shape
vector<size_t>LEAKY_RELU::outputShape()
{
	return _input_shape;
}

//SIGMOID
SIGMOID::SIGMOID(vector<size_t>input_shape) 
{
	_input_shape = input_shape;
}
//Output
Tensor SIGMOID::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<float>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++)
	{
		outdata[i] = 1 / (1 + exp(-outdata[i]));
	}
	return Tensor(in.shape(), outdata);

}
//Backpropagate
Tensor SIGMOID::backpropagate(Tensor feedback)
{
	backprop_count++;
	vector<float> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++)
	{
		float e = exp(-diffdata[i]);
		diffdata[i] = e / pow((1 + e), 2);
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}

//Output shape
vector<size_t>SIGMOID::outputShape()
{
	return _input_shape;
}
//Tanh 
TANH::TANH(vector<size_t>input_shape) 
{
	_input_shape = input_shape;
}
//Output
Tensor TANH::output(Tensor in)
{
	input = in;
	//The output tensor
	vector<float>outdata = in.data();
	for (int i = 0; i < outdata.size(); i++)
	{
		outdata[i] = tanh(outdata[i]);//tanh 
	}
	return Tensor(in.shape(), outdata);

}
//Backpropagate
Tensor TANH::backpropagate(Tensor feedback)
{
	backprop_count++;
	vector<float> diffdata = input.data();
	//Create the differentiation vector
	for (int i = 0; i < diffdata.size(); i++)
	{
		diffdata[i] = 1 - pow(tanh(diffdata[i]),2);//1-tanh^2 x
	}
	Tensor diff = Tensor(input.shape(), diffdata);

	//Return hadamard product
	return (diff % feedback);
}

//Output shape
vector<size_t> TANH::outputShape()
{
	return _input_shape;
}

//Softmax 
SOFTMAX::SOFTMAX(vector<size_t>input_shape) 
{
	_input_shape = input_shape;
}
//Output
Tensor SOFTMAX::output(Tensor in)
{
	input = in;
	float denom = 0;
	vector<float>outdata = in.data();
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
	backprop_count++;
	float denom = 0;
	vector<float>diffdata = input.data();
	//Get the value of the constant summation e^c
	for (int i = 0; i < diffdata.size(); i++)
	{
		diffdata[i] = exp(diffdata[i]);
		denom += diffdata[i];
	}
	//The derivative tensor
	for (int i = 0; i < diffdata.size(); i++)
	{
		float c = denom - diffdata[i];
		diffdata[i] *= c / (pow(denom,2)+EPSILON);//We want c e^x/(e^x + c) where c is e^x2 + e^x3 + .....
	}
	Tensor diff = Tensor(input.shape(), diffdata);
	//Return hadamard product
	return (diff % feedback);
}

//Output shape
vector<size_t>SOFTMAX::outputShape()
{
	return _input_shape;
}
//Flatten
FLATTEN::FLATTEN(vector<size_t>input_shape) 
{
	_input_shape = input_shape;
}
//Output
Tensor FLATTEN::output(Tensor in)
{
	input = in;
	return input.flatten();

}
//Flatten
Tensor FLATTEN::backpropagate(Tensor feedback)
{
	backprop_count++;
	return feedback.reshape(input.shape());
}
vector<size_t>FLATTEN::outputShape()
{
	size_t k=1;
	for (unsigned int i = 0; i < _input_shape.size(); i++) 
	{
		k *= _input_shape[i];
	}

	return {1,k};
}
//CONV
TPP::CONV::CONV(unsigned int n, unsigned int fsize, std::vector<size_t>input_shape, time_t seed, unsigned int stride, float min, float max)
{
	_stride = stride;
	_fsize = fsize;
	unsigned int resultx = (int)((input_shape[input_shape.size() - 1]-fsize)/stride) + 1;
	unsigned int resulty = (int)((input_shape[input_shape.size() - 2] - fsize) / stride) + 1;
	
	if (input_shape[input_shape.size() - 1] % stride != 0 || input_shape[input_shape.size() - 2] % stride != 0) 
	{
		throw std::invalid_argument("input size is not a multiple of filter size");
	}
	
	_input_shape = input_shape;
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
		Tensor df = Tensor(filter_shape, 0);
		Tensor db = Tensor(bias_shape, 0);
		//Push back the filters and bias;
		filter.push_back(f);
		bias.push_back(b);
		//Gradients
		dfilter.push_back(df);
		dbias.push_back(db);
		//Reset Tensors
		rfilter.push_back(df);
		rbias.push_back(db);
	}

}
//Output
Tensor CONV::output(Tensor in)
{
	//Check if input is valid
	if (in.shape() != _input_shape) 
	{
	
		throw invalid_argument("Invalid input");
	}

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
//Backpropagate convolutional layer
Tensor CONV::backpropagate(Tensor feedback)
{
	backprop_count++;
	//Dilate the feedback tensor
	Tensor fb_filter = feedback.dilate(_stride - 1);

	//New feedback
	Tensor fb = Tensor(_input_shape, 0);

	//Convolutional operation with input
	for (unsigned int i = 0; i < dfilter.size(); i++)
	{
		//Get the delta change
		
		Tensor delta = input.convMult(fb_filter.at({ i }), 1);



		//Update the values
		dfilter[i] += delta;
		
		dbias[i] += feedback.at({ i });

	
		//Calculating new feedback
		//Increase in padding 
		unsigned int padl = _fsize - 1;
		//Increase in dilation
		unsigned int dill = _stride-1;

		Tensor imm = dbias[i].dilate(dill);
		imm = imm.pad(padl);

	
		
		fb += imm.convMult(filter[i].rotate().rotate(), 1);//Since dbias[i] = feedback.at({i})
		
		
	}



	return fb;
}

//Update
void CONV::update(float lr) 
{
	if (backprop_count > 0)
	{
		for (unsigned int i = 0; i < filter.size(); i++)
		{
			filter[i] -= lr * dfilter[i] * (1.0 / backprop_count);
			bias[i] -= lr * dbias[i] * (1.0 / backprop_count);
		}
		backprop_count = 0;
		dfilter = rfilter;
		dbias = rbias;
	}
}

//Output shape
vector<size_t>CONV::outputShape()
{
	vector<size_t> output_shape = bias[0].shape();
	//Since it returns a tensor 
	output_shape.insert(output_shape.begin(), bias.size());
	return output_shape;
}


//MAXPOOLING
//CONV
TPP::MAXPOOLING::MAXPOOLING(unsigned int fsize, std::vector<size_t>input_shape)
{

	//The input shape
	_input_shape = input_shape;

	//The output shape
	
	unsigned int resultx = ((input_shape[input_shape.size() - 1]) / fsize) ;
	unsigned int resulty = ((input_shape[input_shape.size() - 2]) / fsize) ;


	_output_shape = input_shape;
	_output_shape[_output_shape.size() - 1] = resultx;
	_output_shape[_output_shape.size() - 2] = resulty;

	//The filter shape
	_pool_shape = input_shape;
	_pool_shape[input_shape.size() - 2] = fsize;
	_pool_shape[input_shape.size() - 1] = fsize;


}


//Faster pooling
//Function for quick pooling
Tensor quickPool(Tensor first, vector<size_t>pool_shape)
{
	//Input
	unsigned int in_row = first.shape()[first.dim() - 2];
	unsigned int in_col = first.shape()[first.dim() - 1];

	//Filter
	unsigned int f_row = pool_shape[pool_shape.size() - 2];
	unsigned int f_col = pool_shape[pool_shape.size() - 1];

	//New data
	vector<float> newdata;

	unsigned int y = (in_col/f_col);
	unsigned int x = (in_row/f_row);
	for (int i = 0; i + f_row <= in_row; i += f_row)
	{
		
		for (int j = 0; j + f_col <= in_col; j += f_col)
		{
			//The maximum value
			float max = -INFINITY;

			//Now the operations
			
			for (int k = 0; k < f_row; k++)
			{
			
				for (int l = 0; l < f_col; l++)
				{

					//Value >= max
					float val = first.data()[(i+k) * in_col + (j+l)];
					
					if (val >= max)
					{
						max = val;

					}

					
				
				}
				
			}
			newdata.push_back(max);

		}

	}

	return Tensor({ x,y }, newdata);
}

//Recursive pooling
void recPool(Tensor first, vector<size_t> pool_shape, Tensor* finalT, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recPool(first, pool_shape, finalT, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor res = quickPool(first, pool_shape);

		finalT->set(slice, res);

	}

}



//Output
Tensor MAXPOOLING::output(Tensor in)
{
	//Check if input is valid
	if (in.shape() != _input_shape)
	{

		throw invalid_argument("Invalid input");
	}

	input = in;

	

	//Resultant tensor
	Tensor result(_output_shape, 0);
	recPool(input, _pool_shape, &result);
	
	

	
	return result;



}


//Function for quick pooling backpropagation
Tensor quickPoolBack(Tensor input,Tensor fb, vector<size_t>pool_shape)
{
	//Input
	unsigned int in_row = input.shape()[input.dim() - 2];
	unsigned int in_col = input.shape()[input.dim() - 1];

	//Feedback dimension
	unsigned int fb_row = fb.shape()[fb.dim() - 2];
	unsigned int fb_col = fb.shape()[fb.dim() - 1];
	//Filter
	unsigned int f_row = pool_shape[pool_shape.size() - 2];
	unsigned int f_col = pool_shape[pool_shape.size() - 1];
	//New data
	vector<float>new_data(in_row*in_col,0);

	unsigned int y = 0;
	for (int i = 0; i + f_row <= in_row; i += f_row)
	{
		unsigned int x = 0;
		for (int j = 0; j + f_col <= in_col; j += f_col)
		{
			float max = -INFINITY;

			//Index of max value
			unsigned int maxindex = -1;
			//Now the operations
			for (int k = 0; k < f_row; k++)
			{
				for (int l = 0; l < f_col; l++)
				{
					//index
					unsigned int index = (i + k) * in_col + (j + l);

					float val = input.data()[index];
					if (val >= max)
					{
						max = val;
						maxindex = index;
					}


				}

			}

			if (max == -INFINITY) 
			{
				throw std::exception("ERROR::INPUT IS NOT DEFINED");
			}

			//Feedback index
			
			unsigned int fbindex = y * f_col + x;

			//Setting the output
			new_data[maxindex] = fb.data()[fbindex];
			x++;
		}
		y++;
	}

	
	return Tensor({ in_row,in_col }, new_data);
}


//Recursive maxpooling backpropagation
void recPoolBack(Tensor first,Tensor fb, vector<size_t> pool_shape, Tensor* finalT, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recPoolBack(first,fb, pool_shape, finalT, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor mat = first.at(slice);
		Tensor fbmat = fb.at(slice);

		Tensor ret = quickPoolBack(mat, fbmat, pool_shape);
		finalT->set(slice, ret);

	}

}




//Backpropagate maxpooling layer
Tensor MAXPOOLING::backpropagate(Tensor feedback)
{
	backprop_count++;
	//Tensor to return 
	Tensor ret = Tensor(_input_shape, 0);
	recPoolBack(input,feedback, _pool_shape, &ret);
	return ret;
}



//Output shape
vector<size_t>MAXPOOLING::outputShape()
{
	return _output_shape;
}

