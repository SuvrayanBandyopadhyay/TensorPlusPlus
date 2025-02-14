#include"Tensor.h"

using namespace std;
using namespace TPP;

//Helper function to see how many elements are there for a given shape
unsigned int numberFromShape(vector<size_t>shape) 
{
	unsigned int size = 1;
	for (int i = 0; i < shape.size(); i++) 
	{
		size *= shape[i];
	}
	return size;
}

//Constructor
Tensor::Tensor(vector<size_t>shape, vector<float>data) 
{
	if (numberFromShape(shape) == data.size())
	{
		_shape = shape;
		_data = data;
	}
	else 
	{
		throw invalid_argument("Invalid shape");
	}
}
Tensor::Tensor(std::vector<size_t>shape, float value) 
{
	_shape = shape;


	//Subtract 1 from each element in shape to get the raw size

	unsigned int raw_size = numberFromShape(_shape);

	vector<float>data;
	for (unsigned i = 0; i < raw_size; i++) 
	{
		data.push_back(value);
	}
	_data = data;
	
}

//Function defintion for getFlatIndex get flat index is used to convert an index of a tensor to raw 1D index. It can also be used to find the total raw length of the data contained in the tensor
size_t Tensor::getFlatIndex(vector<size_t> index)
{
	//The flattened index
	size_t flat_index = 0;
	size_t multiplier = 1;

	//Find the value

	for (int i=index.size()-1;i>=0;i--)
	{
		flat_index += multiplier * index[i];
		multiplier *= _shape[i];
		
	}

	//Return the flattened index
	return flat_index;
}
//Shape of the tensor
vector<size_t> Tensor::shape()
{
	return _shape;
}
//Get raw data 
vector<float> Tensor::data() 
{
	return _data;
}
//Assignment operator
void Tensor::operator=(Tensor second) 
{
	_shape = second._shape;
	_data = second._data;

}

//Equlity operator
bool Tensor::operator==(Tensor second) 
{
	return(_data == second._data && _shape == second._shape);
}

//Addition operator
Tensor Tensor::operator+(Tensor second) 
{
	//If unequal shapes
	if (_shape != second._shape) 
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else 
	{
		vector<float> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++) 
		{
			float element = _data[i] + second._data[i];
			data2.push_back(element);
		}
		Tensor t2(_shape, data2);
		return t2;
	}

}

//Addition assignment operator
void Tensor::operator+=(Tensor second) 
{
	//If unequal shapes
	if (_shape != second._shape)
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else
	{
	

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			//Add and assign data
			_data[i] = _data[i] + second._data[i];
			
		}
		
	}
}

//Subtraction operator
Tensor Tensor::operator-(Tensor second)
{
	//If unequal shapes
	if (_shape != second._shape)
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else
	{
		vector<float> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			float element = _data[i] - second._data[i];
			data2.push_back(element);
		}
		Tensor t2(_shape, data2);
		return t2;
	}

}

//Subtraction assignment operator
void Tensor::operator-=(Tensor second)
{
	//If unequal shapes
	if (_shape != second._shape)
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else
	{


		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			//Add and assign data
			_data[i] = _data[i] - second._data[i];

		}

	}
}

//Hadamard Multiplication operator
Tensor Tensor::operator%(Tensor second)
{
	//If unequal shapes
	if (_shape != second._shape)
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else
	{
		vector<float> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			float element = _data[i] * second._data[i];
			data2.push_back(element);
		}
		Tensor t2(_shape, data2);
		return t2;
	}

}

//Hadamard Multiplication assignment operator
void Tensor::operator%=(Tensor second)
{
	//If unequal shapes
	if (_shape != second._shape)
	{
		throw std::invalid_argument("Tensors should be of the same shape to add");
	}
	else
	{


		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			//Add and assign data
			_data[i] = _data[i] * second._data[i];

		}

	}
}

//At function
Tensor Tensor::at(std::initializer_list<size_t>pos) 
{
	int index = pos.size();
	//The new shape of the tensor
	vector<size_t> _newshape;
	//The new data
	

	//We are slicing the tensor
	vector<size_t>startpos = pos;
	vector<size_t>endpos = pos;
	
	for (int i = pos.size(); i < _shape.size(); i++) 
	{
		startpos.push_back(0);
		endpos.push_back(_shape[i]-1);
		_newshape.push_back(_shape[i]);
	}

	unsigned int start_flat = getFlatIndex(startpos);
	unsigned int end_flat = getFlatIndex(endpos);
	
	//We include the last element as well
	vector<float> _newdata(_data.begin() + start_flat, _data.begin() + end_flat+1);
	
	return Tensor(_newshape,_newdata);
	
}
Tensor Tensor::at(std::vector<size_t>pos)
{
	int index = pos.size();
	//The new shape of the tensor
	vector<size_t> _newshape;
	//The new data
	vector<float> _newdata;

	//We are slicing the tensor
	vector<size_t>startpos = pos;
	vector<size_t>endpos = pos;

	for (int i = pos.size(); i < _shape.size(); i++)
	{
		startpos.push_back(0);
		endpos.push_back(_shape[i] - 1);
		_newshape.push_back(_shape[i]);
	}

	unsigned int start_flat = getFlatIndex(startpos);
	unsigned int end_flat = getFlatIndex(endpos);


	for (unsigned int i = start_flat; i <= end_flat; i++)
	{
		_newdata.push_back(_data[i]);
	}


	return Tensor(_newshape, _newdata);

}

//Set function
void Tensor::set(std::initializer_list<size_t>pos, Tensor val) 
{
	int index = pos.size();
	//The new shape of the tensor
	vector<size_t> _newshape;
	//The new data
	vector<float> _newdata;

	//We are slicing the tensor
	vector<size_t>startpos = pos;
	vector<size_t>endpos = pos;

	for (int i = pos.size(); i < _shape.size(); i++)
	{
		startpos.push_back(0);
		endpos.push_back(_shape[i] - 1);
		_newshape.push_back(_shape[i]);
	}

	unsigned int start_flat = getFlatIndex(startpos);
	unsigned int end_flat = getFlatIndex(endpos);


	//If there is a dimension mismatch
	if (_newshape != val.shape()) 
	{
		throw std::invalid_argument("Tensor dimensions dont match");
	}

	//Set the values
	int x = 0;
	for (unsigned int i = start_flat; i <= end_flat; i++)
	{
		_data[i] = val.data()[x];
		x++;
	}
	

	
}
void Tensor::set(std::vector<size_t>pos, Tensor val)
{
	int index = pos.size();
	//The new shape of the tensor
	vector<size_t> _newshape;
	//The new data
	vector<float> _newdata;

	//We are slicing the tensor
	vector<size_t>startpos = pos;
	vector<size_t>endpos = pos;

	for (int i = pos.size(); i < _shape.size(); i++)
	{
		startpos.push_back(0);
		endpos.push_back(_shape[i] - 1);
		_newshape.push_back(_shape[i]);
	}

	unsigned int start_flat = getFlatIndex(startpos);
	unsigned int end_flat = getFlatIndex(endpos);


	//If there is a dimension mismatch
	if (_newshape != val.shape())
	{
		throw std::invalid_argument("Tensor dimensions dont match");
	}

	//Set the values
	int x = 0;
	for (unsigned int i = start_flat; i <= end_flat; i++)
	{
		_data[i] = val.data()[x];
		x++;
	}



}

// Scalar, Vector and Matrix definition
Tensor TPP::Scalar(float val) 
{
	vector<size_t>shape;
	return Tensor(shape, vector<float>({ val }));
}

Tensor TPP::Vector(std::initializer_list<float>data) 
{
	vector<size_t>shape;
	shape.push_back(data.size());
	return Tensor(shape, vector<float>(data));
}

Tensor TPP::Matrix(std::initializer_list<std::initializer_list<float>>data)
{
	vector<size_t>shape;
	shape.push_back(data.size());

	int columns = data.begin()->size();
	shape.push_back(columns);
	
	
	vector<float> _data;
	for (auto i : data) 
	{
		if (i.size() != columns) 
		{
			throw std::invalid_argument("Matrix shape is invalid");
		}
		for (auto j : i) 
		{
			
			_data.push_back(j);
		}
	}

	return Tensor(shape, _data);
}

//Returns dimension of a tensor
unsigned int Tensor::dim() 
{
	return _shape.size();
}

//Returns string of the shape of a tensor
string Tensor::shapeString() 
{
	string ret = "(";
	for (int i = 0; i < _shape.size(); i++) 
	{
		ret += to_string(_shape[i]) + ",";
	}
	ret += ")";
	return ret;
}

Tensor quickMatMult(Tensor &a, Tensor &b) 
{
	unsigned int rows = a.shape()[a.dim() - 2];
	unsigned int inter = a.shape()[a.dim() - 1];
	unsigned int cols = b.shape()[b.dim() - 1];

	vector<float>newdata(rows*cols,0);
	unsigned int c_index = 0;
	for (unsigned int i = 0; i < rows; i++) 
	{
		
		for (unsigned int j = 0; j < cols;j++) 
		{
		
			float val = 0;


			unsigned int a_index = i * inter;
			unsigned int b_index = j;
			for (unsigned int k = 0; k < inter; k++) 
			{
				

				val += a.data()[a_index++] * b.data()[b_index];
				b_index += cols;
			}
			
			newdata[c_index++] = val;
		
		}
	}
	return Tensor({ rows,cols }, newdata);
	
}

void recMultiply(Tensor &first, Tensor &second, Tensor* finalT, vector<size_t>slice = {}, int currdim=0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			
			slice.push_back(i);
			recMultiply(first, second, finalT,slice,currdim+1);

		}
	}
	//If we have already reached the penultimate depth
	else 
	{
		Tensor a = first.at(slice);
		Tensor b = second.at(slice);

		Tensor c = quickMatMult(a, b);
		
		finalT->set(slice,c);

	
	}
	
}


//Batch matrix multiplication formula
Tensor Tensor::matMul(Tensor second) 
{
	vector<size_t> _newshape;
	//Check if they can be multiplied in batches
	if (_shape.size() != second._shape.size()) 
	{
		string error = "Cant batch multiply tensors of shapes " + shapeString() + " and " + second.shapeString();
		throw std::invalid_argument(error.c_str());
	}

	for (unsigned int i = 0; i < _shape.size() - 2; i++) 
	{
		if (_shape[i] != second._shape[i])
		{
			string error = "Cant batch multiply tensors of shapes " + shapeString() + " and " + second.shapeString();
			throw std::invalid_argument(error.c_str());
		}
	}
	//Check if we can multiply the last 2 dimensions
	if (_shape[_shape.size() - 1] != second.shape()[second.shape().size() - 2])
	{
		string error = "Cant batch multiply tensors of shapes " + shapeString() + " and " + second.shapeString();
		throw std::invalid_argument(error.c_str());
	}

	//Create the final tensor shape
	vector<size_t>final_shape = _shape;
	final_shape.pop_back();
	final_shape.push_back(second._shape[second._shape.size() - 1]);

	Tensor finalTensor(final_shape, 0);

	recMultiply(*this, second, &finalTensor);
	return finalTensor;
}
Tensor Tensor::operator*(Tensor second) 
{
	return this->matMul(second);
}
void Tensor::operator*=(Tensor second) 
{
	(*this) = this->matMul(second);
}


//Quick transpose
Tensor quickTranspose(Tensor &a) 
{
    //The number of rows and columns
	unsigned int rows = a.shape()[a.shape().size() - 2];
	unsigned int columns = a.shape()[a.shape().size() - 1];

	vector<float>newdata(rows * columns, 0);

	for (unsigned int i = 0; i < rows; i++) 
	{
		
		for (unsigned int j = 0; j < columns; j++) 
		{
			unsigned int a_index = i * columns + j;
			unsigned int b_index = j * rows + i;
			newdata[b_index] = a.data()[a_index];
		}
	}

	return Tensor({ columns,rows }, newdata);

}

//Transpose

void recTranspose(Tensor &first, Tensor* finalT, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recTranspose(first, finalT, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor c = quickTranspose(first);
		finalT->set(slice, c);
	}

}

Tensor Tensor::transpose()
{
	if (dim() < 2)
	{
		throw invalid_argument("Cannot Find transpose of this tensor. Must have atleast 2 dimensions");
	}
	else 
	{
		vector<size_t>tshape = _shape;
		size_t temp = tshape[tshape.size() - 1];
		tshape[tshape.size() - 1] = tshape[tshape.size() - 2];
		tshape[tshape.size() - 2] = temp;

		Tensor result(tshape, 0);
		recTranspose(*this, &result);
		return result;
	}
	
}

//function to recursively print a tensor

void printTensor(std::ostream& os, Tensor t)
{
	if (t.dim() == 0) 
	{
		os << t.data()[0] << ",";
	}
	else 
	{
		os << "{";
		for (int i = 0; i < t.shape()[0]; i++) 
		{
			printTensor(os, t.at({(size_t)i}));
		}
		os << "}\n";
	}

}

//Function defintion for printing the tensor
std::ostream& TPP::operator<<(std::ostream& os, const Tensor& m)
{

	printTensor(os, m);
	return os;
}
//Reshape function
Tensor Tensor::reshape(vector<size_t>newshape) 
{
	if (numberFromShape(newshape) == numberFromShape(_shape)) 
	{
		return Tensor(newshape, _data);
	}
	else
	{
		throw invalid_argument("Invalid shape");
	}
}

//Flatten function
Tensor Tensor::flatten() 
{
	vector<size_t> nshape = { 1,data().size() };
	return Tensor(nshape, _data);
}
//Flatten Column
Tensor Tensor::flattenCol()
{
	vector<size_t> nshape = { data().size(),1 };
	
	return Tensor(nshape, _data);
}

//Scalar Multiplication
Tensor Tensor::operator*(float second)
{
	vector<float>newdata = _data;
	for (int i = 0; i < newdata.size(); i++)
	{
		newdata[i] *= second;
	}
	return Tensor(_shape, newdata);
}

void Tensor::operator*=(float second)
{
	for (int i = 0; i < _data.size(); i++)
	{

		_data[i] *= second;
	}
}

Tensor TPP::operator*(float second, const Tensor& tensor)
{
	Tensor ret = (Tensor)tensor * second;
	return ret;
}

//Function to help in quick convolution operations
Tensor quickConv(Tensor &first, Tensor second, unsigned int stride)
{
	//Input
	unsigned int in_row = first.shape()[first.dim() - 2];
	unsigned int in_col = first.shape()[first.dim() - 1];

	//Filter
	unsigned int f_row = second.shape()[second.dim() - 2];
	unsigned int f_col = second.shape()[second.dim() - 1];

	vector<float>newdata;
	unsigned int finalx, finaly;
	unsigned int y = 0;
	for (int i = 0; i + f_row <=in_row; i += stride)
	{
		
		unsigned int x = 0;
		for (int j = 0; j + f_col <= in_col; j += stride)
		{
			float convVal = 0;
			//Now the operations
			for (int k = 0; k < f_row; k++)
			{
				for (int l = 0; l < f_col; l++)
				{
					//First index
					unsigned int findex = (i+k) * in_col + (j + l);
					//Second index
					unsigned int sindex = k * f_col + l;

					convVal += first.data()[findex] * second.data()[sindex];
				}
			}

			//Final index
			newdata.push_back(convVal);
			x++;
		}
		y++;
		finalx = x;
	}
	finaly = y;

	return Tensor({ finalx,finaly }, newdata);
}

//Recursive convolution function
void recConv(Tensor &first, Tensor &second, Tensor* finalT, unsigned int stride, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recConv(first, second, finalT,stride, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		//Get the tensors at this level of recursion
		Tensor a = first.at(slice);
		Tensor b = second.at(slice);
		Tensor c = quickConv(a, b, stride);

		//Set the value
		finalT->set(slice, c);


	}

}

//Convolutional multiplication
Tensor Tensor::convMult(Tensor second,unsigned int stride)
{
	vector<size_t> _newshape;
	//Check if they can be multiplied in batches
	if (_shape.size() != second._shape.size())
	{
		string error = "Cant batch multiply tensors of shapes " + shapeString() + " and " + second.shapeString();
		throw std::invalid_argument(error.c_str());
	}
	//Check if all but the last 2 dimensions are the same
	for (unsigned int i = 0; i < _shape.size() - 2; i++)
	{
		if (_shape[i] != second._shape[i])
		{
			string error = "Cant batch multiply tensors of shapes " + shapeString() + " and " + second.shapeString();
			throw std::invalid_argument(error.c_str());
		}
	}


	//Create the final tensor shape
	vector<size_t>final_shape = _shape;

	//Shape after convolution
	final_shape[final_shape.size() - 1] = (int)((final_shape[final_shape.size() - 1]- second.shape()[second.shape().size() - 1])/stride)+1;
	final_shape[final_shape.size() - 2] = (int)((final_shape[final_shape.size() - 2] - second.shape()[second.shape().size() - 2]) / stride) + 1;

	Tensor finalTensor(final_shape, 0);

	recConv(*this, second, &finalTensor,stride);
	return finalTensor;
}
//quick dilation function
Tensor quickDil(Tensor &first, unsigned int dilation,vector<size_t>outshape) 
{
	//Input x and y bounds
	unsigned int in_x = first.shape()[first.dim() - 1];
	unsigned int in_y = first.shape()[first.dim() - 2];
	//Final x and y
	unsigned int fin_x = outshape[0];
	unsigned int fin_y = outshape[1];

	vector<float> newdata(fin_x * fin_y,0);

	unsigned int out_y = 0;
	for (unsigned int i = 0; i < in_y; i++)
	{
		unsigned int out_x = 0;
		for (unsigned int j = 0; j < in_x; j++)
		{
			//Getting the value
			float val = first.data()[i * in_x + j];
	
			//Setting the value in the final tensor
			newdata[out_y * fin_x + out_x] = val;
			//Go to the next pos
			out_x += 1 + dilation;
		}
		out_y += 1 + dilation;

	}

	return Tensor(outshape, newdata);
}

//Recursive dilation function
void recDil(Tensor &first, Tensor* finalT, unsigned int dilation, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recDil(first, finalT, dilation, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor a = first.at(slice);
		Tensor f = quickDil(a, dilation, finalT->at(slice).shape());

		finalT->set(slice, f);

	}

}





//Tensor dilation function
Tensor Tensor::dilate(unsigned int dilation)
{
	//Dilate the tensor
	unsigned int x_dil = (_shape[_shape.size() - 1] - 1)*dilation;
	unsigned int y_dil = (_shape[_shape.size() - 2] - 1)*dilation;

	//The new shape
	vector<size_t> nshape = _shape;
	nshape[nshape.size() - 1] += x_dil;
	nshape[nshape.size() - 2] += y_dil;

	Tensor finalT(nshape, 0);

	recDil(*this, &finalT, dilation);
	return finalT;
}

//Sum of elements
float Tensor::sumOfElements() 
{
	float sum = 0;
	for (int i = 0; i < _data.size(); i++) 
	{
		sum += _data[i];
	}
	return sum;
}

//Random Tensor
Tensor TPP::RandomTensor(std::vector<size_t>shape,time_t seed, float min, float max)
{
	//The random number engine
	mt19937_64 eng(seed);
	uniform_real_distribution<float> dis(min, max);
	
	


	unsigned int raw_size = numberFromShape(shape);

	vector<float>data;
	for (unsigned i = 0; i < raw_size; i++)
	{
		data.push_back(dis(eng));
	}

	return Tensor(shape, data);
	

}

float Tensor::value() 
{
	if (dim() != 0) 
	{
		throw std::invalid_argument("Value is only defined for scalars with no dimension. Use data() access raw tensor data");
	}
	return _data[0];
}



//Padding
//quick dilation function
Tensor quickPad(Tensor& first, unsigned int padding, vector<size_t>outshape)
{
	//Final x and y
	unsigned int fin_x = outshape[0];
	unsigned int fin_y = outshape[1];

	vector<float> newdata(fin_x * fin_y, 0);

	//To track which element of the original element we are adding right now
	unsigned int in_index = 0;


	for (unsigned int i = 0; i < fin_y; i++) 
	{
		for (unsigned int j = 0; j < fin_x; j++) 
		{

			//If part of the padded region
			if (i < padding || i >= fin_y - padding || j < padding || j >= fin_x - padding) 
			{
				continue;
			}
			//Insert value from original
			else 
			{
				newdata[i * fin_x + j] = first.data()[in_index];
				in_index++;
			}
		}
	}

	return Tensor(outshape, newdata);
}

//Recursive padding function
void recPad(Tensor& first, Tensor* finalT, unsigned int padding, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recPad(first, finalT, padding, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor a = first.at(slice);
		Tensor f = quickPad(a, padding, finalT->at(slice).shape());

		finalT->set(slice, f);

	}

}





//Tensor dilation function
Tensor Tensor::pad(unsigned int padding)
{
	//Dilate the tensor
	unsigned int x_dil = (_shape[_shape.size() - 1]) +2*padding;
	unsigned int y_dil = (_shape[_shape.size() - 2]) +2*padding;

	//The new shape
	vector<size_t> nshape = _shape;
	nshape[nshape.size() - 1] = x_dil;
	nshape[nshape.size() - 2] = y_dil;

	Tensor finalT(nshape, 0);

	//Recursive padding
	recPad(*this, &finalT, padding);
	return finalT;
}


//Function to rotate
//Quick transpose
Tensor quickRotate(Tensor& a)
{
	//The number of rows and columns
	unsigned int rows = a.shape()[a.shape().size() - 2];
	unsigned int columns = a.shape()[a.shape().size() - 1];

	vector<float>newdata(rows * columns, 0);

	for (unsigned int i = 0; i < rows; i++)
	{

		for (unsigned int j = 0; j < columns; j++)
		{
			unsigned int a_index = i * columns + j;
			int b_index = (j)*rows + (rows - i - 1);
			newdata[b_index] = a.data()[a_index];
		}
	}

	return Tensor({ columns,rows }, newdata);

}

//Rotate

void recRotate(Tensor& first, Tensor* finalT, vector<size_t>slice = {}, int currdim = 0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recRotate(first, finalT, newslice, currdim + 1);

		}
	}
	//If we have already reached the penultimate depth
	else
	{
		Tensor c = quickRotate(first);
		finalT->set(slice, c);
	}

}

Tensor Tensor::rotate()
{
	if (dim() < 2)
	{
		throw invalid_argument("Cannot Find transpose of this tensor. Must have atleast 2 dimensions");
	}
	else
	{
		vector<size_t>tshape = _shape;
		size_t temp = tshape[tshape.size() - 1];
		tshape[tshape.size() - 1] = tshape[tshape.size() - 2];
		tshape[tshape.size() - 2] = temp;

		Tensor result(tshape, 0);
		recRotate(*this, &result);
		return result;
	}

}
