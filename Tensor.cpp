#include"Tensor.h"

using namespace std;
using namespace TPP;

//Constructor
Tensor::Tensor(vector<size_t>shape, vector<long double>data) 
{
	_shape = shape;
	_data = data;
}
Tensor::Tensor(std::vector<size_t>shape, long double value) 
{
	_shape = shape;


	//Subtract 1 from each element in shape to get the raw size
	std::for_each(shape.begin(), shape.end(), [](size_t &x) {x = x - 1; });
	

	int raw_size = getFlatIndex(shape)+1;

	vector<long double>data;
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
vector<long double> Tensor::data() 
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
		vector<long double> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++) 
		{
			long double element = _data[i] + second._data[i];
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
		vector<long double> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			long double element = _data[i] - second._data[i];
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
		vector<long double> data2;

		//Append summed data
		for (unsigned int i = 0; i < _data.size(); i++)
		{
			long double element = _data[i] * second._data[i];
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
	vector<long double> _newdata;

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
	
	for (unsigned int i = start_flat; i <= end_flat; i++) 
	{
		_newdata.push_back(_data[i]);
	}
	
	
	return Tensor(_newshape,_newdata);
	
}
Tensor Tensor::at(std::vector<size_t>pos)
{
	int index = pos.size();
	//The new shape of the tensor
	vector<size_t> _newshape;
	//The new data
	vector<long double> _newdata;

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
	vector<long double> _newdata;

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
	vector<long double> _newdata;

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
Tensor TPP::Scalar(long double val) 
{
	vector<size_t>shape;
	return Tensor(shape, vector<long double>({ val }));
}

Tensor TPP::Vector(std::initializer_list<long double>data) 
{
	vector<size_t>shape;
	shape.push_back(data.size());
	return Tensor(shape, vector<long double>(data));
}

Tensor TPP::Matrix(std::initializer_list<std::initializer_list<long double>>data)
{
	vector<size_t>shape;
	shape.push_back(data.size());

	int columns = data.begin()->size();
	shape.push_back(columns);
	
	
	vector<long double> _data;
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


void recMultiply(Tensor first, Tensor second, Tensor* finalT, vector<size_t>slice = {}, int currdim=0)
{
	//If not the deepest level
	if (currdim < first.dim() - 2)
	{
		for (unsigned int i = 0; i < first.shape()[currdim]; i++)
		{
			vector<size_t>newslice = slice;
			newslice.push_back(i);
			recMultiply(first, second, finalT,newslice,currdim+1);

		}
	}
	//If we have already reached the penultimate depth
	else 
	{
		int rows = first.shape()[first.dim() - 2];
		int inter = first.shape()[first.dim() - 1];
		int cols = second.shape()[first.dim() - 1];

		//Multiplying the matrices
		for (int i = 0; i < rows; i++) 
		{
			for (int j = 0; j < cols; j++) 
			{
				long double val3 = 0;

				vector<size_t>final_slice = slice;
				final_slice.push_back(i);
				final_slice.push_back(j);
				for (int k = 0; k < inter; k++) 
				{
					vector<size_t>first_slice = slice;
					first_slice.push_back(i);
					first_slice.push_back(k);
					vector<size_t>second_slice = slice;
					second_slice.push_back(k);
					second_slice.push_back(j);
					vector<size_t>shape = first_slice;

					


					long double val1 = first.at(first_slice).data()[0];
					long double val2 = second.at(second_slice).data()[0];
					val3 += val1 * val2;
				}
				finalT->set(final_slice, Scalar(val3));

			}
		}
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
	if (_shape[_shape.size() - 2] != _shape[_shape.size() - 1] || _shape[_shape.size() - 1] != _shape[_shape.size() - 2]) 
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

//Transpose

void recTranspose(Tensor first, Tensor* finalT, vector<size_t>slice = {}, int currdim = 0)
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
		unsigned int rows = first.shape()[first.shape().size() - 2];
		unsigned int columns = first.shape()[first.shape().size() - 1];

		for (unsigned int i = 0; i < rows; i++) 
		{
			for (unsigned int j = 0; j < columns; j++) 
			{
				vector<size_t> initial = slice;
				vector<size_t> result = slice;

				//Initial will be at (,,i,j)
				initial.push_back(i);
				initial.push_back(j);
				//Final will be at (,,j,i)
				result.push_back(j);
				result.push_back(i);


				Tensor value = first.at(initial);
				finalT->set(result, value);


			}
		}
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
