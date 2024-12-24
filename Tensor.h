#pragma once
#include<iostream>
#include<vector>
#include<type_traits>
#include<typeinfo>
#include<initializer_list>
#include<string>
#include<algorithm>


namespace TPP
{
	/*Class definition for the tensor class*/
	class Tensor
	{
	private:
		std::vector<long double>_data;//The data stored in our tensor
		std::vector<size_t>_shape;//Shape of our tensor

		
		//Private function to recursively convert to a Tensor
		template<typename T>
		bool convertToTensor(const T& input,bool shapeKnown = false) 
		{
			if constexpr (std::is_floating_point<T>::value) 
			{
				_data.push_back(input);
				return true;
				
			}
			else 
			{
				//If size has not been assigned yet			
				if (!shapeKnown) 
				{
					_shape.push_back(input.size());
				}

				//For each component in nested vector
				bool isShapeKnown = shapeKnown;
				for (const auto& elem : input) 
				{
					isShapeKnown = convertToTensor(elem, isShapeKnown);
					
				}
				return isShapeKnown;
			}
		}
		//Gives flattened index 
		size_t getFlatIndex(std::vector<size_t> index);
		

	public:
		
		//Constructor
		template<typename T>
		Tensor(const T& input) 
		{
		
			convertToTensor(input);
		}
		
		//When shape and raw data are known
		Tensor(std::vector<size_t>shape,std::vector<long double>data);

		//When we want to initialize to a particular shape and a constant value
		Tensor(std::vector<size_t>shape, long double value);

		//Get shape of the data
		std::vector<size_t> shape();

		//Get raw data of the tensor
		std::vector<long double>data();


		//Assignment Operator
		void operator=(Tensor second);

		//Equality operator
	
		bool operator==(Tensor input);
		
		//Addition and subtraction
	
		Tensor operator+(Tensor input);

		
		void operator+=(Tensor input);

		Tensor operator-(Tensor input);

		
		void operator-=(Tensor input);

		//Hadamard (Element wise) multiplication
	
		Tensor operator%(Tensor second);
		
		
		void operator%=(Tensor second);

		//Getting an element
		Tensor at(std::initializer_list<size_t>pos);
		Tensor at(std::vector<size_t>pos);

		//Setting an element
		void set(std::initializer_list<size_t>pos, Tensor val);
		void set(std::vector<size_t>pos, Tensor val);

		//Batch Matrix multiplication 
		Tensor operator*(Tensor second);
		void operator*=(Tensor second);
		Tensor matMul(Tensor second);

		//Returns dimension of a tensor
		unsigned int dim();

		//Returns the shape of the tensor in string form
		std::string shapeString();

		//Gives the transpose of the tensor
		Tensor transpose();

		//Printing the tensor
		friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
		
	};

	//Some helper functions to define scalars, vectors and matrices
	Tensor Scalar(long double val);

	Tensor Vector(std::initializer_list<long double>data);

	Tensor Matrix(std::initializer_list<std::initializer_list<long double>>data);



}