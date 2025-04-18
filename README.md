![Logo v1](https://github.com/user-attachments/assets/392e5189-151d-4c8a-be3c-b3bca2bd19da)

# Tensor++
A simple c++ library to handle Tensors and Machine Learning in C++. Made for educational purposes for people who really want to know what happens when they import ML libraries. 


# Basic usage
To use Tensor++ just add all header and cpp files to your project. Then include them like this
```CPP
#include"TPP/Tensor.h"
#include"TPP/Layer.h"
#include"TPP/Network.h"
using namespace TPP
```

# Creating a Tensor
Creating tensors is easy... just do this
```CPP
Tensor T(vector<vector<vector<float>>>({{ {{{1,2},{1,2}}},{{1,2},{1,2}} }}));

```
This creates a 4D tensor... You can do this for any number of dimensions

Additionally we have some helper functions for common tensor types
## Scalars:
```CPP
Tensor T = Scalar(2);
```

## Vectors:
```CPP
Tensor T = Vector({ 1,2,3,4,5 });
```

## Matrices
```CPP
Tensor T = Matrix({ {1,2},{3,4},{5,6} });
```

Note for these functions you dont need to define a vector, we manage it for you :)


# Creating Neural Networks
The library also supports the following neural network layers
1) Dense
2) RELU
3) Leaky RELU
4) Tanh
5) Sigmoid
6) Softmax
7) Max Pooling
8) Convolution

Some additional features are also planned
1) RNNs


The syntax to define a neural network is easy, here is a sample network
```CPP
	Network n;
	float d = 0.001;
	n.alpha = d;
	n.addLayer(new CONV(2, 2, X[0].shape(), time(NULL),2));
	n.addLayer(new FLATTEN(n.getOutputShape()));
	
	
	n.addLayer(new DENSE(n.getOutputShape()[1], 10, time(NULL)));
	n.addLayer(new RELU(n.getOutputShape()));
	n.addLayer(new DENSE(n.getOutputShape()[1],10,time(NULL)));
	n.addLayer(new SOFTMAX(n.getOutputShape()));
```

Here n.alpha sets the learning rate and n.getOutputLayer() is used to get the output size of the neural network at the current moment which is used to define the shape of the tensors in the next layer

## Training the network
To train the network, use backpropagate()

```CPP
n.backpropagate(fb);
```

And when you need to update the parameters, call update()
```
n.update();
```

# Overall training example
```CPP
//Creating the neural network
Network n;
float d = 0.001;
n.alpha = d;
n.addLayer(new CONV(2, 2, X[0].shape(), time(NULL),2));
n.addLayer(new FLATTEN(n.getOutputShape()));


n.addLayer(new DENSE(n.getOutputShape()[1], 10, time(NULL)));
n.addLayer(new RELU(n.getOutputShape()));
n.addLayer(new DENSE(n.getOutputShape()[1],10,time(NULL)));
n.addLayer(new SOFTMAX(n.getOutputShape()));

int batch = 5;

//For bold driver
float double prev_loss = INFINITY;
Network prev = n;
for (int i = 0; i < 500; i++) 
{

	long double avg_loss = 0;
	for (int j = 0; j < X.size(); j++) 
	{
		
		Tensor in = X[j];
		Tensor act = Y[j];
		Tensor o;
	
		
		o = n.output(in);
	

		Tensor fb = (o - Y[j]);
		
		avg_loss += (0.5*(o-Y[j])%(o-Y[j])).sumOfElements();
	

		n.backpropagate(fb);

		if ((j + 1) % batch == 0) 
		{
			n.update();
		}
	}
	n.update();
	//Average loss
	avg_loss /= (long double)X.size();
	cout <<"Iteration "<<i<< " Average loss is" << avg_loss <<" n is "<<n.alpha<< endl;
	if (avg_loss > prev_loss) 
	{
		cout << "LOSS INCREASED, REVERTING" << endl;
		n = prev;
		n.alpha = n.alpha/2;
	}
	else 
	{
		
		n.alpha *= 1.05;
	}
	prev = n;

	prev_loss = avg_loss;
	
}

for (int i = 0; i < X.size(); i++) 
{
	cout << "PREDICTED" << endl;
	cout << n.output(X[i]) << endl;
	cout << "ACTUAL" << endl;
	cout << Y[i] << endl;
}

```
The above method uses the bold driver learning heuristic (Increase learning rate by 5% if loss decreases, halve it if loss increases)

The final result looks like this
![image](https://github.com/user-attachments/assets/09da6454-e281-4000-8049-5d4b8646cc87)


# Additional notes
Please feel free if you have any doubts, queries or suggestions. I'll be happy to respond. Also i'll be happy if you can report any bugs or improvements, for it means you've gone through my code
