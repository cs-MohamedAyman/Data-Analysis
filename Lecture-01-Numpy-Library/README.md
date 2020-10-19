<img align="right" width="90" height="90" src="https://github.com/cs-MohamedAyman/Computer-Science-Textbooks/blob/master/logos/data-analysis.jpg">

# Lecture 01 - Numpy Library

## 1.1 Numpy Basics

NumPy’s main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy dimensions are called axes.

NumPy’s array class is called ndarray. It is also known by the alias array. Note that numpy.array is not the same as the Standard Python Library class array.array, which only handles one-dimensional arrays and offers less functionality. The more important attributes of an ndarray object are:

- ***ndarray.ndim***: the number of axes (dimensions) of the array.

- ***ndarray.shape***: the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, shape will be (n,m). The length of the shape tuple is therefore the number of axes, ndim.

- ***ndarray.size***: the total number of elements of the array. This is equal to the product of the elements of shape.

- ***ndarray.dtype***: an object describing the type of the elements in the array. One can create or specify dtype’s using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.

- ***ndarray.itemsize***: the size in bytes of each element of the array. For example, an array of elements of type float64 has itemsize 8 (=64/8), while one of type complex32 has itemsize 4 (=32/8). It is equivalent to ndarray.dtype.itemsize.

- ***ndarray.data***: the buffer containing the actual elements of the array. Normally, we won’t need to use this attribute because we will access the elements in an array using indexing facilities.

```python
import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))
b = np.array([6, 7, 8])
print(b)
print(type(b))
```
```text
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
(3, 5)
2
'int64'
8
15
<class 'numpy.ndarray'>
array([6, 7, 8])
<class 'numpy.ndarray'>
```

## 1.2 Array Creation

There are several ways to create arrays.
For example, you can create an array from a regular Python list or tuple using the array function. The type of the resulting array is deduced from the type of the elements in the sequences.
```python
import numpy as np
a = np.array([2,3,4])
print(a)
print(a.dtype)
b = np.array([1.2, 3.5, 5.1])
print(b.dtype)
```
```text
array([2, 3, 4])
dtype('int64')
dtype('float64')
```
array transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into three-dimensional arrays, and so on.
```python
b = np.array([(1.5,2,3), (4,5,6)])
print(b)
c = np.array([[1,2], [3,4]], dtype=complex)
print(c)
```
```text
array([[1.5, 2. , 3. ],
       [4. , 5. , 6. ]])
array([[1.+0.j, 2.+0.j],
       [3.+0.j, 4.+0.j]])
```
Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.

The function zeros creates an array full of zeros, the function ones creates an array full of ones, and the function empty creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64.
```python
x = np.zeros((3, 4))
print(x)
x = np.ones((2,3,4), dtype=np.int16)
print(x)
x = np.empty((2,3))
print(x)
```
```text
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],
       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],  # may vary
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```
To create sequences of numbers, NumPy provides the arange function which is analogous to the Python built-in range, but returns an array.
```python
x = np.arange(10, 30, 5)
print(x)
x = np.arange(0, 2, 0.3)
print(x)
```
```text
array([10, 15, 20, 25])
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
```
When arange is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function linspace that receives as an argument the number of elements that we want, instead of the step:
```python
import numpy as np
x = np.linspace(0, 2*np.pi, 5)
f = np.sin(x)
print(x)
print(f)
```
```text
[0.         1.57079633 3.14159265 4.71238898 6.28318531]
[ 0.0000000e+00  1.0000000e+00  1.2246468e-16 -1.0000000e+00 -2.4492936e-16]
```

## 1.3 Printing Arrays

When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:
the last axis is printed from left to right, the second-to-last is printed from top to bottom, the rest are also printed from top to bottom, with each slice separated from the next by an empty line.
One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.

```python
import numpy as np
a = np.arange(6)                         # 1d array
print(a)
b = np.arange(12).reshape(4,3)           # 2d array
print(b)
c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)
```
```text
[0 1 2 3 4 5]
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```
If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:
```python
import numpy as np
x = np.arange(10000)
print(x)
x = np.arange(10000).reshape(100,100)
print(x)
```
```text
[   0    1    2 ... 9997 9998 9999]
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```

## 1.4 Basic Operations

Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.
```python
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(b)
c = a - b
print(c)
print(b ** 2)
print(10 * np.sin(a))
print(a < 35)
```
```text
array([0, 1, 2, 3])
array([20, 29, 38, 47])
array([0, 1, 4, 9])
array([9.12945251, -9.88031624,  7.4511316 , -2.62374854])
array([True,  True, False, False])
```
Unlike in many matrix languages, the product operator * operates elementwise in NumPy arrays. The matrix product can be performed using the @ operator (in python >=3.5) or the dot function or method:
```python
A = np.array([[1,1],
              [0,1]])
B = np.array([[2,0],
              [3,4]])
x = A * B
print(x)
x = A @ B
print(x)
x = A.dot(B)
print(x)
```
```text
array([[2, 0],
       [0, 4]])
array([[5, 4],
       [3, 4]])
array([[5, 4],
       [3, 4]])
```
Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.
```python
rg = np.random.default_rng(1)
a = np.ones((2,3), dtype=int)
b = rg.random((2,3))
a *= 3
print(a)
b += a
print(b)
```
```text
array([[3, 3, 3],
       [3, 3, 3]])
array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])
```
When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).
```python
a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3)
print(b.dtype.name)
c = a + b
print(c)
print(c.dtype.name)
d = np.exp(c * 1j)
print(d)
print(d.dtype.name)
```
```text
'float64'
array([1.        , 2.57079633, 4.14159265])
'float64'
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j, 
       -0.54030231-0.84147098j])
'complex128'
```
Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ndarray class.
```python
a = rg.random((2,3))
print(a)
print(a.sum())
print(a.min())
print(a.max())
```
```text
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])
3.1057109529998157
0.027559113243068367
0.8277025938204418
```
By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:
```python
b = np.arange(12).reshape(3,4)
print(b)
x = b.sum(axis=0)                            # sum of each column
print(x)
x = b.min(axis=1)                            # min of each row
print(x)
x = b.cumsum(axis=1)                         # cumulative sum along each row
print(x)
```
```text
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
array([12, 15, 18, 21])
array([0, 4, 8])
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

## 1.5 Universal Functions¶





```python

```
```text

```







```python

```
```text

```






```python

```
```text

```
