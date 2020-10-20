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

## 1.5 Universal Functions

NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions”(ufunc). Within NumPy, these functions operate elementwise on an array, producing an array as output.
```python
B = np.arange(3)
print(B)
x = np.exp(B)
print(x)
x = np.sqrt(B)
C = np.array([2., -1., 4.])
x = np.add(B, C)
print(x)
```
```text
array([0, 1, 2])
array([1.        , 2.71828183, 7.3890561 ])
array([0.        , 1.        , 1.41421356])
array([2., 0., 6.])
```

## 1.6 Indexing, Slicing and Iterating

One-dimensional arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.
```python
a = np.arange(10)**3
print(a)
print(a[2])
print(a[2:5])
a[:6:2] = 1000
print(a)
print(a[ : :-1])
for i in a:
    print(i**(1/3.))
```
```text
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
8
array([ 8, 27, 64])
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])
9.999999999999998
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```
Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:
```python
def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)
print(b)
print(b[2, 3])
print(b[0:5, 1])                       # each row in the second column of b
print(b[:, 1])                         # equivalent to the previous example
print(b[1:3, :])                      # each column in the second and third row of b
```
```text
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
23
array([ 1, 11, 21, 31, 41])
array([ 1, 11, 21, 31, 41])
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```
When fewer indices are provided than the number of axes, the missing indices are considered complete slices:
```python
print(b[-1])                                  # the last row. Equivalent to b[-1,:]
```
```text
array([40, 41, 42, 43])
```
The expression within brackets in b[i] is treated as an i followed by as many instances of : as needed to represent the remaining axes. NumPy also allows you to write this using dots as b[i,...].
The dots (...) represent as many colons as needed to produce a complete indexing tuple. For example, if x is an array with 5 axes, then
x[1,2,...] is equivalent to x[1,2,:,:,:],
x[...,3] to x[:,:,:,:,3] and
x[4,...,5,:] to x[4,:,:,5,:].
```python
c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
print(c.shape)
print(c[1,...])                                   # same as c[1,:,:] or c[1]
print(c[...,2])                                   # same as c[:,:,2]
```
```text
(2, 2, 3)
array([[100, 101, 102],
       [110, 112, 113]])
array([[  2,  13],
       [102, 113]])
```
Iterating over multidimensional arrays is done with respect to the first axis:
```python
for row in b:
    print(row)
```
```text
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```
However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:
```python
for element in b.flat:
    print(element, end = ' ')
```
```text
0 1 2 3 10 11 12 13 20 21 22 23 30 31 32 33 40 41 42 43 
```

## 1.7 Shape Manipulation - Changing the shape of an array

An array has a shape given by the number of elements along each axis:
```python
a = np.floor(10*rg.random((3,4)))
print(a)
print(a.shape)
```
```text
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
(3, 4)
```
The shape of an array can be changed with various commands. Note that the following three commands all return a modified array, but do not change the original array:
```python
print(a.ravel())  # returns the array, flattened
print(a.reshape(6, 2))  # returns the array with a modified shape
print(a.T)  # returns the array, transposed
print(a.T.shape)
print(a.shape)
```
```text
array([3., 7., 3., 4., 1., 4., 2., 2., 7., 2., 4., 9.])
array([[3., 7.],
       [3., 4.],
       [1., 4.],
       [2., 2.],
       [7., 2.],
       [4., 9.]])
array([[3., 1., 7.],
       [7., 4., 2.],
       [3., 2., 4.],
       [4., 2., 9.]])
(4, 3)
(3, 4)
```
The order of the elements in the array resulting from ravel() is normally “C-style”, that is, the rightmost index “changes the fastest”, so the element after a[0,0] is a[0,1]. If the array is reshaped to some other shape, again the array is treated as “C-style”. NumPy normally creates arrays stored in this order, so ravel() will usually not need to copy its argument, but if the array was made by taking slices of another array or created with unusual options, it may need to be copied. The functions ravel() and reshape() can also be instructed, using an optional argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

The reshape function returns its argument with a modified shape, whereas the ndarray.resize method modifies the array itself:
```python
print(a)
a.resize((2, 6))
print(a)
```
```text
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
array([[3., 7., 3., 4., 1., 4.],
       [2., 2., 7., 2., 4., 9.]])
```
If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:
```python
print(a.reshape(3, -1))
```
```text
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
```

## 1.8 Shape Manipulation - Stacking together different arrays

Several arrays can be stacked together along different axes:
```python
a = np.floor(10*rg.random((2,2)))
print(a)
b = np.floor(10*rg.random((2,2)))
print(b)
x = np.vstack((a, b))
print(x)
x = np.hstack((a, b))
print(x)
```
```text
array([[9., 7.],
       [5., 2.]])
array([[1., 9.],
       [5., 1.]])
array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
```
The function column_stack stacks 1D arrays as columns into a 2D array. It is equivalent to hstack only for 2D arrays:
```python
from numpy import newaxis
x = np.column_stack((a, b))     # with 2D arrays
print(x)
a = np.array([4., 2.])
b = np.array([3., 8.])
x = np.column_stack((a, b))     # returns a 2D array
print(x)
x = np.hstack((a, b))           # the result is different
print(x)
x = a[:,newaxis]               # view `a` as a 2D column vector
print(x)
x = np.column_stack((a[:,newaxis],b[:,newaxis]))
print(x)
x = np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
print(x)
```
```text
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
array([[4., 3.],
       [2., 8.]])
array([4., 2., 3., 8.])
array([[4.],
       [2.]])
array([[4., 3.],
       [2., 8.]])
array([[4., 3.],
       [2., 8.]])
```
On the other hand, the function row_stack is equivalent to vstack for any input arrays. In fact, row_stack is an alias for vstack:
```python
x = np.column_stack is np.hstack
print(x)
x = np.row_stack is np.vstack
print(x)
```
```text
False
True
```
In general, for arrays with more than two dimensions, hstack stacks along their second axes, vstack stacks along their first axes, and concatenate allows for an optional arguments giving the number of the axis along which the concatenation should happen.

Note

In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals (“:”)
```python
x = np.r_[1:4,0,4]
print(x)
```
```text
array([1, 2, 3, 0, 4])
```
When used with arrays as arguments, r_ and c_ are similar to vstack and hstack in their default behavior, but allow for an optional argument giving the number of the axis along which to concatenate.

## 1.9 Shape Manipulation - Splitting one array into several smaller ones

Using hsplit, you can split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:
```python
a = np.floor(10*rg.random((2,12)))
print(a)
# Split a into 3
x = np.hsplit(a,3)
print(x)
# Split a after the third and the fourth column
x = np.hsplit(a,(3,4))
print(x)
```
```text
array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
       [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])
[array([[6., 7., 6., 9.],
       [8., 5., 5., 7.]]), array([[0., 5., 4., 0.],
       [1., 8., 6., 7.]]), array([[6., 8., 5., 2.],
       [1., 8., 1., 0.]])]
[array([[6., 7., 6.],
       [8., 5., 5.]]), array([[9.],
       [7.]]), array([[0., 5., 4., 0., 6., 8., 5., 2.],
       [1., 8., 6., 7., 1., 8., 1., 0.]])]
```
vsplit splits along the vertical axis, and array_split allows one to specify along which axis to split.

## 1.10 Copies and Views
When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not. This is often a source of confusion for beginners. There are three cases:

### No Copy at All

Simple assignments make no copy of objects or their data.
```python
>>> a = np.array([[ 0,  1,  2,  3],
...               [ 4,  5,  6,  7],
...               [ 8,  9, 10, 11]])
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
```
```text

```
Python passes mutable objects as references, so function calls make no copy.
```python
>>> def f(x):
...     print(id(x))
...
>>> id(a)                           # id is a unique identifier of an object
148293216  # may vary
>>> f(a)
148293216  # may vary

```
```text

```

### View or Shallow Copy

Different array objects can share the same data. The view method creates a new array object that looks at the same data.
```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c = c.reshape((2, 6))                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0, 4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```
```text

```
Slicing an array returns a view of it:
```python
>>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:, 1:3]"
>>> s[:] = 10           # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```
```text

```

### Deep Copy

The copy method makes a complete copy of the array and its data.
```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```
```text

```
Sometimes copy should be called after slicing if the original array is not required anymore. For example, suppose a is a huge intermediate result and the final result b only contains a small fraction of a, a deep copy should be made when constructing b with slicing:
```python
>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # the memory of ``a`` can be released.
```
```text

```
If b = a[:100] is used instead, a is referenced by b and will persist in memory even if del a is executed.

## 1.11 Linear Algebra

```python
>>> import numpy as np
>>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> print(a)
[[1. 2.]
 [3. 4.]]

>>> a.transpose()
array([[1., 3.],
       [2., 4.]])

>>> np.linalg.inv(a)
array([[-2. ,  1. ],
       [ 1.5, -0.5]])

>>> u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
>>> u
array([[1., 0.],
       [0., 1.]])
>>> j = np.array([[0.0, -1.0], [1.0, 0.0]])

>>> j @ j        # matrix product
array([[-1.,  0.],
       [ 0., -1.]])

>>> np.trace(u)  # trace
2.0

>>> y = np.array([[5.], [7.]])
>>> np.linalg.solve(a, y)
array([[-3.],
       [ 4.]])

>>> np.linalg.eig(j)
(array([0.+1.j, 0.-1.j]), array([[0.70710678+0.j        , 0.70710678-0.j        ],
       [0.        -0.70710678j, 0.        +0.70710678j]]))
```
```text

```
