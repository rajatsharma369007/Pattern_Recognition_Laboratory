# creating function and classes in python

# calling a fucntion
def funct(var):
    print("Functional called : ", var)

var=input() 
funct(var)


# swapping two numbers
def swap(var1, var2):
    print("Numbers before swapping = %d, %d" %(var1, var2))
    var1, var2 = var2, var1
    print("Numbers after swapping = {}, {}".format(var1, var2))

var1 = int(input())
var2 = int(input())
swap(var1, var2)


# creating a class and a constructor
class student:
    count=0
    def __init__(self, name, rollno):
        self.name = name
        self.rollno = rollno
        student.count +=1
    def display(self):
        print("Name: ", self.name, end=', ')
        print("RollNo: ", self.rollno)
        
student1 = student("rajat", "CSB15059")
student2 = student("sharma", "CSB15050")

student1.display()
student2.display()


# Inheritance
class Parent:
    def __init__(self):
        print("From parent constructor")
    def display(self):
        print("Calling parent version of display")


class Child(Parent):
    def __init__(self):
        print("From child constructor")
    def display(self):
        print("Calling child version of display")    
        
child1 = Child()
child1.display()
parent1 = Parent()        
parent1.display()   


# creating a numpy array
import numpy as np

arr = np.array([[1,3,4,67],[1,3,4,67]])
print("first array : ",arr)

arr1 = np.array([3,6,3,7,9])
print("second array : ",arr1)

# c = arr+arr1    # this line will show error diffrent shape

print(arr1.ndim)
print(arr.shape)
print(arr.size)
print(type(arr))

# creating a sorting function 
for i in range(len(arr1)):
    for j in range(len(arr1+1)):
        if arr1[i] < arr1[j]:
            arr1[i], arr1[j] = arr1[j], arr1[i]
        else:
            continue
        
print(arr1)








