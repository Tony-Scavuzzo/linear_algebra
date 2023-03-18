import math

#these functions are general purpose for this code
def get_dimensions(matrix):
    #returns a tuple describing the dimensionality of a matrix
    return(len(matrix),len(matrix[0]))

def is_vector(matrix):
    #determines if a matrix is a vector and if it is a vector, what type ("row" or "column")
    #Returns a tuple of the form (bool, "type")
    dimensions = get_dimensions(matrix)

    if dimensions[0] == 1:
        return (True, "row")

    elif dimensions[1] == 1:
        return (True, "column")

    else:
        return (False, None)

def is_square_matrix(matrix):
    #determines if a matrix is a square matrix and if it is, what the dimensionality is
    #returns a tuple of the form (bool, dimensions)

    dimensions = get_dimensions(matrix)

    if dimensions[0] == dimensions[1]:
        return (True, dimensions[0])

    else:
        return (False, None)

def matrix_maker():
    #this function interactively sets up a matrix
    #here, the dimensionality is established
    dimensions_done = False
    while(dimensions_done == False):
        dimension_string = input("   What is the dimensionality of your matrix?\n   Enter as <#rows>,<#columns>.\n>")
        dimensions = dimension_string.split(",")
        if len(dimensions) == 2:
            i = 0
            while i < 2:
                dimensions[i] = int(dimensions[i].strip())
                i = i + 1
            dimensions_done = True
        else:
            print("A matrix has two dimensions - try again")
        
    
    #here, the rows of the matrix are input one at a time
    matrix = []
    while len(matrix) < dimensions[0]:
        row = clean_vector(input(f"   Input the values for row {len(matrix)+1} as a comma seperated list without brackets.\n>"))
        if len(row) == dimensions[1]:
            matrix.append(row)
        else:
            print("Row was not the expected size. Try again")
    
    return matrix

""" def square_matrix_maker():
    #this function has been depreciated because I have generalized to matrix_maker()
    #this function interactively sets up a square mat1rix
    order = int(input("   What is the size of your square matrix?\n   Enter an integer.\n>"))
    matrix = []
    while len(matrix) < order:
        row = clean_vector(input(f"   Input the values for row {len(matrix)+1} as a comma seperated list without brackets.\n>"))
        if len(row) == order:
            matrix.append(row)
        else:
            print("Row was not the expected size. Try again")
    return(matrix) """

def clean_vector(vector_input):
    #this function takes a user input vector as a comma seperated string and converts it to a list while cleaning spaces.
    vector = vector_input.split(",")
    i = 0
    while i < len(vector):
        vector[i] = float(vector[i].strip())
        i = i + 1
    return vector

def check_vector_dim(vector1, vector2):
    #this function returns an error if two given vectors are of different dimension
    if len(vector1) != len(vector2):
        print("   Error: This operation is only defined for two vectors of equal order.")
        exit()

def matrix_printer(matrix):
    #formats a matrix as a string with line jumps
    outstring = ""
    i = 0
    if len(matrix) == 1:
        outstring = "   [" + str(matrix[i]) + "]\n"
    else:
        while i < len(matrix):
            if i == 0:
                outstring = outstring + "   [" + str(matrix[i]) + "\n"
            elif i == len(matrix) - 1:
                outstring = outstring + "    " + str(matrix[i]) + "]\n"
            else:
                outstring = outstring + "    " + str(matrix[i]) + "\n"
            i = i + 1
    return outstring

def output(string):
    #formats outputs consistently
    print("\n   ********************************************************************************")
    print("   " + string)
    print("   ********************************************************************************\n")

#These functions are, thus far, merely subsets of the determinant solver
def two_by_two_det(matrix):
    #this function finds the determinant of a 2x2 matrix
    return(matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1])

def det_decomp(matrix):
    #this function takes an [nxn] determinant as an argument
    #and simplifies it to c1[n-1xn-1]-c2[n-1,n-1]+c3[n-1,n-1]-c4[n-1,n-1]...cn[n-1,n-1]
    #it returns a list of n 2 element sublists where the first element is cn and the second element is [n-1,n-1]
    #e.g. [[c1, [n-1,n-1]], [-c2, [n-1,n-1]],[c3, [n-1,n-1]], [-c4, [n-1,n-1]]...[cn, [n-1,n-1]]]
    #this function accounts for the alternating sign of determinant simplification
    order = len(matrix)
    new_sum = []

    #creates list of items that will one day be summed up
    i = 0
    
    while i < order:
    
        #adds coefficient of each submatrix as element[0] in each sublist and sets to negative for every other submatrix
        new_sum.append([matrix[0][i],[]])
        if i % 2 != 0:
            new_sum[i][0] = -1*new_sum[i][0]
            
        #appends the submatrix as the second element in each sublist
        j = 1
        while j < order:
            new_sum[i][1].append([])
            k = 0
            while k < order:
                if k != i:
                    new_sum[i][1][j-1].append(matrix[j][k])
                k = k + 1
            j = j + 1
        i = i + 1
    return new_sum

def distributive(list1):
    #This function performs the distributive property of multiplication for the formula k(a[nxn]+b[nxn]+c[nxn]...) = ka[nxn]+kb[nxn]+kc[nxn]...
    #it assumes an input of the form [k,[[a,[nxn]],[b,[nxn]],[c,[nxn]]...]]
    #and returns the form [[ka,[nxn]],[kb,[nxn]],[kc,[nxn]]...]
    coefficient = list1[0]
    new_list = list1[1]
    i = 0
    while i < len(new_list):
        new_list[i][0] = coefficient * new_list[i][0]
        i = i + 1
    return(new_list)

#these functions are, thus far, merely subsets of the cross product function.
def index_loop(index, max):
    while index > max - 1:
        index = index - max
    return index

#and these are the functions that you find as options in the menu
def magnitude(vector):
    #finds the magnitude of a vector
    sum_of_squares= 0
    for element in vector:
        sum_of_squares = sum_of_squares + element * element
    return math.sqrt(sum_of_squares)

def transpose(matrix):
    dimensions = get_dimensions(matrix)
    new_matrix = []
    #indeces traverse new matrix, not old matrix
    i = 0
    while i < dimensions[1]:
        new_matrix.append([])
        j = 0
        while j < dimensions[0]:
            new_matrix[i].append(matrix[j][i])
            j = j + 1
        i = i + 1
    return(new_matrix)

def matrix_add(matrix1, matrix2):
    #adds two matrices together
    dimensions = get_dimensions(matrix1)
    matrix_sum = []
    i = 0
    while i < dimensions[0]:
        matrix_sum.append([])
        j = 0
        while j < dimensions[1]:
            matrix_sum[i].append(matrix1[i][j]+matrix2[i][j])
            j = j + 1
        i = i + 1
    return matrix_sum

def scalar_matrix_multiply(scalar, matrix):
    dimensions = get_dimensions(matrix)
    i = 0
    new_matrix = []
    while i < dimensions[0]:
        new_matrix.append([])
        j = 0
        while j < dimensions[1]:
            new_matrix[i].append(scalar * matrix[i][j])
            j = j + 1
        i = i + 1
    return(new_matrix)

def matrix_matrix_multiply(matrix1, matrix2):
    dimensions1 = get_dimensions(matrix1)
    dimensions2 = get_dimensions(matrix2)
    new_matrix = []
    #indices i and j traverse the elements of the new matrix
    i = 0
    while i < dimensions1[0]:
        new_matrix.append([])
        j = 0
        while j < dimensions2[1]:
            #index k traverses the original matrices
            element = 0
            k = 0
            while k < dimensions1[1]:
                element = element + matrix1[i][k] * matrix2[k][j]
                k = k + 1
            new_matrix[i].append(element)
            j = j + 1
        i = i + 1
    return(new_matrix)

def dot_product(vector1, vector2):
    #finds the dot product of two vectors
    dot_product_sum = 0
    i = 0
    while i < len(vector1):
        dot_product_sum = dot_product_sum + vector1[i]*vector2[i]
        i = i + 1
    return(dot_product_sum)
        
def find_angle(vector1, vector2):
    #finds the angle of two vectors in radians
    return(math.acos(dot_product(vector1, vector2)/(magnitude(vector1)*magnitude(vector2))))

def cross_product(vector1, vector2):
    #returns the cross product of two vectors
    #if the vectors are not in 3D space, returns None

    #creates empty cross product vector with same order as inputs
    order = len(vector1)
    if order == 3:
        cross_vector = []
        while len(cross_vector) < order:
            cross_vector.append("")
        
        #populates the cross vector
        i = 0
        while i < len(cross_vector):
            cross_vector[i] = vector1[index_loop(i+1, order)]*vector2[index_loop(i+2, order)] - vector1[index_loop(i+2, order)]*vector2[index_loop(i+1, order)]
            i = i + 1    
        return(cross_vector)
    else:
        return(None)

def determinant_solver(determinant):
    #finds the determinant of a square matrix

    #sets up the initial sum
    total = [[1, determinant]]

    #right now the total takes the form of [1, [nxn]]
    #or more clearly, 1*[nxn]
    #this loops through total and breaks down the determinant iteratively until every determinant is a 2x2
    while len(total[0][1]) > 2:
        new_total = []
        i = 0
        while i < len(total):
            scratch = total[i]
            scratch[1] = det_decomp(scratch[1])
            scratch = distributive(scratch)
            new_total = new_total + scratch
            i = i + 1
        total = new_total

    #now that the total takes the form of [[c1, [2x2]],[c2, [2x2]],[c3, [2x2]]...[cn, [2x2]]]
    #or more clearly: c1[2x2]+c2[2x2]+c3[2x2]...cn[2x2]
    final_sum = 0
    i = 0
    while i < len(total):
        #it is time to perform the two_by_two_det function on each 2x2
        #getting you [[c1,k1],[c2,k2],[c3,k3]...[cn,]]
        #or more clearly: c1k1+c2k2+c3k3...cnkn
        total[i][1] = two_by_two_det(total[i][1])
        
        #now we multiply cn by kn
        total[i] = total[i][0]*total[i][1]
        
        #and finally add the terms together
        final_sum = final_sum + total[i]
        i = i + 1
    return(final_sum)
 
#this flag is for working in degrees rather than radians 
degree_switch = True

print("   Welcome to the interactive linear algebra application!")

#interactive session
while True:

    #interactive text, includuing menu
    print("   Please select an operation from the following menu by typing a number and pressing enter.")
    print("   0: exit\n   1: vector magnitude\n   2: matrix transpose\n   3: matrix addition\n   4: multiply matrix by scalar\n   5: multiply matrix by matrix\n   6: find angle between vectors\n   7: cross product\n   8: matrix determinant\n   -1: test code")  
    #changes menu to reflect degrees/radians
    if degree_switch == True:
        print('\n   you are currently working in degrees. To switch to radians, enter "r."')
    else:
        print('\n   you are currently working in radians. To switch to degrees, enter "d."')

    #User input and response
    choice = input(">")
    
    if choice == "0":
        exit()

    elif choice == "1":
        #vector magnitude
        vector = clean_vector(input("   Input a vector as a comma seperated list without brackets:\n>"))
        output(f"The magnitude of {vector} is {magnitude(vector)}.")

    elif choice == "2":
        matrix = matrix_maker()
        output(f"The transpose of\n{matrix_printer(matrix)}   is\n{matrix_printer(transpose(matrix))}")

    elif choice == "3":
        #add matrices
        print("   For matrix 1:")
        matrix1 = matrix_maker()
        print("   for matrix 2:")
        matrix2 = matrix_maker()
        
        if get_dimensions(matrix1) == get_dimensions(matrix2):
            output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   M1 + M2 =\n{matrix_printer(matrix_add(matrix1,matrix2))}")
        else:
            output("Matrix addition is not supported for matrices of different dimensionality")
        #print(matrix_add(matrix1,matrix2))
        #print(matrix_printer(matrix_add(matrix1,matrix2)))
        
    elif choice == "4":
        #multiply matrix by scalar
        matrix = matrix_maker()
        scalar = float(input("   Input scalar:\n>"))
        output(f"{scalar} *\n{matrix_printer(matrix)}   =\n{matrix_printer(scalar_matrix_multiply(scalar, matrix))}")
    
    elif choice == "5":
        #mutliply matrix by matrix
        print("   For matrix 1:")
        matrix1 = matrix_maker()
        print("   for matrix 2:")
        matrix2 = matrix_maker()

        if get_dimensions(matrix1)[1] == get_dimensions(matrix2)[0]:
            output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   M1 * M2 =\n{matrix_printer(matrix_matrix_multiply(matrix1,matrix2))}")

        elif get_dimensions(matrix1)[0] == get_dimensions(matrix2)[0] and get_dimensions(matrix1)[1] == 1 and get_dimensions(matrix2)[1] == 1:
            if input("   The product of two column vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                Tmatrix1 = transpose(matrix1)
                output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   The dot product of M1 and M2 is {matrix_matrix_multiply(Tmatrix1,matrix2)[0][0]}")
        
        elif get_dimensions(matrix1)[1] == get_dimensions(matrix2)[1] and get_dimensions(matrix1)[0] == 1 and get_dimensions(matrix2)[0] == 1:
            if input("   The product of two row vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                Tmatrix2 = transpose(matrix2)
                output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   The dot product of M1 and M2 is {matrix_matrix_multiply(matrix1,Tmatrix2)[0][0]}")
        
        else:
            output("The product of two matrices of dimensions axb and cxd is not defined when b != c.")

    elif choice == "6":
        #angle of two vectors
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_vector_dim(vector1,vector2)
        answer = find_angle(vector1, vector2)
        if degree_switch == True:
            answer = answer/math.pi*180
        output(f"The angle between vector 1 and vector 2 is {answer}.")
        
    elif choice == "7":
        #cross product
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_vector_dim(vector1,vector2)
        answer = cross_product(vector1, vector2)
        if answer == None:
            output("Error: The cross product is only defined in three dimensional space")
        else:
            output(f"The cross product of vector 1 and vector 2 is {answer}")
    
    elif choice == "8":
        #determinant solver
        
        matrix = matrix_maker()
        dimensions = get_dimensions(matrix)

        if dimensions[0] == dimensions[1]:
            #formats answer
            answer = "the determinant of\n"
            for row in matrix:
                answer = answer + "   " + str(row) + "\n"
            answer = answer + f"   is {determinant_solver(matrix)}"
            output(answer)

        else:
            output("Determinants are only defined for square matrices.")

    elif choice == "-1":
        print("   This is the current test code.\n   If you are running this code, you should already know what it does.")
        matrix1 = matrix_maker()
        print(matrix_printer(transpose(matrix1)))
    
    elif choice == "r":
        degree_switch = False
    
    elif choice == "d":
        degree_switch = True
    