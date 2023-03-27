import math

#these functions are general purpose for this code
def clean_row(vector_input):
    #this function takes a user input vector as a comma seperated string and converts it to a list while cleaning spaces.
    #when this is used to create a row vector, ensure that it is initialized inside a set of brackets to format it as a vector/matrix
    #TODO: error handling and pass "" to main
    vector = vector_input.split(",")
    i = 0
    while i < len(vector):
        vector[i] = float(vector[i].strip())
        i = i + 1
    return vector

def matrix_maker():
    #this function interactively sets up a matrix
    #currently, this function does error checking for positive values
    #TODO: error handling for nonints, empty string passes to main
    #here, the dimensionality is established
    dimensions_done = False
    while(dimensions_done == False):

        #formats the dimensions
        dimension_string = input("   What is the dimensionality of your matrix?\n   Enter as <#rows>,<#columns>.\n>")
        dimensions = dimension_string.split(",")
        if len(dimensions) == 2:
            i = 0
            while i < 2:
                dimensions[i] = int(dimensions[i].strip())
                i = i + 1

            #checks the dimensions for sanity
            if dimensions[0] > 0 and dimensions[1] > 0:
                dimensions_done = True
            else:
                print("   A matrix's dimensions must be positive - try again")
        else:
            print("   A matrix has two dimensions - try again")

    #here, the rows of the matrix are input one at a time
    matrix = []
    while len(matrix) < dimensions[0]:
        row = clean_row(input(f"   Input the values for row {len(matrix)+1} as a comma seperated list without brackets.\n>"))
        if len(row) == dimensions[1]:
            matrix.append(row)
        else:
            print("Row was not the expected size. Try again")
    
    return matrix

def get_dimensions(matrix):
    #returns a tuple describing the dimensionality of a matrix
    return(len(matrix),len(matrix[0]))

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

def output(input):
    #this function formats an input as an output consistent with this application's standards
    #for a single line output, input should be a string
    #for a multi line output, input should be a list
    #This function prints matrices with the interior brackets, which I find desirable
    print("\n   ********************************************************************************")
    if type(input) is str:
        print("   " + input)
    elif type(input) is list:
        for line in input:
            print("   " + str(line))
    print("   ********************************************************************************\n")

#These functions are used for the row echelon function(s)
def num_zeros(row):
    i = 0
    still_zero = True
    while still_zero == True and i < len(row):
        if row[i] == 0:
            i = i + 1
        else:
            still_zero = False
    return(i)

def row_multiply(row, scalar):
    #for multiplying a row of a matrix - not the same as scalar_matrix_multiply due to different number of brackets
    new_row = []
    i = 0
    while i < len(row):
        new_row.append(row[i] * scalar)
        i = i + 1
    return(new_row)

def clean_negs(row):
    #multiplies a row by -1 if the leading nonzero term is negative
    i = num_zeros(row)
    if i < len(row):
        if row[i] < 0:
            row = row_multiply(row, -1)
    return(row)

def ref(matrix):
    #This function converts a matrix to row echelon form
    dimensions = get_dimensions(matrix)
    
    #cleans possible leading negative values, then sorts matrix
    i = 0
    while i < dimensions[0]:
        matrix[i] = clean_negs(matrix[i])
        i = i + 1
    matrix.sort(reverse = True)

    #subtracts row i from row i-1 after multiplying both to least common multiple of first element
    #this changes the leading element of row i to a 0
    #does not change row i-1
    i = 1
    while i < dimensions[0]:
        while (num_zeros(matrix[i-1]) == num_zeros(matrix[i]) and num_zeros(matrix[i]) < dimensions[1]):
            j = num_zeros(matrix[i])
            coefficients = (matrix[i-1][j],matrix[i][j])
            k = 0
            while k < dimensions[1]:
                matrix[i][k] = matrix[i][k] * coefficients[0] - matrix[i-1][k] * coefficients[1]
                k = k + 1

            #cleans possible leading negatives, then resorts matrix
            matrix[i] = clean_negs(matrix[i])
            matrix.sort(reverse = True)
        i = i + 1

    return(matrix)

#These functions are, thus far, merely subsets of the determinant solver
def two_by_two_det(matrix):
    #this function finds the determinant of a 2x2 matrix
    return(matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1])

def det_decomp(matrix):
    #TODO: incorporate row_multiply

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

#and these are the functions that you find as options in the menu
def magnitude(vector):
    #finds the magnitude of a vector
    #TODO error handling, empty string pass to main
    dimensions = get_dimensions(vector)
    sum_of_squares = 0

    if dimensions[0] == 1:
        for element in vector[0]:
            sum_of_squares = sum_of_squares + element * element
    elif dimensions[1] == 1:
        for element in vector:
            sum_of_squares = sum_of_squares + element[0] * element[0]
    
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
    #TODO: incorporate row_multiply
    #TODO: error handling for scalar

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
    #TODO:
    #figure out if dimensions are backwards from convention
    #multiplies two matrices
    #note that this is used whenever dot products are desired
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
     
def find_angle(vector1, vector2):
    #finds the angle of two vectors in radians
    #returns None in the if either vector is the zero vector
    #this uses a dot product in the form of vector multiplication
    if magnitude(vector1) != 0 and magnitude(vector2) != 0:
        Tvector2 = transpose(vector2)
        return(math.acos(matrix_matrix_multiply(vector1,Tvector2)[0][0]/(magnitude(vector1) * magnitude(Tvector2))))
    else:
        return(None)

def cross_product(vector1, vector2):
    #returns the cross product of two vectors
    #if the vectors are not in 3D space, returns None
    #there's probably a cooler way to code this, but the cross product can't be generalized, so I won't worry about it

    if len(vector1[0]) == 3 and len(vector2[0]) == 3:
        cross_vector = [[]]
        cross_vector[0].append(vector1[0][1] * vector2[0][2] - vector1[0][2] * vector2[0][1])
        cross_vector[0].append(vector1[0][2] * vector2[0][0] - vector1[0][0] * vector2[0][2])
        cross_vector[0].append(vector1[0][0] * vector2[0][1] - vector1[0][1] * vector2[0][0])
        return(cross_vector)
    else:
        return(None)

def determinant_solver(determinant):
    #finds the determinant of a square matrix
    #TODO: Error handling for 1x1 matrix

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

def rref(matrix):
    dimensions = get_dimensions(matrix)
    matrix = ref(matrix)

    #finds horizontal indices of pivot points
    pivots = []
    for row in matrix:
        if num_zeros(row) < dimensions[1]:
            pivots.append(num_zeros(row))

    #pivot_number refers to which pivot point
    #pivot number 0 is skipped because there are no values above it
    pivot_number = 1
    while pivot_number < len(pivots) and pivots[pivot_number] < dimensions[0]:
        #i refers to what row number being modified
        i = 0
        while i < pivot_number:
            coefficients = (matrix[i][pivots[pivot_number]], matrix[pivot_number][pivots[pivot_number]])
            #j refers to the element within the row
            j = 0
            while j < dimensions[1]:
                matrix[i][j] = matrix[i][j] * coefficients[1] - matrix[pivot_number][j] * coefficients[0]
                j = j + 1
            i = i + 1
        pivot_number = pivot_number + 1


    #divides each row by the leading non-zero term
    i = 0
    while i < dimensions[0] and num_zeros(matrix[i]) < dimensions[1]:
        factor = 1/matrix[i][num_zeros(matrix[i])]
        matrix[i] = row_multiply(matrix[i], factor)
        i = i + 1

    return(matrix)

#flags for various states
degree_switch = True
info = False
state = "main menu"

print("   Welcome to the interactive linear algebra application!")

#interactive session
while True:

    if info == False:
        if state == "main menu":
            #interactive text of the menu
            print("   Please select an option from the following menu by typing a number and pressing enter.")
            print(
"""
   0: exit
   1: vector operations
   2: matrix operations
""")
            #changes menu to reflect flags
            if degree_switch == True:
                print('   You are currently working in degrees. To switch to radians, enter "a".')
            else:
                print('   You are currently working in radians. To switch to degrees, enter "a".')
            print('   To enter information mode, enter "i".')

            #user input and response
            choice = input("\n>")

            if choice == "0":
                exit()

            elif choice == "1":
                state = "vectors"

            elif choice == "2":
                state = "matrices"

            elif choice == "-1":
                print("this is the current test code. You should already know what it does")

            elif choice == "a":
                degree_switch = not degree_switch
            
            elif choice == "i":
                info = not info

        elif state == "vectors":
            #interactive text of the vector menu
            print("   Please select a vector operation.")
            print(
"""
   0: return
   1: vector magnitude
   2: vector addition
   3: multiply vector by scalar
   4: dot product
   5: find angle between vectors
   6: cross product
   """)
            print(
"""
   Note that if a vector transformation can also be generally
   applied to matrices, you will find it in the matrix section.
""")

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #vector magnitude
                vector = [clean_row(input("   Input a vector as a comma seperated list without brackets:\n>"))]
                output(f"The magnitude of {vector[0]} is {magnitude(vector)}.")

            elif choice == "2":
                #vector addition
                #this is an intentionally redundant function which uses matrix addition
                #TODO: Add error handling for wrong dimensions
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                vector_sum = matrix_add(vector1,vector2)
                output([vector1[0],"+",vector2[0],"=",vector_sum[0]])

            elif choice == "3":
                #multiply vector by scalar
                #this is an intentionally redundant function which uses scalar-matrix multiplication
                vector = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                scalar = float(input("   Input scalar\n>"))
                new_vector = scalar_matrix_multiply(scalar, vector)
                output([scalar,"*",vector[0],"=",new_vector[0]])

            elif choice == "4":
                #dot product
                #this is an intentionally redundant function which uses matrix-matrix multiplication
                #TODO: Add error handling
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                dot_product = matrix_matrix_multiply(vector2,transpose(vector1))[0][0]
                output(["the dot product of",vector1[0],"and",vector2[0],"is",dot_product])

            elif choice == "5":
                #angle of two vectors
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                if len(vector1[0]) == len(vector2[0]):
                    answer = find_angle(vector1, vector2)
                    if answer != None:
                        if degree_switch == True:
                            answer = answer/math.pi*180
                        output(f"The angle between vector 1 and vector 2 is {answer}.")
                    else:
                        output("The angle of any vector with the zero vector is not defined")
                else:
                    output("The angle between two vectors is only defined if they have the same number of elements")
            
            elif choice == "6":
                #cross product
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                answer = cross_product(vector1, vector2)[0]
                if answer == None:
                    output("Error: The cross product is only defined in three dimensional space")
                else:
                    output(f"The cross product of vector 1 and vector 2 is {answer}")

        elif state == "matrices":
            #interactive text of the matrix menu
            print("   Please select a matrix operation.")
            print(
"""
   0: return
   1: matrix transpose
   2: matrix addition
   3: multiply matrix by scalar
   4: multiply matrix by matrix
   5: matrix determinant
   6: reduced row echelon
""")  

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #matrix transpose
                matrix = matrix_maker()
                output(["The transpose of"] + matrix + ["is"]+ transpose(matrix))

            elif choice == "2":
                #add matrices
                print("   For matrix 1:")
                matrix1 = matrix_maker()
                print("   for matrix 2:")
                matrix2 = matrix_maker()
                
                if get_dimensions(matrix1) == get_dimensions(matrix2):
                    output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 + M2 ="] + matrix_add(matrix1,matrix2))
                else:
                    output("Matrix addition is not supported for matrices of different dimensionality")

                
            elif choice == "3":
                #multiply matrix by scalar
                matrix = matrix_maker()
                scalar = float(input("   Input scalar:\n>"))
                output([scalar,"*"] + matrix + ["="] + scalar_matrix_multiply(scalar, matrix))
            
            elif choice == "4":
                #mutliply matrix by matrix
                print("   For matrix 1:")
                matrix1 = matrix_maker()
                print("   for matrix 2:")
                matrix2 = matrix_maker()

                if get_dimensions(matrix1)[1] == get_dimensions(matrix2)[0]:
                    output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(matrix1,matrix2))

                elif get_dimensions(matrix1)[0] == get_dimensions(matrix2)[0] and get_dimensions(matrix1)[1] == 1 and get_dimensions(matrix2)[1] == 1:
                    if input("   The product of two column vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                        Tmatrix1 = transpose(matrix1)
                        output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(Tmatrix1,matrix2)[0][0])
                
                elif get_dimensions(matrix1)[1] == get_dimensions(matrix2)[1] and get_dimensions(matrix1)[0] == 1 and get_dimensions(matrix2)[0] == 1:
                    if input("   The product of two row vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                        Tmatrix2 = transpose(matrix2)
                        output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(matrix1,Tmatrix2)[0][0])
                
                else:
                    output("The product of two matrices of dimensions axb and cxd is not defined when b != c.")

            elif choice == "5":

                matrix = matrix_maker()
                dimensions = get_dimensions(matrix)

                if dimensions[0] == dimensions[1]:
                    output(["The determinant of"] + matrix + ["is"] + [str(determinant_solver(matrix))])
                else:
                    output("Determinants are only defined for square matrices.")

            elif choice == "6":
                #put matrix reduced row echelon
                matrix = matrix_maker()
                output(["The reduced row echelon form of"] + matrix + ["is"] + rref(matrix))

    else:
        if state == "main menu":
            #interactive text of the menu
            print("   Please select an option from the following menu by typing a number and pressing enter.")
            print(
"""
   0: exit
   1: vector operations
   2: matrix operations
""")
            if info == True:
                print('   You are in information mode. To exit information mode, enter "i".')

            #user input and response
            choice = input("\n>")

            if choice == "0":
                exit()

            elif choice == "1":
                state = "vectors"

            elif choice == "2":
                state = "matrices"
            
            elif choice == "i":
                info = not info

        elif state == "vectors":
            #interactive text of the vector menu
            print("   You are in info mode. Please select a vector operation to learn more about it.")
            print(
"""
   0: return
   1: vector magnitude
   2: vector addition
   3: multiply vector by scalar
   4: dot product
   5: find angle between vectors
   6: cross product
""")
            
            print(
"""
   Note that if a vector transformation can also be generally
   applied to matrices, you will find it in the matrix section.
""")

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #vector magnitude
                output([
                    "The magnitude of a vector is the length of the vector,",
                    "which can be calculated for a vector of any dimension",
                    "The magnitude of a vector A with n dimensions is",
                    "",
                    "|A| = sqrt(a1^2 + a2^2 + a3^2 + ... an^2"
                    ])

            elif choice == "2":
                #vector addition
                #this is an intentionally redundant function which uses matrix addition
                output([
                    "For vectors A and B which both have n dimensions",
                     "",
                    "A = [a1, a2, a3, ... an]   B = [b1, b2, b3, ... bn]",
                    "",
                    "vector addition is defined as",
                    "",
                    "A + B = [a1 + b1, a2 + b2, a3 + b3, ... an + bn]",
                    "",
                    "Note that vector addition is not defined for vectors of different length.",
                    "Vector addition is commutative e.g. a + b = b + a",
                    "Vector addition is associative e.g. (a + b) + c = a + (b + c)",
                    "One can visualize vector addition by placing the beginning of vector B at the end of vector A.",
                    "The vector which connects the beginning of A to the end of B is A + B."
                    ])

            elif choice == "3":
                #multiply vector by scalar
                #this is an intentionally redundant function which uses scalar-matrix multiplication
                output([
                    "The multiplication of a vector A",
                    "",
                    "A = [a1, a2, a3, ... an]",
                    "",
                    "by a scalar b is defined as",
                    "",
                    "bA = [ba1, ba2, ba3, ... ban]",
                    "",
                    "Scalar-Vector multiplication is associative e.g. b(cA) = (bc)A",
                    "Scalar-Vector multiplication is commutative e.g. bA = Ab",
                    "Scalar-Vector multiplication is distributive e.g. (b+c)A = bA + cA and c(A + B) = cA + cB",
                    "Scalar-Vector multiplication can be conceptualized as transforming vector A by elongating it by a factor of b.",
                    "If b is negative, it reverses the direction of A and also multiplies its length by a factor of |b|."
                    ])

            elif choice == "4":
                #dot product
                #this is an intentionally redundant function which uses matrix-matrix multiplication
                output([
                    "For vectors A and B",
                    "",
                    "A = [a1, a2, a3, ... an]   B = [b1, b2, b3, ... bn]",
                    "",
                    "the dot product is defined as"
                    "",
                    "A * B = [a1b1 + a2b2 + a3b3 + ... anbn]",
                    "",
                    "where the dot product is not defined for vectors of different length.",
                    "The dot product also satisfies the equation",
                    "",
                    "A * B = |A||B|cos(theta)",
                    "",
                    "where theta is the angle between the two vectors."
                    "One can consider the dot product as a special case of more general matrix multiplication"
                    ])

            elif choice == "5":
                #angle of two vectors
                output("The angle between two vectors is determined using the two definitions of the dot product")
            
            elif choice == "6":
                #cross product
                output([
                    "For two vectors A and B in R3",
                    "",
                    "A = [a1, a2, a3]   B = [b1, b2, b3]",
                    "",
                    "the cross product",
                    "",
                    "AxB = [a2b3-a3b2, -(a1b3-a3b1), a1b2-a2b1]",
                    "",
                    "The cross product AxB is always perpendicular to A and B.",
                    "and the direction of the cross product is determined by the right hand rule.",
                    "Furthermore,",
                    "",
                    "|AxB| = |A||B|sin(theta)",
                    ])

        elif state == "matrices":
            #interactive text of the matrix menu
            print("   You are in info mode. Please select a matrix operation to learn more about it.")
            print(
"""
   0: return
   1: matrix transpose
   2: matrix addition
   3: multiply matrix by scalar
   4: multiply matrix by matrix
   5: matrix determinant
   6: reduced row echelon
""")  

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #transpose matrix
                output([
                    "The Transpose T of an nxm matrix A is defined as",
                    "",
                    "       [a11 a12 a13 ... a1m]       [a11 a21 a31 ... an1]",
                    "       [a21 a22 a23 ... a2m]       [a12 b22 b32 ... an2]",
                    "T(A) = [a31 a32 a33 ... a3m]   =   [a13 a23 a33 ... an3]",
                    "       [ .   .   .  ...  . ]       [ .   .   .  ...  . ]",
                    "       [an1 an2 an3 ... anm]       [a1m a2m a3m ... anm]",
                    "",
                    "Note that this transforms an nxm matrix into an mxn matrix.",
                    "This operation is used as an intermediate step in many important operations."])

            elif choice == "2":
                #add matrices
                output([
                    "For matrices A and B, which both nxm dimensions:",
                    "",
                    "       [a11 a12 a13 ... a1m]       [b11 b12 b13 ... b1m]",
                    "       [a21 a22 a23 ... a2m]       [b21 b22 b23 ... b2m]",
                    "   A = [a31 a32 a23 ... a3m]   B = [b31 b32 b33 ... b3m]",
                    "       [ .   .   .  ...  . ]       [ .   .   .  ...  . ]",
                    "       [an1 an2 an3 ... anm]       [bn1 bn2 bn3 ... bnm]",
                    "", 
                    "matrix addition is defined as",
                    "",
                    "        [a11+b11 a12+b12 a13+b13 ... a1m+b1m]",
                    "        [a21+b21 a22+b22 a23+b23 ... a2m+b2m]",
                    "A + B = [a31+b31 a32+b32 a33+b33 ... a3m+b3m]",
                    "        [   .       .       .    ...    .   ]",
                    "        [an1+bn1 an2+bn2 an3+bn3 ... anm+bnm]",
                    "",
                    "Note that matrix addition is not defined for matrices of different dimensions",
                    "Matrix addition is commutative e.g. A + B = B + A",
                    "Matrix addition is associative e.g. A + (B + C) = (A + B) + C"
                    ])
                
            elif choice == "3":
                #multiply matrix by scalar
                output([
                "For a matrix A",
                "",
                "       [a11 a12 a13 ... a1m]",
                "       [a21 a22 a23 ... a2m]",
                "   A = [a31 a32 a33 ... a3m]",
                "       [ .   .   .  ...  . ]",
                "       [an1 an2 an3 ... anm]",
                "",
                "the multiplication of A by a scalar b is defined as",
                "",
                "       [ba11 ba12 ba13 ... ba1m]",
                "       [ba21 ba22 ba23 ... ba2m]",
                "  bA = [ba31 ba32 ba33 ... ba3m]",
                "       [  .    .    .  ...   . ]",
                "       [ban1 ban2 ban3 ... banm]",
                "",
                "Scalar-Matrix multiplication is associative e.g. b(cA) = (bc)A",
                "Scalar-Matrix multiplication is commutative e.g. bA = Ab"
                ])
            
            elif choice == "4":
                #mutliply matrix by matrix
                output([
                    "For matrix A, which has dimensions of lxm, and matrix B, which has dimensions of mxn:",
                    "",
                    "       [a11 a12 a13 ... a1m]       [b11 b12 b13 ... b1n]",
                    "       [a21 a22 a23 ... a2m]       [b21 b22 b23 ... b2n]",
                    "   A = [a31 a32 a23 ... a3m]   B = [b31 b32 b33 ... b3n]",
                    "       [ .   .   .  ...  . ]       [ .   .   .  ...  . ]",
                    "       [al1 al2 al3 ... alm]       [bm1 bm2 bm3 ... bmn]",
                    "", 
                    "matrix multiplication is defined as",
                    "",
                    "        [a11b11+a12b21+a13b31...+a1mbm1 a11b12+a12b22+a13b32...+a1mbm2 a11b13+a12b23+a13b33...+a1mbm3 ... a11b1n+a12b2n+a13b3n...+a1mbmn]",
                    "        [a21b11+a22b21+a23b31...+a2mbm1 a21b12+a22b22+a23b32...+a2mbm2 a21b13+a22b23+a23b33...+a2mbm3 ... a21b1n+a22b2n+a23b3n...+a2mbmn]",
                    "A * B = [a31b11+a32b21+a33b31...+a3mbm1 a31b12+a32b22+a33b32...+a3mbm2 a31b13+a32b23+a33b33...+a3mbm3 ... a31b1n+a32b2n+a33b3n...+a3mbmn]",
                    "        [             .                              .                              .                 ...              .                ]",
                    "        [al1b11+al2b21+al3b31...+almbm1 al1b12+al2b22+al3b32...+almbm2 al1b13+al2b23+al3b33...+almbm3 ... al1b1n+al2b2n+al3b3n...+almbmn]",
                    "",
                    "Note that A * B is a lxn matrix",
                    "Note also that matrix multiplication is only defined when the width of A and the height of B are be equal.",
                    "Another way to think of matrix multiplication is as an extension of dot products.",
                    "If matrix A is rewritten as a collection of row vectors R1-Rl",
                    "And matrix B is rewritten as a collection of column vectors C1-Cn",
                    "",
                    "       [R1]                            ",
                    "       [R2]                            ",
                    "   A = [R3]   B = [C1 C2 C3 ... Cn]",
                    "       [ .]                            ",
                    "       [Rl]                            ",
                    "",
                    "matrix multiplication can be described as",
                    "",
                    "        [R1*C1 R1*C2 R1*C3 ... R1*Cn]",
                    "        [R2*C1 R2*C2 R2*C3 ... R2*Cn]",
                    "A * B = [R3*C1 R3*C2 R3*C3 ... R3*Cn]",
                    "        [  .     .     .   ...   .  ]",
                    "        [Rl*C1 Rl*C2 Rl*C3 ... Rl*Cn]"
                    ])

            elif choice == "5":
                #determinant solver
                output("some stuff")

            elif choice == "6":
                output("some stuff")