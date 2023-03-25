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

def output(string):
    #formats outputs consistently
    print("\n   ********************************************************************************")
    print("   " + string)
    print("   ********************************************************************************\n")

def new_output(input):
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
            print("   Please select an option from the following menu by typing a number and pressing enter.\n")
            print("   0: exit\n   1: vector operations\n   2: matrix operations\n")
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
            print("   Please select a vector operation.\n")
            print("   0: return\n   1: vector magnitude\n   2: vector addition\n   3: multiply vector by scalar\n   4: dot product\n   5: find angle between vectors\n   6: cross product\n")
            print("   note that if a vector transformation can also be generally\n   applied to matrices, you will find it in the matrix section.")

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #vector magnitude
                vector = [clean_row(input("   Input a vector as a comma seperated list without brackets:\n>"))]
                new_output(f"The magnitude of {vector[0]} is {magnitude(vector)}.")

            elif choice == "2":
                #vector addition
                #this is an intentionally redundant function which uses matrix addition
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                vector_sum = matrix_add(vector1,vector2)
                new_output([vector1[0],"+",vector2[0],"=",vector_sum[0]])

            elif choice == "3":
                #multiply vector by scalar
                #this is an intentionally redundant function which uses scalar-matrix multiplication
                vector = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                scalar = float(input("   Input scalar\n>"))
                new_vector = scalar_matrix_multiply(scalar, vector)
                new_output([scalar,"*",vector[0],"=",new_vector[0]])

            elif choice == "4":
                #dot product
                #this is an intentionally redundant function which uses matrix-matrix multiplication
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                dot_product = matrix_matrix_multiply(vector2,transpose(vector1))[0][0]
                new_output(["the dot product of",vector1[0],"and",vector2[0],"is",dot_product])

            elif choice == "5":
                #angle of two vectors
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                if len(vector1[0]) == len(vector2[0]):
                    answer = find_angle(vector1, vector2)
                    if answer != None:
                        if degree_switch == True:
                            answer = answer/math.pi*180
                        new_output(f"The angle between vector 1 and vector 2 is {answer}.")
                    else:
                        new_output("The angle of any vector with the zero vector is not defined")
                else:
                    new_output("The angle between two vectors is only defined if they have the same number of elements")
            
            elif choice == "6":
                #cross product
                vector1 = [clean_row(input("   Input vector 1 as a comma seperated list without brackets:\n>"))]
                vector2 = [clean_row(input("   Input vector 2 as a comma seperated list without brackets:\n>"))]
                answer = cross_product(vector1, vector2)[0]
                if answer == None:
                    new_output("Error: The cross product is only defined in three dimensional space")
                else:
                    new_output(f"The cross product of vector 1 and vector 2 is {answer}")

        elif state == "matrices":
            #interactive text of the matrix menu
            print("   Please select a matrix operation.\n")
            print("   0: return\n   1: matrix transpose\n   2: matrix addition\n   3: multiply matrix by scalar\n   4: multiply matrix by matrix\n   5: matrix determinant\n   6: reduced row echelon\n")  

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #matrix transpose
                matrix = matrix_maker()
                new_output(["The transpose of"] + matrix + ["is"]+ transpose(matrix))

            elif choice == "2":
                #add matrices
                print("   For matrix 1:")
                matrix1 = matrix_maker()
                print("   for matrix 2:")
                matrix2 = matrix_maker()
                
                if get_dimensions(matrix1) == get_dimensions(matrix2):
                    new_output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 + M2 ="] + matrix_add(matrix1,matrix2))
                    #output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   M1 + M2 =\n{matrix_printer(matrix_add(matrix1,matrix2))}")
                else:
                    new_output("Matrix addition is not supported for matrices of different dimensionality")

                
            elif choice == "3":
                #multiply matrix by scalar
                matrix = matrix_maker()
                scalar = float(input("   Input scalar:\n>"))
                new_output([scalar,"*"] + matrix + ["="] + scalar_matrix_multiply(scalar, matrix))
                #output(f"{scalar} *\n{matrix_printer(matrix)}   =\n{matrix_printer(scalar_matrix_multiply(scalar, matrix))}")
            
            elif choice == "4":
                #mutliply matrix by matrix
                print("   For matrix 1:")
                matrix1 = matrix_maker()
                print("   for matrix 2:")
                matrix2 = matrix_maker()

                if get_dimensions(matrix1)[1] == get_dimensions(matrix2)[0]:
                    new_output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(matrix1,matrix2))
                    #output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   M1 * M2 =\n{matrix_printer(matrix_matrix_multiply(matrix1,matrix2))}")

                elif get_dimensions(matrix1)[0] == get_dimensions(matrix2)[0] and get_dimensions(matrix1)[1] == 1 and get_dimensions(matrix2)[1] == 1:
                    if input("   The product of two column vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                        Tmatrix1 = transpose(matrix1)
                        new_output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(Tmatrix1,matrix2)[0][0])
                        #output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   The dot product of M1 and M2 is {matrix_matrix_multiply(Tmatrix1,matrix2)[0][0]}")
                
                elif get_dimensions(matrix1)[1] == get_dimensions(matrix2)[1] and get_dimensions(matrix1)[0] == 1 and get_dimensions(matrix2)[0] == 1:
                    if input("   The product of two row vectors is not defined. Did you intend to find the dot product? (y/n)\n>") == "y":
                        Tmatrix2 = transpose(matrix2)
                        new_output(["M1 ="] + matrix1 + ["M2 ="] + matrix2 + ["M1 * M2 ="] + matrix_matrix_multiply(matrix1,Tmatrix2)[0][0])
                        #output(f"M1 =\n{matrix_printer(matrix1)}\n   M2 =\n{matrix_printer(matrix2)}\n   The dot product of M1 and M2 is {matrix_matrix_multiply(matrix1,Tmatrix2)[0][0]}")
                
                else:
                    output("The product of two matrices of dimensions axb and cxd is not defined when b != c.")

            elif choice == "5":
                #determinant solver
                
                matrix = matrix_maker()
                dimensions = get_dimensions(matrix)

                if dimensions[0] == dimensions[1]:
                    new_output(["The determinant of"] + matrix + ["is"] + [str(determinant_solver(matrix))])
                    #formats answer
                    """ answer = "the determinant of\n"
                    for row in matrix:
                        answer = answer + "   " + str(row) + "\n"
                    answer = answer + f"   is {determinant_solver(matrix)}"
                    output(answer) """

                else:
                    new_output("Determinants are only defined for square matrices.")

            elif choice == "6":
                #put matrix reduced row echelon
                matrix = matrix_maker()
                new_output(["The reduced row echelon form of"] + matrix + ["is"] + rref(matrix))
                #output(f"The reduced row echelon form of\n{matrix_printer(matrix)}   is\n{matrix_printer(rref(matrix))}")

    else:
        if state == "main menu":
            #interactive text of the menu
            print("   Please select an option from the following menu by typing a number and pressing enter.\n")
            print("   0: exit\n   1: vector operations\n   2: matrix operations\n")
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
            print("   You are in info mode. Please select a vector operation to learn more about it.\n")
            print("   0: return\n   1: vector magnitude\n   2: vector addition\n   3: multiply vector by scalar\n   4: dot product\n   5: find angle between vectors\n   6: cross product\n")
            print("   note that if a vector transformation can also be generally\n   applied to matrices, you will find it in the matrix section.")

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #vector magnitude
                output("stuff")

            elif choice == "2":
                #vector addition
                #this is an intentionally redundant function which uses matrix addition
                output("stuff")

            elif choice == "3":
                #multiply vector by scalar
                #this is an intentionally redundant function which uses scalar-matrix multiplication
                output("stuff")

            elif choice == "4":
                #dot product
                #this is an intentionally redundant function which uses matrix-matrix multiplication
                output("stuff")

            elif choice == "5":
                #angle of two vectors
                output("stuff")
            
            elif choice == "6":
                #cross product
                output("stuff")

        elif state == "matrices":
            #interactive text of the matrix menu
            print("   You are in info mode. Please select a matrix operation to learn more about it.\n")
            print("   0: return\n   1: matrix transpose\n   2: matrix addition\n   3: multiply matrix by scalar\n   4: multiply matrix by matrix\n   5: matrix determinant\n   6: reduced row echelon\n")  

            #User input and response
            choice = input(">")
        
            if choice == "0":
                state = "main menu"

            elif choice == "1":
                #transpose matrix
                output("some stuff")

            elif choice == "2":
                #add matrices
                output("some stuff")
                
            elif choice == "3":
                #multiply matrix by scalar
                output("some stuff")
            
            elif choice == "4":
                #mutliply matrix by matrix
                output("some stuff")

            elif choice == "5":
                #determinant solver
                output("some stuff")

            elif choice == "6":
                output("some stuff")