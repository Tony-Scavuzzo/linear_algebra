import math

#these functions are general purpose for this code
def square_matrix_maker():
    #this function interactively sets up a square matrix
    order = int(input("   What is the size of your square matrix?\n   Enter an integer.\n>"))
    matrix = []
    while len(matrix) < order:
        row = clean_vector(input(f"   Input the values for row {len(matrix)+1} as a comma seperated list without brackets.\n>"))
        if len(row) == order:
            matrix.append(row)
        else:
            print("Row was not the expected size. Try again")
    return(matrix)

def clean_vector(vector_input):
    #this function takes a user input vector as a comma seperated string and converts it to a list while cleaning spaces.
    vector = vector_input.split(",")
    i = 0
    while i < len(vector):
        vector[i] = float(vector[i].strip())
        i = i + 1
    return vector

def check_dimensions(vector1, vector2):
    #this function returns an error if two given vectors are of different dimension
    if len(vector1) != len(vector2):
        print("   Error: This operation is only defined for two vectors of equal order.")
        exit()

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

def vector_add(vector1, vector2):
    #adds two vectors together
    vector_sum = []
    i = 0
    while i < len(vector1):
        vector_sum.append(vector1[i]+vector2[i])
        i = i + 1
    return vector_sum

def scalar_vector_multiply(scalar, vector):
    i = 0
    new_vector = []
    while i < len(vector):
        new_vector.append(scalar*vector[i])
        i = i + 1
    return(new_vector)

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
    print("   0: exit\n   1: vector magnitude\n   2: vector addition\n   3: multiply vector by scalar\n   4: dot product\n   5: find angle between vectors\n   6: cross product\n   7: matrix determinant")  
    #changes menu to reflect degrees/radians
    if degree_switch == True:
        print('\n   you are currently working in degrees. To switch to radians, enter "r."')
    else:
        print('\n   you are currently working in radians. To switch to degrees, enter "r."')

    #User input and response
    choice = input(">")
    
    if choice == "0":
        exit()

    elif choice == "1":
        #vector magnitude
        vector = clean_vector(input("   Input a vector as a comma seperated list without brackets:\n>"))
        output(f"The magnitude of {vector} is {magnitude(vector)}.")
        
    elif choice == "2":
        #add vectors
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_dimensions(vector1,vector2)
        output(f"The sum of {vector1} and {vector2} is {vector_add(vector1,vector2)}")
        
    elif choice == "3":
        #multiply vector by scalar
        vector = clean_vector(input("   Input vector as a comma seperated list without brackets:\n>"))
        scalar = float(input("   Input scalar:\n>"))
        output(f"{scalar} * {vector} = {scalar_vector_multiply(scalar, vector)}")
    
    elif choice == "4":
        #dot product
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_dimensions(vector1,vector2)
        output(f"The dot product of vector 1 and vector 2 is {dot_product(vector1, vector2)}.")
        
    elif choice == "5":
        #angle of two vectors
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_dimensions(vector1,vector2)
        answer = find_angle(vector1, vector2)
        if degree_switch == True:
            answer = answer/math.pi*180
        output(f"The angle between vector 1 and vector 2 is {answer}.")
        
    elif choice == "6":
        #cross product
        vector1 = clean_vector(input("   Input vector 1 as a comma seperated list without brackets:\n>"))
        vector2 = clean_vector(input("   Input vector 2 as a comma seperated list without brackets:\n>"))
        check_dimensions(vector1,vector2)
        answer = cross_product(vector1, vector2)
        if answer == None:
            output("Error: The cross product is only defined in three dimensional space")
        else:
            output(f"The cross product of vector 1 and vector 2 is {answer}")
    
    elif choice == "7":
        #determinant solver
        
        matrix = square_matrix_maker()
        
        #formats answer
        answer = "the determinant of\n"
        for row in matrix:
            answer = answer + "   " + str(row) + "\n"
        answer = answer + f"   is {determinant_solver(matrix)}"
        output(answer)
    
    elif choice == "r":
        degree_switch = False
    
    elif choice == "d":
        degree_switch = True
    