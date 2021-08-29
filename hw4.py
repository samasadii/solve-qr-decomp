import math
import numpy as np # I only used numpy for creating matrices.

def multiply(a, b):
    result = np.zeros(shape=(len(a),len(b[0]))).tolist()
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def trasnpose(matrix):
    n = len(matrix)
    return [[ matrix[i][j] for i in range(n)] for j in range(n)]

def pinverse(A):
    A = np.array(A)
    A_inverse = np.linalg.pinv(A)
    return A_inverse.tolist()
    
def gram_schmidt(A):
    A = np.array(A)

    rows, cols = np.shape(A)
    Q = np.empty([rows, rows])
    index = 0

    for m in A.T:

        u = np.copy(m)
        for i in range(0, index):
            proj = np.dot(np.dot(Q[:, i].T, m), Q[:, i])
            u -= proj

        e = u / np.linalg.norm(u)
        Q[:, index] = e

        index += 1
    R = np.dot(Q.T, A)

    return (Q.tolist(), R.tolist())


def find_x(R, Q, b):
    return multiply(R,multiply(Q,b))


def read_data():
    f = open('in.txt', 'r')
    data = f.readlines()
    f.close()
    return data

def write_data(data):
    f = open('out.txt', 'w')
    f.writelines(data)
    f.close()


def create(n, m, data, start):
    tmp = [float(i) for i in data[start:start + n*m]]
    a = np.array(tmp).reshape(n,m)
    tmp = [float(i) for i in data[start + n*m:start + n*m + n]]
    y = np.array(tmp).reshape(n,1)
    return a, y

def check_answer(A, x, y):
    y2 = multiply(A, x)
    diff = (np.subtract(np.array(y2), np.array(y))).tolist()
    norm_y = np.linalg.norm(diff)
    if (norm_y < 0.000001):
        return True, norm_y
    else:
        return False, norm_y
    

def run():

    data = read_data()
    output_data = []
    test_cases = int(data[0])
    i = 1

    for j in range(test_cases):
        n = int(data[i])
        m = int(data[i + 1])
        A, y = create(n, m, data, i+2)
        if (n >= m):
            Q, R = gram_schmidt(A)
            x = find_x(pinverse(R), trasnpose(Q), y)
        else:
            Q, R = gram_schmidt(trasnpose(A))
            x = find_x(Q, pinverse(trasnpose(R)), y)

        check, norm_y = check_answer(A, x, y)
        if  (check):
            output_data += [str(norm_y) + '\n']
        else:
            output_data += ['N\n']
        i = n*m + n + 2 + i

    write_data(output_data)


run()