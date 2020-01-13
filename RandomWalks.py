import numpy as np
import matplotlib.pyplot as plt
import random
import math

def simple_sample(trials = 100):

    deadEnds = 0

    data = []
    RData = []
    effData = []

    sizes = np.arange(0,40,1)

    for size in sizes:
        n = size * 3
        deadEnds = 0
        for t in range(trials):

            space = np.zeros((n,n))   
            x = n//2
            y = n//2

            while (x > 0) and (x < n-1) and (y > 0) and (y < n-1):
                # Check for dead end and make a random move.
                space[x][y] = 1
                if space[x-1][y] and space[x+1][y] and space[x][y-1] and space[x][y+1]:
                    deadEnds += 1
                    break
                r = random.randrange(1, 5)
                if   (r == 1) and (not space[x+1][y]):
                    x += 1
                elif (r == 2) and (not space[x-1][y]):
                    x -= 1
                elif (r == 3) and (not space[x][y+1]):
                    y += 1
                elif (r == 4) and (not space[x][y-1]):
                    y -= 1

        prc = (deadEnds*100 / trials)
        data.append((size+1,100 - prc))

        R = calculate_Rsquared(size)
        RData.append((R,deadEnds))

        effData.append(( size,((trials - deadEnds) / trials) * 100))


    shift = 0
    for elts in data[1:]:
        plt.bar(elts[0],elts[1], color=(1-shift,0.2,.3+(shift/1.4),1))
        shift += 0.016

    plt.xlim(1,61)
    plt.xlabel('Length')
    plt.ylabel('Probability')
    plt.show()

    plt.cla()
    shift = 0
    for elts in effData[1:]:
        plt.scatter(elts[0],elts[1], color=(1-shift,0.2,.3+(shift/1.4),1))
        shift += 0.016

    plt.xlabel('Efficiency (%)')
    plt.ylabel('Size')
    plt.show()   

def calculate_R(coord1, coord2):
    x1 = coord1[0]
    x2 = coord2[0]
    y1 = coord1[1]
    y2 = coord2[1]

    return math.hypot(x2 - x1, y2 - y1)

def valid_moves(a):
    '''
    array order is "left,right,up,down"
    '''

    #First move; Obsolete?
    if a == [0.0,0.0,0.0,0.0]:
        return ['left','right','up','down'][random.randint(0,3)]

    idx = random.randint(0,3)

    while a[idx] == 1:
        idx = random.randint(0,3)

    #return a string; in case the order needs printed.
    if idx == 0:
        return 'left'        
    elif idx == 1:
        return 'right'
    elif idx == 2:
        return 'up'
    elif idx == 3:
        return 'down'

def calculate_Rsquared(N):
    v = 0.75
    a = 1.1

    return (a * ((N) ** (2 * v)))

def calculate_DeltaR(R):
    
    return (math.sqrt((R ** 4) - (R) ** 2)) / (R ** 2)

def rosenbluth(n = 20, trials = 100):

    data = []
    verify = []
    RData = []
    effData = []

    
    #n = 20
    #trials = 100

    #initialize the counting variables
    successful = 0
    R = 0

    sn = 0
    wn = 0
    rosen_sizes = np.arange(0,160,1)

    for size in rosen_sizes:
        successful = 0
        goal = size
        total = 0

        for t in range(trials): # reset params for each trial
            weight = 1
            length = 0  
            wn = 1  

            space = np.zeros((n,n))   
            x = n//2
            y = n//2
            startPoint = (x,y)

            #Rosenbluth looks for length completion
            #length == goal is an exit condition
            while (x > 0) and (x < n-1) and (y > 0) and (y < n-1) and (length < goal):
                #place the one for each coordinate
                space[x][y] = 1
                
                first_move = True
                if first_move:
                    move = valid_moves([0,0,0,0])
                    first_move = False

                atomosphere = [space[x-1][y], space[x+1][y], space[x][y+1], space[x][y-1]]
                #print(atomosphere)
                if atomosphere == [1,1,1,1]:
                    break
                else:
                    a = sum([abs(x - 1) for x in atomosphere])
                
                #returns a random available move
                move = valid_moves(atomosphere)

                length += 1
                sn += 1
                weight = length / 5
                weight *= a
                wn += weight
                #print(wn)

                #apply the the move
                if  move == 'right':
                    space[x+1][y] = 1
                    x += 1               
                elif  move == 'left':
                    space[x-1][y] = 1
                    x -= 1
                elif  move == 'up':
                    space[x][y+1] = 1
                    y += 1
                elif  move == 'down':
                    space[x][y-1] = 1
                    y -= 1

                
                if length == goal:
                    successful += 1
                    total += 1
                    #endPoint = (x,y)
                    R = calculate_DeltaR (calculate_Rsquared(size))
                    
                    #print(space)
                else:
                    continue
        #Efficiency
        effData.append((size,(total * wn) / trials))

        #<R^2>
        RData.append((successful,R))

        #Obsolete weight plot
        verify.append((size+1,wn))

        #print(successful)
        data.append((size+1,successful*100 // trials))

    inc = 1/(len(rosen_sizes))
    shift = 0
    for elts in data:
        plt.bar(elts[0],elts[1], color=(1 - shift, 0, 0 + shift, 1), zorder=1)
        shift += inc

    #remove first point, always zero
    '''
    shift = 0
    for elts in verify[1:]:
        if elts[0] != 0:
            plt.scatter(elts[0],100 - elts[1]/10,color=str(0 + shift), zorder=2)
            shift += inc
    '''
    plt.ylabel('Probability')
    plt.xlabel('Length')
    plt.xlim(0,200)
    plt.ylim(0,105)
    plt.show()

    plt.cla()

    shift = 0
    for elt in effData[1:]:
        plt.scatter(elt[0],elt[1],color=(1 - shift, 0, 0 + shift, 1), zorder=1)
        shift += inc
        
    plt.xlabel('Number of Walks')
    plt.ylabel('<R^2>')
    plt.show()
   
def main():
    '''
    simple_sample()
    print()
    print('break')
    print()
    rosenbluth() 
    last = True
    '''

if __name__ == "__main__":
    main()
