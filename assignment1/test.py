result=[[1,2,3],[4,5,3424234],[1,2,3]]
max=0
for i in range(len(result)):
    for j in range(len(result[i])):
        if result[i][j]>max:
            max=result[i][j]
            max_i=i
            max_j=j
print(max_i,max_j)