
a=input("enter the string=")
v ="aeiou"

a=a.lower()

d={}
for i in a:
    if i in v:
        d[i]=count+1