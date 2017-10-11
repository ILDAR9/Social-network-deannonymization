integers = open('/media/ildar/ILDAR-HHD/Downloads/big_data/data/facebook/facebook_combined.csv', 'r')
largestInt = float('-inf')
mingestInt = float('inf')

fst_line = True
s = set()
for line in integers:
    c = line.split()
    if fst_line:
        fst_line = False
        continue
    a, b = int(c[0]), int(c[1])
    s.add(a)
    s.add(b)
    if largestInt < a:
        largestInt = a
    if largestInt < b:
        largestInt = b
    if mingestInt > a:
        mingestInt = a
    if mingestInt > b:
        mingestInt = b

print('min = %d' % mingestInt)
print('max vertex num = %d' % largestInt)
print('real vertex count = %d' % len(s))