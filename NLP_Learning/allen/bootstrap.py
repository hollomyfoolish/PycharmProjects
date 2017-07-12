def add(a, b):
    return a + b

print add.__name__
add.a = [1, 2, 3]
if __name__ == '__main__':
    print 'Hello Python'
    print add(1, 2)
    print add.a
