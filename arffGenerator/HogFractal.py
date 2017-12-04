information = "@RELATION gaborfractal\n\n"
for i in range(1, 1 + 81 + 1):
    information += "@ATTRIBUTE a" + str(i) + " REAL\n"
information += "@ATTRIBUTE class {1,2,3,4,5}\n\n"
information += "@DATA\n"

labels = open("/home/emre/Desktop/labels.txt", "r")
file = open("/home/emre/Desktop/Project/arff files/hogfractal.arff", "w+")
file.write(information)

hogName = "/home/emre/Desktop/Feature Vectors/HOG/SCUT-FBP-"
fractalName = "/home/emre/Desktop/Feature Vectors/FRACTAL/SCUT-FBP-"

for i in range(1, 501):

    foo2 = open(fractalName + str(i) + ".txt", "r")
    token = foo2.readline()
    float_number = float(token)
    file.write(str(float_number) + ", ")
    foo2.close()

    foo = open(hogName + str(i) + ".txt", "r")
    for j in range(81):
        token = foo.readline()
        float_number = float(token)
        file.write(str(float_number) + ", ")
    foo.close()


    label = int(labels.readline())
    file.write(str(label) + "\n")

file.close()
labels.close()