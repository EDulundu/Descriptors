length = 81 + 1
information = "@RELATION hog\n\n"
for i in range(1,length):
    information += "@ATTRIBUTE a" + str(i) + " REAL\n"
information += "@ATTRIBUTE class {1,2,3,4,5}\n\n"
information += "@DATA\n"

labels = open("/home/emre/Desktop/labels.txt", "r")
read = "/home/emre/Desktop/Feature Vectors/HOG/SCUT-FBP-"
result = "/home/emre/Desktop/Project/arff files/hog.arff"

file = open(result, "w+")
file.write(information)

for i in range(1, 501):

    foo = open(read + str(i) + ".txt", "r")
    for j in range(length-1):
        token = foo.readline()
        float_number = float(token)
        file.write(str(float_number) + ", ")
    foo.close()

    label = int(labels.readline())
    file.write(str(label) + "\n")

file.close()
labels.close()