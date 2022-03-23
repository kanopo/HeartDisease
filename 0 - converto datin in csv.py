import os

my_files = os.listdir("./0 - dati grezzi")

for file_to_convert in my_files:
    print(file_to_convert)
    f1 = open("./0 - dati grezzi/" + str(file_to_convert), "r")
    data = f1.readlines()
    f1.close()

    data = ''.join(data)
    data = data.replace('\n', ' ').replace('name ', 'name\n')
    data = data.replace(" ", ",")
    print(data)

    f1 = open("./1 - dati in csv/" + str(file_to_convert) + ".csv", "w+")
    f1.write(data)
    f1.close()