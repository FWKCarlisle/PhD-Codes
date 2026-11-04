import matplotlib.pyplot as plt

temps = []
remenances = []
temps_1 = []
integration_1 = []
temps_2 = []
integration_2 = []

with open(r"PhD-Codes\Analysis code\Hysteresis\data_for_ppt\remenance.txt") as file:
    for line in file:
        if line.strip():
            file_name, temp, remenance = line.split(",")[0], float(line.split(",")[1]), float(line.split(",")[2])
            # print(temp, remenance)
            temps.append(temp)
            remenances.append(remenance)

with open(r"PhD-Codes\Analysis code\Hysteresis\data_for_ppt\mvt.txt") as file:
    for line in file:
        if line.strip():
            file_name, temp, integration = line.split(",")[0], float(line.split(",")[1]), float(line.split(",")[2])
            if file_name == "Step":
                temps_1.append(temp)
                integration_1.append(integration)
            elif file_name == "Step_2":
                temps_2.append(temp)
                integration_2.append(integration)   
            else:
                print("Error in file name")


plt.title('Remenance vs Temperature')
plt.scatter(temps, remenances, marker='o', color='blue', label='Data Points')
plt.scatter(temps_1, integration_1, marker='o', color='red', label='MvT 1')
plt.scatter(temps_2, integration_2, marker='o', color='green', label='MvT 2')
plt.xlabel('Temperature (K)')
plt.ylabel('"Remenance" (mT)')
plt.legend()
plt.show()