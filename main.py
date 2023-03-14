from clasificacionLOL import clasificacion
from fifa import clasificacionFIFA

path = "high_diamond_ranked_10min.csv"

path2 = "CompleteDataset.csv"

# se hace clasificacion de partidas de league of legends
print('\n Clasificacion de partidas de League of Legends \n')
clasificacion = clasificacion(path)


print('\n Clasificacion de partidas de FIFA \n')
clasiFifa = clasificacionFIFA(path2)
