import numpy as np
import pandas
import matplotlib.pyplot as plt
"""
    Generiert einen Fit für die von Ansys Optic Studio generierten Daten zum Wellenfrontfehler für das Feld 1 mit den in Dokument angegebenen Parametern.
    Die wahre Funktion für den Fit ist 
        W000 + W020 ρ^2 + W220 η^2 ρ^2 + W040 ρ^4 + 
        W111 η ρ Cos[θ] + 
        W311 η^3 ρ Cos[θ] + 
        W131 η ρ^2 Cos[θ] + 
        W222 η^2 ρ^2 Cos[θ]^2
    Für das Feld 1 ist der Eintrittswinkel 0° und alle Strahlen treffen auf den Mittelpunkt dem nach ist η=0 und der Wellenfrontfehler wird zu
        W000 + W020 ρ^2 + W040 ρ^4
    Der Kosinusterm fällt dann ebenso. Im folgenden wird ρ=Sqrt(x^2+y^2) verwendet.
"""
df = pandas.read_csv('wavefrontfield1.txt',delimiter='\t',encoding='utf-16',decimal=',',header=None) # Importiert die von OpticStudio generierte Data als CSV, wobei Manuell der Header der Datei entfernt wurde.
dataArray = df.to_numpy() #Konvertiert Pandas Dataframe in NumPy Array

#Generiert ein äquidistantes Gitter von -1 bis 1 mit einer Schrittweite von jeweils den x und y Schritten
nx,ny= dataArray.shape
xSpace = np.linspace(-1,1,nx)
ySpace = np.linspace(-1,1,ny)
X,Y=np.meshgrid(xSpace , ySpace)

#Generiert eine Maske, sodass die nur die Werte =/= Null betrachtet werden. Der Radius wird hier auf 1 gesetzt wie man aus dem Wellenfrontfehler Diagramm entnehmen kann.
mask=np.sqrt(X**2 + Y**2)<=0.99
maskX=X[mask]
maskY=Y[mask]
maskData=dataArray[mask]
R2_masked = maskX**2 + maskY**2

#Initialisierung der Design-Matrix um einen Least-Squares-Fit zu dem Daten auszuführen
A = np.column_stack([
    np.ones_like(R2_masked),# W000
    R2_masked, # W020
    R2_masked**2, # W040
    ])

#Least-Squares-Fit an die Daten
coeffs, residuals, rank, s = np.linalg.lstsq(A, maskData, rcond=None)

#Zuweisung der Koeffizienten
W000, W020, W040 = coeffs
print(f"Fit Parameter: W000 = {W000:.4f};W020 = {W020:.4f};W040 = {W040:.4f}")
#Fit Funktion
fit = W000 + W020*(X**2 + Y**2) + W040*(X**2 + Y**2)**2

#Plot
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(maskX, maskY, maskData,c=maskData, cmap="cividis")
ax1.set_title("Original Daten")
ax1.view_init(elev=45, azim=45)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("W(x,y)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(maskX, maskY, fit[mask], c=fit[mask],cmap="magma")
ax2.set_title(f"Fit der Daten")
ax2.view_init(elev=45, azim=45)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("W(x,y)")
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()