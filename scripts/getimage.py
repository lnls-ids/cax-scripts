
import epics
import numpy as np
import matplotlib.pyplot as plt

ny = epics.caget("CAX:B:BASLER01:image1:ArraySize0_RBV")
nx = epics.caget("CAX:B:BASLER01:image1:ArraySize1_RBV")

print(f"(nx, ny) = ({nx}, {ny})")

f = epics.caget("CAX:B:BASLER01:image1:ArrayData")
screen_pos = epics.caget("CAX:B:PP01:E.RBV")
file_name = f"CAUSTICA_01_MS_{screen_pos}_mm"

print(file_name)

g = f.reshape((nx, ny))

np.save(file_name, g)

fig, ax = plt.subplots()
ax.imshow(g)
plt.show()