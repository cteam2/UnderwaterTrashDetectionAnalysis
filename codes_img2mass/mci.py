import subprocess
import re
import sys

# Run predict_mass.py and capture its output
filename,l,w = sys.argv[1:]
result = subprocess.run(['python3.5', 'predict_mass.py', filename, l, w], stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
output = result.stdout.decode('utf-8')

# Use regular expressions to extract the mass value from the output
match = re.search(r'probably weighs about (\d+\.?\d*) grams', output)
if match:
    mass = match.group(1)  # This is the mass value as a string

    # Or if mci.py can be used as a module, you can import and call a function directly
    # from mci import calculate_something
    # calculate_something(float(mass))
else:
    print("Mass value could not be found in the output.")

mass= float(mass)
material_index= 1
materials= ["Fabric", "Metal", "Paper", "Plastic", "Rubber", "Wood"]
material= materials[material_index]
recycle_rates= [0.15, 0.98, 0.6, 0.55, 0.25, 0.18]
reuse_rates= [0.1, 0, 0, 0, 0.2, 0.15]

Cr= recycle_rates[material_index] * (mass/1000)
Cu= reuse_rates[material_index] * mass
Fx= 0.8
V= mass

W0= mass*(1-Cr-Cu)
Wc= mass*(1-0.7)*Cr
Wf= 0

W= W0 + (Wc+Wf)/2

LFI= (V+W)/(2*mass)

MCI = 0.999 if (1 - (LFI * Fx)) > 0.999 else (1 - (LFI * Fx))

print(MCI)
