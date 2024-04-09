# Carla Auto Parking

APA demo in Carla Simulator.

```{bash}
git clone git@github.com:houchangmeng/CarlaAutoParking.git  
```

```{bash}
# Run CARLA simulator
# Your_Carla_Folder/CarlaUE4.sh

# Run PNC
python3 ./CarlaAutoParking/statemachine_carla.py

# Then, Click a feasible parking slot in matplotlib window.
```

```{bash}
# Run Parking slot detection
python3 ./CarlaAutoParking/parking_rectangle_detection.py
```

Demo videos saved in [./videos](./videos).

---
carla 0.9.13
