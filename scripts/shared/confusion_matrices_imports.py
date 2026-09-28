import os
p = "/content/drive/MyDrive/A3IDPS_COLAB/models/nsl_v2"
print(sorted(f for f in os.listdir(p) if "kd_feedback" in f))
