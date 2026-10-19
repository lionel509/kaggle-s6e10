import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
t = np.load("logs/pub124_best_test.npy"); te = pd.read_csv("data/test.csv", usecols=["id"])
p = (rankdata(t) - .5) / len(t)
b = pd.read_csv("subs/b26_k20all.csv")
print("spearman vs b26", spearmanr(p, b.satisfaction.values)[0], len(t), len(b))
pd.DataFrame({"id": te.id, "satisfaction": p}).to_csv("subs/b27_lr_tfm_pub124.csv", index=False)
