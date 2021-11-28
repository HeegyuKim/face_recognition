from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap
import umap.plot

def save_pca_plot(data, labels):
  tf = PCA(2).fit_transform(data)
  df = pd.concat([
                  pd.DataFrame(tf, columns=["x", "y"]), 
                  pd.DataFrame(labels, columns=["label"])]
                 , axis=1)
  return sns.scatterplot(data=df, x="x", y="y", hue="label",
                         alpha=.2)

def save_umap_plot(data, labels):
  mapper = umap.UMAP().fit(data)
  return umap.plot.points(mapper, labels=labels)

save_umap_plot(np.random.rand(1500, 5), np.array([1, 2, 3] * 500))
plt.savefig("./tmp.png")